import logging
import warnings
from typing import Any, List, Tuple

import coremltools as ct
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache

warnings.filterwarnings("ignore")
logging.getLogger("coremltools").setLevel(logging.ERROR)
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
MODEL_ID: str = "meta-llama/Llama-3.2-1B-Instruct"
METADATA_TOKENIZER: str = "co.huggingface.exporters.name"


class SliceUpdateKeyValueCache(Cache):
    def __init__(
        self,
        shape: Tuple[int, ...],
        device="cpu",
        dtype=torch.float32,
    ) -> None:
        """KV cache of shape (#layers, batch_size, #kv_heads, context_size, head_dim)."""
        super().__init__(layers=[])
        self.past_seen_tokens: int = 0
        self._max_cache_len: int = shape[-2]
        self.k_cache: torch.Tensor = torch.zeros(shape, dtype=dtype, device=device)
        self.v_cache: torch.Tensor = torch.zeros(shape, dtype=dtype, device=device)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update key/value cache tensors for slice [begin, end).

        Current Transformers Llama calls:
            past_key_values.update(key_states, value_states, layer_idx)

        It does not pass cache_position anymore.
        """
        begin = self.past_seen_tokens
        end = begin + key_states.shape[-2]
        self.k_cache[layer_idx, :, : key_states.shape[1], begin:end, :] = key_states
        self.v_cache[layer_idx, :, : value_states.shape[1], begin:end, :] = value_states

        return (
            self.k_cache[layer_idx, :, :, :end, :],
            self.v_cache[layer_idx, :, :, :end, :],
        )

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        return self.past_seen_tokens

    def get_max_cache_shape(self, layer_idx: int | None = 0) -> int:
        return self.max_cache_len

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        kv_length = self.past_seen_tokens + query_length
        kv_offset = 0
        return kv_length, kv_offset

    @property
    def max_cache_len(self) -> int:
        return self._max_cache_len

    @property
    def is_compileable(self) -> bool:
        return False

    @property
    def is_sliding(self) -> list[bool]:
        return [False] * self.k_cache.shape[0]


class StatefulModelForCausalLM(torch.nn.Module):
    def __init__(self, model_path: str, max_context_size: int = 2048, batch_size: int = 1) -> None:
        super().__init__()

        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)

        # Register KV cache buffers to be recognized as Core ML states
        config = self.model.config
        self.kv_cache_shape: Tuple[int, ...] = (
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_context_size,
            config.hidden_size // config.num_attention_heads,
        )
        self.kv_cache = SliceUpdateKeyValueCache(shape=self.kv_cache_shape)
        self.register_buffer("keyCache", self.kv_cache.k_cache)
        self.register_buffer("valueCache", self.kv_cache.v_cache)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Compute past seen tokens used for updating key/value cache slices
        self.kv_cache.past_seen_tokens = causal_mask.shape[-1] - input_ids.shape[-1]
        return self.model(
            input_ids,
            attention_mask=causal_mask,
            past_key_values=self.kv_cache,
            use_cache=True,
        ).logits


def export() -> None:
    # Construct model from transformers and trace to TorchScript
    max_context_size: int = 2048
    torch_model = StatefulModelForCausalLM(MODEL_ID, max_context_size=max_context_size)
    torch_model.eval()
    input_ids: torch.Tensor = torch.zeros((1, 2), dtype=torch.int32)
    causal_mask: torch.Tensor = torch.zeros((1, 1, 2, 5), dtype=torch.float32)
    traced_model = torch.jit.trace(torch_model, [input_ids, causal_mask])
    kv_cache_shape = torch_model.kv_cache_shape
    del torch_model

    # Convert traced TorchScript to Core ML format
    query_length = ct.RangeDim(lower_bound=1, upper_bound=max_context_size, default=1)
    end_step_dim = ct.RangeDim(lower_bound=1, upper_bound=max_context_size, default=1)
    inputs: List[ct.TensorType] = [
        ct.TensorType(shape=(1, query_length), dtype=np.int32, name="inputIds"),
        ct.TensorType(
            shape=(1, 1, query_length, end_step_dim),
            dtype=np.float16,
            name="causalMask",
        ),
    ]
    outputs: List[ct.TensorType] = [ct.TensorType(dtype=np.float16, name="logits")]
    states: List[ct.StateType] = [
        ct.StateType(
            wrapped_type=ct.TensorType(shape=kv_cache_shape, dtype=np.float16),
            name="keyCache",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(shape=kv_cache_shape, dtype=np.float16),
            name="valueCache",
        ),
    ]

    # Convert model with FP16 precision
    mlmodel_fp16: ct.models.MLModel = ct.convert(
        traced_model,
        inputs=inputs,
        outputs=outputs,
        states=states,
        minimum_deployment_target=ct.target.iOS18,
        skip_model_load=True,
    )
    mlmodel_fp16._spec.description.metadata.userDefined.update({METADATA_TOKENIZER: MODEL_ID})
    del traced_model
    mlmodel_fp16.save("./models/StatefulLlama3.2FP16.mlpackage")

    # Block-wise quantize model weights to int4
    op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int4",
        granularity="per_block",
        block_size=32,
    )
    config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
    mlmodel_int4 = ct.optimize.coreml.linear_quantize_weights(mlmodel_fp16, config=config)
    mlmodel_int4._spec.description.metadata.userDefined.update({METADATA_TOKENIZER: MODEL_ID})
    del mlmodel_fp16
    mlmodel_int4.save("./models/StatefulLlama3.2Int4.mlpackage")


if __name__ == "__main__":
    export()
