from time import perf_counter
import argparse
from typing import Generator, Tuple

import numpy as np
import coremltools as ct
from coremltools.models import MLModel
from transformers import AutoTokenizer, PreTrainedTokenizer

from export import METADATA_TOKENIZER


def load(model_path: str) -> Tuple[MLModel, AutoTokenizer]:
    """Load a Core ML model and corresponding tokenizer."""
    model: MLModel = MLModel(model_path, optimization_hints={ 'specializationStrategy': ct.SpecializationStrategy.FastPrediction })
    description = model.get_spec().description
    if METADATA_TOKENIZER not in description.metadata.userDefined:
        raise ValueError("Model metadata does not contain tokenizer path.")
    tokenizer_path: str = description.metadata.userDefined[METADATA_TOKENIZER]
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

def sample(logits: np.ndarray) -> int:
    """Perform greedy decoding on the logits array to get the next token."""
    return int(np.argmax(logits[0][-1], axis=-1))

def inference(model: ct.models.MLModel, input_ids: np.ndarray, num_past_tokens: int, kv_cache_state) -> np.ndarray:
    """Perform inference with the given model and input data."""
    causal_mask: np.ndarray = np.triu(
        np.full(
            (1, 1, input_ids.shape[-1], num_past_tokens + input_ids.shape[-1]),
            fill_value=-np.inf if num_past_tokens == 0 else 0,
        ),
        k=1,
    ).astype(np.float16)
    outputs: dict[str, np.ndarray] = model.predict(
        data={"inputIds": input_ids, "causalMask": causal_mask},
        state=kv_cache_state,
    )
    return outputs["logits"]

def get_next_token(model: ct.models.MLModel, prompt_tokens: np.ndarray) -> Generator[int, None, None]:
    """Generate a sequence of tokens with naive greedy decoding."""

    kv_cache_state = model.make_state()
    logits: np.ndarray = inference(
        model, 
        input_ids=prompt_tokens, 
        num_past_tokens=0, 
        kv_cache_state=kv_cache_state
    )
    token: int = sample(logits=logits)
    num_past_tokens: int = prompt_tokens.shape[-1]

    while True:
        yield token
        logits: np.ndarray = inference(
            model,
            input_ids=np.array([[token]], dtype=np.int32),
            num_past_tokens=num_past_tokens,
            kv_cache_state=kv_cache_state,
        )
        token: int = sample(logits=logits)
        num_past_tokens += 1

def generate(
    model: ct.models.MLModel,
    prompt: str,
    tokenizer: PreTrainedTokenizer,
    max_new_tokens: int,
) -> str:
    messages = [
        {
            "role": "user", 
            "content": prompt,
        }
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    prefill_start = perf_counter()
    prompt_tokens: np.ndarray = tokenizer(formatted_prompt, return_tensors="np").input_ids
    extend_tokens: list[int] = []
    for i, token in enumerate(get_next_token(model, prompt_tokens=prompt_tokens.astype(np.int32))):
        extend_tokens.append(token)
        if i == 0:
            prefill_end = perf_counter()
            decode_start = prefill_end
            ttft = (prefill_end - prefill_start) * 1000
            print(f"Time to first token: {ttft:.2f} seconds")
        if token == tokenizer.eos_token_id or i + 1 == max_new_tokens:
            decode_end = perf_counter()
            decode_tps = i / (decode_end - decode_start)
            print(f"decode throughput: {decode_tps:.2f} tok/s")
            break
    return tokenizer.decode(prompt_tokens[0].tolist() + extend_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()
    model, tokenizer = load(args.model_path)
    extend_text: str = generate(
        model,
        prompt=args.prompt,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
    )
    print(extend_text)
