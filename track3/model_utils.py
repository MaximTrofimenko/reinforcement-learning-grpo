"""Model loading and fixed decoding settings shared by training and evaluation."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from task import prompt_for


MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MAX_NEW_TOKENS = 64
SYSTEM_MESSAGE = "Solve the arithmetic puzzle. Follow the requested answer format exactly."


def load_base():
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=dtype)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def prompt_tokens(tokenizer, task: dict, device: str):
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt_for(task)},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    )
    return encoded["input_ids"].to(device)


@torch.no_grad()
def generate_one(model, tokenizer, task: dict, device: str, sample: bool = False):
    ids = prompt_tokens(tokenizer, task, device)
    generated = model.generate(
        ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=sample,
        temperature=0.8 if sample else None,
        top_p=0.95 if sample else None,
        pad_token_id=tokenizer.eos_token_id,
    )
    completion_ids = generated[0, ids.shape[1] :]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return text, int(completion_ids.shape[0])
