"""Small on-policy GRPO run with LoRA for Qwen3-0.6B.

One prompt is sampled per step. Four completions are compared with a group
baseline; there is one optimizer update per group and a reference KL penalty.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model

from evaluate import evaluate, read_tasks
from model_utils import MAX_NEW_TOKENS, load_base, prompt_tokens
from task import reward


HERE = Path(__file__).resolve().parent
GROUP_SIZE = 4
DEFAULT_SEED = 20261114
LEARNING_RATE = 5e-5
DEFAULT_KL_BETA = 0.02


def completion_mask(sequences: torch.Tensor, prompt_length: int, eos_id: int) -> torch.Tensor:
    completion = sequences[:, prompt_length:]
    eos = completion.eq(eos_id)
    prior_eos = torch.cat(
        [torch.zeros_like(eos[:, :1]), eos[:, :-1].int().cumsum(dim=1) > 0], dim=1
    )
    return ~prior_eos


def token_logprobs(model, sequences: torch.Tensor, prompt_length: int, mask: torch.Tensor) -> torch.Tensor:
    attention = torch.cat(
        [torch.ones((sequences.shape[0], prompt_length), device=sequences.device, dtype=torch.long), mask.long()],
        dim=1,
    )
    logits = model(input_ids=sequences, attention_mask=attention, use_cache=False).logits
    next_logits = logits[:, prompt_length - 1 : -1, :].float()
    next_tokens = sequences[:, prompt_length:]
    return torch.log_softmax(next_logits, dim=-1).gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)


def train_step(model, tokenizer, device: str, task: dict, optimizer, kl_beta: float) -> dict:
    model.eval()
    prompt = prompt_tokens(tokenizer, task, device)
    with torch.no_grad():
        sequences = model.generate(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            num_return_sequences=GROUP_SIZE,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    prompt_length = prompt.shape[1]
    mask = completion_mask(sequences, prompt_length, tokenizer.eos_token_id)
    responses = [
        tokenizer.decode(row[prompt_length :], skip_special_tokens=True)
        for row in sequences
    ]
    rewards = np.array([reward(task, response) for response in responses], dtype=np.float32)
    std = float(rewards.std())
    advantages = (rewards - rewards.mean()) / (std + 1e-4) if std > 0 else np.zeros_like(rewards)
    adv = torch.tensor(advantages, device=device)[:, None]

    model.train()
    policy_lp = token_logprobs(model, sequences, prompt_length, mask)
    with model.disable_adapter(), torch.no_grad():
        reference_lp = token_logprobs(model, sequences, prompt_length, mask)
    delta = reference_lp - policy_lp
    kl = torch.exp(delta.clamp(max=20)) - delta - 1.0
    ratio = torch.exp(policy_lp - policy_lp.detach())
    objective = ratio * adv - kl_beta * kl
    loss = -(objective * mask).sum() / mask.sum()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "task_id": task["id"],
        "mean_reward": float(rewards.mean()),
        "positive_samples": int(rewards.sum()),
        "mixed_group": bool(std > 0),
        "mean_kl": float((kl.detach() * mask).sum().item() / mask.sum().item()),
        "mean_tokens": float(mask.sum(dim=1).float().mean().item()),
        "loss": float(loss.detach().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--eval-every", type=int, default=40)
    parser.add_argument("--kl-beta", type=float, default=DEFAULT_KL_BETA)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tokenizer, base, device = load_base()
    model = get_peft_model(
        base,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        ),
    )
    model.to(device)
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LEARNING_RATE)
    train_tasks = read_tasks("train")
    dev_tasks = read_tasks("dev")
    log_path = args.output / "training_log.jsonl"
    if log_path.exists():
        raise SystemExit("Training log already exists. Use a new output directory for a new run.")
    with log_path.open("w", encoding="utf-8") as log:
        for step in range(1, args.steps + 1):
            task = train_tasks[int(rng.integers(0, len(train_tasks)))]
            metrics = train_step(model, tokenizer, device, task, optimizer, args.kl_beta)
            metrics["step"] = step
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            if step % 10 == 0:
                print(f"step {step}/{args.steps}: reward {metrics['mean_reward']:.2f}, KL {metrics['mean_kl']:.4f}, mixed {metrics['mixed_group']}", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                model.eval()
                evaluate(model, tokenizer, device, dev_tasks, args.output / f"dev_step_{step}.json")
                model.save_pretrained(args.output / "adapter")
    (args.output / "training_config.json").write_text(
        json.dumps(
            {
                "model": "Qwen/Qwen3-0.6B",
                "steps": args.steps,
                "group_size": GROUP_SIZE,
                "seed": args.seed,
                "learning_rate": LEARNING_RATE,
                "kl_beta": args.kl_beta,
                "max_new_tokens": MAX_NEW_TOKENS,
                "eval_every": args.eval_every,
                "device": device,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
