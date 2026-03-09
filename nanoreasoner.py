"""
nanoReasoner: Teaching Language Models to Reason via GRPO
=========================================================
A single-file, educational implementation of Group Relative Policy Optimization
(GRPO) for training language models to solve math problems step-by-step.

Inspired by Karpathy's nanoGPT/nanochat/autoresearch lineage.
Implements the core algorithm from DeepSeek R1 (arXiv:2501.12948) in ~900 lines.

The idea: take a pretrained language model, show it math problems, let it generate
multiple attempts, reward the correct ones, penalize the wrong ones. Over hundreds
of iterations, the model learns to "think" — producing intermediate reasoning steps
before arriving at an answer. No reward model needed. No human feedback. Just math.

Usage:
    # Fine-tune Qwen2.5-1.5B with simplified GRPO (nano mode)
    python nanoreasoner.py --model Qwen/Qwen2.5-1.5B --mode nano --G 4

    # Full GRPO with clipping and KL penalty
    python nanoreasoner.py --model Qwen/Qwen2.5-0.5B-Instruct --mode full --G 8

    # Train a tiny GPT from scratch (educational, limited reasoning)
    python nanoreasoner.py --model scratch --depth 8 --mode nano --G 16

    # Evaluate a saved checkpoint
    python nanoreasoner.py --eval --checkpoint runs/latest/model.pt

Dependencies: torch, transformers, datasets, matplotlib
Optional: bitsandbytes (8-bit optimizer), wandb (cloud logging)

Reference: https://arxiv.org/abs/2501.12948 (DeepSeek R1)
           https://arxiv.org/abs/2402.03300 (DeepSeekMath / GRPO)
           https://github.com/karpathy/nanochat (nanochat GRPO scaffolding)
"""

import os
import re
import json
import time
import math
import random
import argparse
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    """Model architecture config. Following nanochat's 'single dial' design:
    depth controls everything. model_dim = depth * 64."""
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512          # depth * 64
    vocab_size: int = 32000
    sequence_len: int = 512
    dropout: float = 0.0

@dataclass
class GRPOConfig:
    """GRPO algorithm hyperparameters."""
    mode: str = "nano"         # "nano" | "standard" | "full"
    G: int = 8                 # group size: completions per prompt
    epsilon: float = 0.2       # PPO clip ratio (standard/full modes)
    beta: float = 0.001        # KL coefficient (full mode only)
    temperature: float = 1.0   # sampling temperature
    top_k: int = 50            # top-k sampling
    max_completion_len: int = 512
    ref_update_every: int = 400  # steps between reference model updates (full mode)

@dataclass
class TrainConfig:
    """Training loop configuration."""
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    mode: str = "nano"
    G: int = 8
    depth: int = 8             # for scratch model only
    lr: float = 5e-6
    weight_decay: float = 0.1
    warmup_steps: int = 50
    total_steps: int = 500
    batch_size: int = 4        # prompts per step
    grad_accum: int = 4        # effective batch = batch_size * grad_accum
    eval_every: int = 50
    save_every: int = 50
    sample_every: int = 25
    log_every: int = 5
    run_name: str = ""
    seed: int = 42
    use_8bit_optim: bool = False
    gradient_checkpointing: bool = True
    git_tracking: bool = True
    device: str = "cuda"
    dtype: str = "bfloat16"
    eval_only: bool = False
    checkpoint: str = ""

# ---------------------------------------------------------------------------
# RMSNorm (no learnable bias, following nanochat)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight

# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

def precompute_rope(dim: int, max_len: int, base: float = 10000.0):
    """Precompute cos/sin for rotary embeddings."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x, cos, sin):
    """Apply rotary embeddings to query/key tensors."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    seq_len = x.shape[-2]
    cos = cos[:seq_len].to(x.device)
    sin = sin[:seq_len].to(x.device)
    # broadcast over batch and head dims
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

# ---------------------------------------------------------------------------
# GPT Model (simplified nanochat architecture)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # QK normalization (following nanochat)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, rope_cos, rope_sin):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, T, n_head, head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q.transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)
        # Scaled dot-product attention with causal mask (PyTorch 2.0+)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.proj(y)

class MLP(nn.Module):
    """Feed-forward with ReLU² activation (following nanochat)."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = 4 * config.n_embd
        self.up = nn.Linear(config.n_embd, hidden, bias=False)
        self.gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        return self.down(F.relu(self.gate(x)).pow(2) * self.up(x))

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.ln1(x), rope_cos, rope_sin)
        x = x + self.mlp(self.ln2(x))
        return x

class NanoGPT(nn.Module):
    """Minimal GPT for from-scratch training. Architecture matches nanochat."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # no weight tying (following nanochat)
        # precompute RoPE
        rope_cos, rope_sin = precompute_rope(
            config.n_embd // config.n_head, config.sequence_len * 2
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)
        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"NanoGPT: {n_params/1e6:.1f}M parameters, depth={config.n_layer}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.wte(idx)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

# ---------------------------------------------------------------------------
# GSM8K Dataset
# ---------------------------------------------------------------------------

GSM8K_PROMPT_TEMPLATE = """Solve this math problem step by step. Show your work, then give the final answer after ####.

Question: {question}

Solution:"""

def load_gsm8k():
    """Load GSM8K dataset. Returns train and test splits."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main")
        train = [{"question": x["question"], "answer": x["answer"]} for x in ds["train"]]
        test = [{"question": x["question"], "answer": x["answer"]} for x in ds["test"]]
    except Exception as e:
        print(f"[WARNING] Could not load GSM8K from HuggingFace: {e}")
        print("[WARNING] Falling back to synthetic math problems for demo purposes")
        train, test = generate_synthetic_math(n_train=1000, n_test=200)
    print(f"GSM8K loaded: {len(train)} train, {len(test)} test")
    return train, test

def generate_synthetic_math(n_train=1000, n_test=200):
    """Generate simple arithmetic problems as a fallback dataset."""
    random.seed(42)
    def make_problem():
        a, b = random.randint(1, 100), random.randint(1, 100)
        op = random.choice(["+", "-", "*"])
        if op == "+":
            ans = a + b
            q = f"What is {a} + {b}?"
            sol = f"{a} + {b} = {ans}\n#### {ans}"
        elif op == "-":
            a, b = max(a, b), min(a, b)
            ans = a - b
            q = f"What is {a} - {b}?"
            sol = f"{a} - {b} = {ans}\n#### {ans}"
        else:
            a, b = random.randint(1, 20), random.randint(1, 20)
            ans = a * b
            q = f"What is {a} × {b}?"
            sol = f"{a} × {b} = {ans}\n#### {ans}"
        return {"question": q, "answer": sol}
    all_problems = [make_problem() for _ in range(n_train + n_test)]
    return all_problems[:n_train], all_problems[n_train:]

def extract_gold_answer(answer_text: str) -> str:
    """Extract the numerical answer from GSM8K's #### format."""
    match = re.search(r'####\s*(-?[\d,\.]+)', answer_text)
    if match:
        return match.group(1).replace(',', '').strip()
    return ""

def format_prompt(question: str) -> str:
    """Format a question into the training prompt."""
    return GSM8K_PROMPT_TEMPLATE.format(question=question)

# ---------------------------------------------------------------------------
# Reward Computation
# ---------------------------------------------------------------------------

def extract_model_answer(text: str) -> Optional[str]:
    """Extract the final numerical answer from model output.
    Tries #### format first, then falls back to last number."""
    # Try #### format
    match = re.search(r'####\s*(-?[\d,\.]+)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    # Try \\boxed{} format
    match = re.search(r'\\boxed\{(-?[\d,\.]+)\}', text)
    if match:
        return match.group(1).replace(',', '').strip()
    # Fallback: last number in text
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if numbers:
        return numbers[-1].replace(',', '').strip()
    return None

def compute_reward(model_output: str, gold_answer: str) -> float:
    """Binary reward: 1.0 if correct, 0.0 if wrong.
    Also gives 0.5 partial credit for correct format but wrong answer."""
    pred = extract_model_answer(model_output)
    if pred is None:
        return 0.0
    try:
        # Format reward: model at least produced a parseable number
        format_bonus = 0.0
        if "####" in model_output:
            format_bonus = 0.1
        # Correctness check
        if abs(float(pred) - float(gold_answer)) < 1e-5:
            return 1.0
        return format_bonus
    except (ValueError, TypeError):
        return 0.0

# ---------------------------------------------------------------------------
# Text Generation with Log-Probabilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_with_logprobs(model, tokenizer, prompt_tokens, config, device):
    """Generate a completion autoregressively, collecting per-token logprobs.
    Returns (completion_tokens, sum_logprobs, per_token_logprobs)."""
    model.eval()
    tokens = prompt_tokens.clone().to(device)
    prompt_len = tokens.shape[1]
    all_logprobs = []

    for _ in range(config.max_completion_len):
        # Truncate to sequence length (handle both NanoGPT and HuggingFace configs)
        max_seq = getattr(model.config, 'sequence_len', None) or getattr(model.config, 'max_position_embeddings', 512)
        if tokens.shape[1] > max_seq:
            ctx = tokens[:, -max_seq:]
        else:
            ctx = tokens

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(ctx)
            logits = out[0] if isinstance(out, tuple) else out.logits

        logits = logits[:, -1, :]  # last token logits

        # Apply temperature and top-k
        logits = logits / config.temperature
        if config.top_k > 0:
            v, _ = torch.topk(logits, min(config.top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        logprob = torch.log(probs.gather(1, next_token) + 1e-10)
        all_logprobs.append(logprob)
        tokens = torch.cat([tokens, next_token], dim=1)

        # Stop on EOS (tokenizer-dependent)
        if hasattr(tokenizer, 'eos_token_id') and next_token.item() == tokenizer.eos_token_id:
            break

    completion_tokens = tokens[:, prompt_len:]
    per_token_logprobs = torch.cat(all_logprobs, dim=1) if all_logprobs else torch.zeros(1, 0, device=device)
    sum_logprobs = per_token_logprobs.sum()
    model.train()
    return completion_tokens, sum_logprobs, per_token_logprobs

def compute_logprobs_for_sequence(model, full_tokens, prompt_len, device):
    """Compute log-probabilities for completion tokens given prompt+completion.
    This is the differentiable version used in the loss computation."""
    full_tokens = full_tokens.to(device)
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        outputs = model(full_tokens)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits

    # Shift: logits[t] predicts token[t+1]
    shift_logits = logits[:, prompt_len-1:-1, :]  # predictions for completion tokens
    shift_targets = full_tokens[:, prompt_len:]    # actual completion tokens
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logprobs = log_probs.gather(2, shift_targets.unsqueeze(-1)).squeeze(-1)
    return token_logprobs

# ---------------------------------------------------------------------------
# GRPO Core Algorithm
# ---------------------------------------------------------------------------

def grpo_step(model, ref_model, batch, tokenizer, grpo_config, device, step_num=0):
    """One GRPO training step. The heart of nanoReasoner.

    For each prompt in the batch:
      1. Generate G completions
      2. Score each with binary reward
      3. Compute group-relative advantages
      4. Compute policy gradient loss (mode-dependent)

    Returns: loss, metrics dict
    """
    all_losses = []
    total_reward = 0.0
    total_correct = 0
    total_completions = 0
    total_tokens = 0
    skipped_groups = 0
    avg_completion_len = 0

    for prompt_text, gold_answer in batch:
        # Tokenize prompt
        prompt_tokens = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        prompt_len = prompt_tokens.shape[1]

        # --- 1. ROLLOUT: Generate G completions ---
        completions = []     # list of (completion_tokens, old_logprobs_sum, per_token_logprobs)
        rewards = []

        for g in range(grpo_config.G):
            comp_tokens, sum_lp, per_lp = generate_with_logprobs(
                model, tokenizer, prompt_tokens, grpo_config, device
            )
            # Decode completion
            comp_text = tokenizer.decode(comp_tokens[0], skip_special_tokens=True)
            # --- 2. REWARD ---
            r = compute_reward(comp_text, gold_answer)
            completions.append((comp_tokens, sum_lp, per_lp, prompt_len))
            rewards.append(r)
            total_reward += r
            total_correct += (1 if r >= 1.0 else 0)
            total_completions += 1
            avg_completion_len += comp_tokens.shape[1]

        rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)

        # --- 3. ADVANTAGE: Group-relative normalization ---
        if rewards_t.std() < 1e-8:
            # All completions got the same reward — no learning signal
            skipped_groups += 1
            continue

        if grpo_config.mode == "nano":
            # Karpathy-style: simple mean subtraction
            advantages = rewards_t - rewards_t.mean()
        else:
            # Standard GRPO: z-score normalization
            advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

        # --- 4. LOSS ---
        for i, (comp_tokens, old_sum_lp, old_per_lp, p_len) in enumerate(completions):
            adv = advantages[i]
            if comp_tokens.shape[1] == 0:
                continue

            # Build full sequence: prompt + completion
            full_tokens = torch.cat([prompt_tokens, comp_tokens], dim=1)

            # Compute new log-probabilities (differentiable)
            new_token_logprobs = compute_logprobs_for_sequence(model, full_tokens, p_len, device)
            n_tokens = new_token_logprobs.shape[1]
            total_tokens += n_tokens

            if grpo_config.mode == "nano":
                # Simple REINFORCE with group-relative baseline
                # No importance ratio, no clipping — pure on-policy
                loss = -(new_token_logprobs * adv).sum() / max(n_tokens, 1)
            else:
                # PPO-style clipped surrogate objective
                # Compute importance sampling ratio
                old_token_lp = old_per_lp[:, :n_tokens].detach()
                ratio = torch.exp(new_token_logprobs - old_token_lp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - grpo_config.epsilon,
                                           1 + grpo_config.epsilon) * adv
                # DAPO-style: normalize by total tokens in batch
                loss = -torch.min(surr1, surr2).sum() / max(n_tokens, 1)

            # --- 5. KL PENALTY (full mode only) ---
            if grpo_config.mode == "full" and ref_model is not None:
                with torch.no_grad():
                    ref_token_logprobs = compute_logprobs_for_sequence(
                        ref_model, full_tokens, p_len, device
                    )
                # Schulman's unbiased KL estimator
                log_ratio = ref_token_logprobs[:, :n_tokens] - new_token_logprobs
                kl = torch.exp(log_ratio) - log_ratio - 1.0
                loss = loss + grpo_config.beta * kl.mean()

            all_losses.append(loss)

    if len(all_losses) == 0:
        return None, {"loss": 0, "reward_mean": 0, "accuracy": 0,
                       "avg_completion_len": 0, "total_completions": 0,
                       "skipped_groups": skipped_groups, "total_tokens": 0}

    total_loss = torch.stack(all_losses).mean()

    metrics = {
        "loss": total_loss.item(),
        "reward_mean": total_reward / max(total_completions, 1),
        "accuracy": total_correct / max(total_completions, 1),
        "avg_completion_len": avg_completion_len / max(total_completions, 1),
        "total_completions": total_completions,
        "skipped_groups": skipped_groups,
        "total_tokens": total_tokens,
    }
    return total_loss, metrics

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, tokenizer, test_data, grpo_config, device, n_samples=100):
    """Evaluate model on GSM8K test set. Returns accuracy and sample outputs."""
    model.eval()
    correct = 0
    total = 0
    samples = []

    subset = random.sample(test_data, min(n_samples, len(test_data)))
    for item in subset:
        prompt = format_prompt(item["question"])
        gold = extract_gold_answer(item["answer"])
        prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        comp_tokens, _, _ = generate_with_logprobs(model, tokenizer, prompt_tokens, grpo_config, device)
        comp_text = tokenizer.decode(comp_tokens[0], skip_special_tokens=True)
        pred = extract_model_answer(comp_text)

        is_correct = False
        try:
            if pred and gold and abs(float(pred) - float(gold)) < 1e-5:
                is_correct = True
        except (ValueError, TypeError):
            pass

        correct += int(is_correct)
        total += 1

        if len(samples) < 5:
            samples.append({
                "question": item["question"],
                "gold": gold,
                "prediction": pred,
                "correct": is_correct,
                "output": comp_text[:500],
            })

    model.train()
    accuracy = correct / max(total, 1)
    return accuracy, samples

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_training_curves(history, run_dir):
    """Plot reward, accuracy, and completion length curves."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not available, skipping plots")
        return

    steps = [h["step"] for h in history]
    rewards = [h["reward_mean"] for h in history]
    accuracies = [h.get("eval_accuracy", None) for h in history]
    comp_lens = [h["avg_completion_len"] for h in history]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("nanoReasoner: GRPO Training Progress", fontsize=14, fontweight='bold')

    # Reward curve
    axes[0].plot(steps, rewards, 'g-', alpha=0.5, linewidth=0.8)
    # Running average
    window = min(20, len(rewards) // 4 + 1)
    if len(rewards) >= window:
        running_avg = [sum(rewards[max(0,i-window):i+1]) / len(rewards[max(0,i-window):i+1])
                      for i in range(len(rewards))]
        axes[0].plot(steps, running_avg, 'g-', linewidth=2, label='running avg')
    axes[0].set_ylabel("Mean Reward")
    axes[0].set_title("Reward (higher = more correct)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Eval accuracy (sparse, only at eval steps)
    eval_steps = [s for s, a in zip(steps, accuracies) if a is not None]
    eval_accs = [a for a in accuracies if a is not None]
    if eval_accs:
        axes[1].plot(eval_steps, eval_accs, 'bo-', markersize=6, linewidth=2)
    axes[1].set_ylabel("GSM8K Accuracy")
    axes[1].set_title("Evaluation Accuracy")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)

    # Completion length
    axes[2].plot(steps, comp_lens, 'r-', alpha=0.5, linewidth=0.8)
    if len(comp_lens) >= window:
        running_len = [sum(comp_lens[max(0,i-window):i+1]) / len(comp_lens[max(0,i-window):i+1])
                      for i in range(len(comp_lens))]
        axes[2].plot(steps, running_len, 'r-', linewidth=2, label='running avg')
    axes[2].set_ylabel("Tokens")
    axes[2].set_title("Average Completion Length (longer = model learning to 'think')")
    axes[2].set_xlabel("Training Step")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(run_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved training curves to {plot_path}")

# ---------------------------------------------------------------------------
# Git Tracking (autoresearch-style)
# ---------------------------------------------------------------------------

def git_init(run_dir):
    """Initialize git repo for experiment tracking."""
    try:
        if not os.path.exists(os.path.join(run_dir, ".git")):
            subprocess.run(["git", "init"], cwd=run_dir, capture_output=True)
            subprocess.run(["git", "add", "."], cwd=run_dir, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init: nanoReasoner experiment"],
                         cwd=run_dir, capture_output=True)
        return True
    except Exception:
        return False

def git_checkpoint(run_dir, step, metrics):
    """Auto-commit checkpoint with experiment metadata in commit message."""
    try:
        subprocess.run(["git", "add", "-A"], cwd=run_dir, capture_output=True)
        msg = (f"step {step}: reward={metrics.get('reward_mean', 0):.4f} "
               f"acc={metrics.get('accuracy', 0):.3f} "
               f"len={metrics.get('avg_completion_len', 0):.0f}")
        subprocess.run(["git", "commit", "-m", msg, "--allow-empty"],
                      cwd=run_dir, capture_output=True)
        return True
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Model Loading (pretrained or scratch)
# ---------------------------------------------------------------------------

def load_model(config: TrainConfig, device):
    """Load a pretrained model or create a from-scratch NanoGPT."""
    if config.model_name == "scratch":
        print(f"\n{'='*60}")
        print(f"  Building NanoGPT from scratch (depth={config.depth})")
        print(f"{'='*60}")
        gpt_config = GPTConfig(
            n_layer=config.depth,
            n_head=config.depth,
            n_embd=config.depth * 64,
            vocab_size=32000,
            sequence_len=512,
        )
        model = NanoGPT(gpt_config).to(device)
        # Use a simple tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                                   trust_remote_code=True)
        return model, tokenizer
    else:
        print(f"\n{'='*60}")
        print(f"  Loading pretrained model: {config.model_name}")
        print(f"{'='*60}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": getattr(torch, config.dtype),
        }
        if config.use_8bit_optim:
            model_kwargs["load_in_8bit"] = False  # we handle quantization in optimizer
        model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
        model = model.to(device)

        if config.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("[OK] Gradient checkpointing enabled")

        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {n_params/1e6:.1f}M")
        print(f"  Trainable parameters: {n_trainable/1e6:.1f}M")
        return model, tokenizer

# ---------------------------------------------------------------------------
# Learning Rate Schedule
# ---------------------------------------------------------------------------

def get_lr(step, config: TrainConfig):
    """Cosine decay with linear warmup."""
    if step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, config.total_steps - config.warmup_steps)
    return config.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

# ---------------------------------------------------------------------------
# Print Samples (watch reasoning emerge!)
# ---------------------------------------------------------------------------

def print_samples(samples, step):
    """Display sample outputs — this is where you watch the model learn to think."""
    print(f"\n{'─'*70}")
    print(f"  SAMPLE OUTPUTS at step {step}")
    print(f"{'─'*70}")
    for i, s in enumerate(samples):
        status = "✓" if s["correct"] else "✗"
        print(f"\n  [{status}] Q: {s['question'][:100]}")
        print(f"      Gold: {s['gold']}")
        print(f"      Pred: {s['prediction']}")
        # Show first 200 chars of reasoning
        reasoning = s['output'][:200].replace('\n', '\n      ')
        print(f"      Output: {reasoning}")
        if i >= 2:
            break
    print(f"{'─'*70}\n")

# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def train(config: TrainConfig):
    """The main training loop. This is where the magic happens."""

    # Setup
    device = config.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[WARNING] No CUDA device found. Training on CPU will be very slow.")

    torch.manual_seed(config.seed)
    random.seed(config.seed)

    # Run directory
    if not config.run_name:
        config.run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join("runs", config.run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Load model
    model, tokenizer = load_model(config, device)

    # GRPO config
    grpo_config = GRPOConfig(
        mode=config.mode,
        G=config.G,
        max_completion_len=min(config.G * 64, 512),  # scale with group size
    )

    # Reference model (for full GRPO mode)
    ref_model = None
    if config.mode == "full":
        import copy
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        print("[OK] Reference model created for KL penalty")

    # Load data
    train_data, test_data = load_gsm8k()

    # Optimizer
    optimizer_cls = torch.optim.AdamW
    if config.use_8bit_optim:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
            print("[OK] Using 8-bit AdamW")
        except ImportError:
            print("[WARNING] bitsandbytes not available, using standard AdamW")

    optimizer = optimizer_cls(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )

    # Git tracking
    if config.git_tracking:
        git_init(run_dir)

    # Training history for plots
    history = []

    # Fixed eval prompts for sample tracking
    eval_prompts = test_data[:5]

    print(f"\n{'='*60}")
    print(f"  nanoReasoner Training")
    print(f"  Mode: {config.mode} | G: {config.G} | LR: {config.lr}")
    print(f"  Steps: {config.total_steps} | Batch: {config.batch_size}")
    print(f"  Device: {device} | Dtype: {config.dtype}")
    print(f"  Run: {run_dir}")
    print(f"{'='*60}\n")

    # --- Training loop ---
    model.train()
    step_timer = time.time()

    for step in range(1, config.total_steps + 1):
        # Learning rate schedule
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Sample a batch of problems
        batch_items = random.sample(train_data, min(config.batch_size, len(train_data)))
        batch = [(format_prompt(item["question"]), extract_gold_answer(item["answer"]))
                 for item in batch_items]

        # GRPO step
        loss, metrics = grpo_step(model, ref_model, batch, tokenizer, grpo_config, device, step)

        if loss is not None:
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Update reference model periodically (full mode)
        if config.mode == "full" and ref_model is not None and step % grpo_config.ref_update_every == 0:
            import copy
            ref_model.load_state_dict(model.state_dict())
            print(f"  [step {step}] Reference model updated")

        # Logging
        # Ensure all keys exist even when grpo_step skips everything
        default_metrics = {
            "loss": 0, "reward_mean": 0, "accuracy": 0,
            "avg_completion_len": 0, "total_completions": 0,
            "skipped_groups": 0, "total_tokens": 0,
        }
        default_metrics.update(metrics)
        step_metrics = {
            "step": step,
            "lr": lr,
            **default_metrics,
        }

        if step % config.log_every == 0:
            elapsed = time.time() - step_timer
            step_timer = time.time()
            print(f"  step {step:5d}/{config.total_steps} | "
                  f"loss {metrics.get('loss', 0):7.4f} | "
                  f"reward {metrics.get('reward_mean', 0):.3f} | "
                  f"acc {metrics.get('accuracy', 0):.3f} | "
                  f"len {metrics.get('avg_completion_len', 0):5.0f} | "
                  f"skip {metrics.get('skipped_groups', 0)} | "
                  f"lr {lr:.2e} | "
                  f"{elapsed:.1f}s")

        # Evaluation
        if step % config.eval_every == 0:
            eval_acc, eval_samples = evaluate(model, tokenizer, test_data, grpo_config, device)
            step_metrics["eval_accuracy"] = eval_acc
            print(f"\n  >>> EVAL at step {step}: GSM8K accuracy = {eval_acc:.3f} ({eval_acc*100:.1f}%)\n")
            print_samples(eval_samples, step)
        else:
            step_metrics["eval_accuracy"] = None

        history.append(step_metrics)

        # Save checkpoint
        if step % config.save_every == 0:
            ckpt_dir = os.path.join(run_dir, f"step_{step:06d}")
            os.makedirs(ckpt_dir, exist_ok=True)
            # Save model
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
            else:
                torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
            # Save metrics
            with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
                json.dump(step_metrics, f, indent=2)
            print(f"  [SAVED] Checkpoint at {ckpt_dir}")

            # Plot curves
            plot_training_curves(history, run_dir)

            # Git commit
            if config.git_tracking:
                git_checkpoint(run_dir, step, metrics)

        # Sample outputs for watching reasoning emerge
        if step % config.sample_every == 0 and step % config.eval_every != 0:
            _, samples = evaluate(model, tokenizer, eval_prompts[:3], grpo_config, device, n_samples=3)
            print_samples(samples, step)

    # --- Training complete ---
    print(f"\n{'='*60}")
    print(f"  Training complete! {config.total_steps} steps")
    print(f"  Final reward: {history[-1].get('reward_mean', 0):.4f}")
    print(f"  Run saved to: {run_dir}")
    print(f"{'='*60}\n")

    # Final evaluation
    print("Running final evaluation...")
    final_acc, final_samples = evaluate(model, tokenizer, test_data, grpo_config, device, n_samples=200)
    print(f"\n  FINAL GSM8K ACCURACY: {final_acc:.3f} ({final_acc*100:.1f}%)\n")
    print_samples(final_samples, "FINAL")

    # Save final model
    final_dir = os.path.join(run_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
    else:
        torch.save(model.state_dict(), os.path.join(final_dir, "model.pt"))

    # Save training history
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Final plot
    plot_training_curves(history, run_dir)

    print(f"\n  All artifacts saved to: {run_dir}/")
    print(f"  - config.json         (experiment config)")
    print(f"  - history.json        (full training metrics)")
    print(f"  - training_curves.png (reward/accuracy/length plots)")
    print(f"  - final/              (final model weights)")
    print(f"  - step_*/             (intermediate checkpoints)")
    return model, tokenizer, history

# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="nanoReasoner: GRPO training for math reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick start: fine-tune 0.5B model with nano GRPO
  python nanoreasoner.py --model Qwen/Qwen2.5-0.5B-Instruct --mode nano --G 4

  # Full GRPO with KL penalty on 1.5B model
  python nanoreasoner.py --model Qwen/Qwen2.5-1.5B --mode full --G 4

  # From-scratch GPT (educational)
  python nanoreasoner.py --model scratch --depth 8 --mode nano --G 16 --total-steps 200

  # Evaluate a saved checkpoint
  python nanoreasoner.py --eval --checkpoint runs/my_run/final

Algorithm modes:
  nano     - Simplified REINFORCE with group-relative baseline (nanochat-style)
  standard - Full GRPO with PPO clipping and z-score advantages
  full     - GRPO + KL penalty against frozen reference model (DeepSeek R1-style)
        """
    )

    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                       help="HuggingFace model name or 'scratch' for from-scratch GPT")
    parser.add_argument("--depth", type=int, default=8,
                       help="Depth for scratch model (n_layer = n_head = depth)")

    # Algorithm
    parser.add_argument("--mode", type=str, default="nano", choices=["nano", "standard", "full"],
                       help="GRPO mode: nano (simplest), standard (with clipping), full (with KL)")
    parser.add_argument("--G", type=int, default=8,
                       help="Group size: number of completions per prompt")

    # Training
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--total-steps", type=int, default=500, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Prompts per step")
    parser.add_argument("--warmup-steps", type=int, default=50, help="LR warmup steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Evaluation
    parser.add_argument("--eval", action="store_true", help="Evaluation only mode")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint to load")
    parser.add_argument("--eval-every", type=int, default=50, help="Steps between evaluations")

    # Tracking
    parser.add_argument("--run-name", type=str, default="", help="Name for this run")
    parser.add_argument("--no-git", action="store_true", help="Disable git tracking")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Compute dtype")
    parser.add_argument("--use-8bit", action="store_true", help="Use 8-bit AdamW optimizer")
    parser.add_argument("--no-grad-ckpt", action="store_true", help="Disable gradient checkpointing")

    args = parser.parse_args()

    config = TrainConfig(
        model_name=args.model,
        mode=args.mode,
        G=args.G,
        depth=args.depth,
        lr=args.lr,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        eval_only=args.eval,
        checkpoint=args.checkpoint,
        eval_every=args.eval_every,
        run_name=args.run_name,
        git_tracking=not args.no_git,
        device=args.device,
        dtype=args.dtype,
        use_8bit_optim=args.use_8bit,
        gradient_checkpointing=not args.no_grad_ckpt,
    )

    if config.eval_only and config.checkpoint:
        # Evaluation mode
        device = config.device if torch.cuda.is_available() else "cpu"
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading checkpoint: {config.checkpoint}")
        model = AutoModelForCausalLM.from_pretrained(config.checkpoint, trust_remote_code=True,
                                                      torch_dtype=torch.bfloat16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(config.checkpoint, trust_remote_code=True)
        grpo_config = GRPOConfig(mode=config.mode, G=1)
        _, test_data = load_gsm8k()
        acc, samples = evaluate(model, tokenizer, test_data, grpo_config, device, n_samples=200)
        print(f"\nGSM8K Accuracy: {acc:.3f} ({acc*100:.1f}%)")
        print_samples(samples, "EVAL")
    else:
        # Training mode
        train(config)

# ---------------------------------------------------------------------------
# The story so far:
#
# micrograd  → backprop from scratch
# makemore   → character-level language models
# nanoGPT    → GPT pretraining demystified
# minbpe     → tokenization from scratch
# llm.c      → training in raw C/CUDA
# llama2.c   → inference in pure C
# nanochat   → full ChatGPT pipeline for $100
# microGPT   → entire GPT in 200 lines
# autoresearch → agents that optimize training code
#
# nanoReasoner → teaching models to think via RL
#
# One day, the models trained by nanoReasoner will write better nanoReasoners.
# That day is not today. But the loop has begun.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
