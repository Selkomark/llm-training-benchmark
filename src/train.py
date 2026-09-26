"""QLoRA fine-tune benchmark for Qwen3-4B-Instruct on a small synthetic dataset.

Measures load time, per-step time, token throughput, peak GPU memory and
eval loss before/after training. Results go to outputs/results.json.
"""

import argparse
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507"))
    ap.add_argument("--data", default="data/synthetic.jsonl")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--eval-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-quant", action="store_true",
                    help="Skip 4-bit loading (CPU smoke tests with tiny models only)")
    return ap.parse_args()


def encode(tok, ex, max_len):
    """Tokenize one chat example; only the assistant reply contributes to the loss."""
    prompt = [{"role": "user", "content": ex["instruction"]}]
    prompt_ids = tok.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
    if not isinstance(prompt_ids, list):  # transformers 5 returns a BatchEncoding
        prompt_ids = list(prompt_ids["input_ids"])
    answer_ids = tok(ex["response"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + answer_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
    return torch.tensor([input_ids]), torch.tensor([labels])


@torch.no_grad()
def eval_loss(model, batches, device):
    model.eval()
    losses = [model(input_ids=x.to(device), labels=y.to(device)).loss.item() for x, y in batches]
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def sample(model, tok, instruction, device):
    model.eval()
    ids = tok.apply_chat_template([{"role": "user", "content": instruction}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = {k: v.to(device) for k, v in ids.items()}
    out = model.generate(**ids, max_new_tokens=48, do_sample=False)
    model.train()
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cuda = torch.cuda.is_available()
    if not cuda and not args.no_quant:
        raise SystemExit("No CUDA GPU visible in the container. Is the NVIDIA Container "
                         "Toolkit installed? (see scripts/setup-gpu-docker.sh)")
    device = "cuda" if cuda else "cpu"
    gpu = torch.cuda.get_device_name(0) if cuda else "cpu"
    print(f"device={device} gpu={gpu} torch={torch.__version__}")

    examples = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    random.shuffle(examples)
    eval_ex, train_ex = examples[:args.eval_size], examples[args.eval_size:]

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.no_quant:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
    else:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb,
                                                     dtype=torch.bfloat16, device_map={"": 0})
        # Not prepare_model_for_kbit_training: it upcasts the 389M-param embedding
        # to fp32 (+1.45 GiB), which does not fit on a 4 GB GPU.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    load_s = time.perf_counter() - t0
    trainable, total = model.get_nb_trainable_parameters()
    mem_after_load = torch.cuda.memory_allocated() / 2**30 if cuda else 0.0
    print(f"loaded in {load_s:.1f}s, trainable {trainable/1e6:.1f}M / {total/1e9:.2f}B params, "
          f"GPU mem {mem_after_load:.2f} GiB")

    train_batches = [encode(tok, ex, args.max_len) for ex in train_ex]
    eval_batches = [encode(tok, ex, args.max_len) for ex in eval_ex]
    probe = eval_ex[0]["instruction"]

    base_eval = eval_loss(model, eval_batches, device)
    base_sample = sample(model, tok, probe, device)
    print(f"eval loss before: {base_eval:.4f}")

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    total_steps = math.ceil(len(train_batches) / args.grad_accum) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / total_steps))
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    step, tokens, step_times, history = 0, 0, [], []
    model.train()
    t_train = time.perf_counter()
    for epoch in range(args.epochs):
        random.shuffle(train_batches)
        for i, (x, y) in enumerate(train_batches):
            if i % args.grad_accum == 0:
                t_step = time.perf_counter()
                accum_loss = 0.0
                group = min(args.grad_accum, len(train_batches) - i)  # last group may be short
            loss = model(input_ids=x.to(device), labels=y.to(device)).loss / group
            loss.backward()
            accum_loss += loss.item()
            tokens += x.numel()
            last = i == len(train_batches) - 1
            if (i + 1) % args.grad_accum == 0 or last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                if cuda:
                    torch.cuda.synchronize()
                step += 1
                dt = time.perf_counter() - t_step
                step_times.append(dt)
                history.append({"step": step, "epoch": epoch + 1, "loss": round(accum_loss, 4),
                                "step_s": round(dt, 3)})
                print(f"epoch {epoch + 1} step {step}/{total_steps} loss {accum_loss:.4f} "
                      f"({dt:.2f}s/step)")
    train_s = time.perf_counter() - t_train

    final_eval = eval_loss(model, eval_batches, device)
    final_sample = sample(model, tok, probe, device)
    peak = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "adapter")
    results = {
        "model": args.model,
        "gpu": gpu,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "quantization": "none" if args.no_quant else "4-bit nf4 (QLoRA)",
        "train_examples": len(train_batches),
        "eval_examples": len(eval_batches),
        "epochs": args.epochs,
        "optimizer_steps": step,
        "max_len": args.max_len,
        "trainable_params_m": round(trainable / 1e6, 2),
        "load_seconds": round(load_s, 1),
        "train_seconds": round(train_s, 1),
        "mean_step_seconds": round(sum(step_times) / len(step_times), 3),
        "tokens_per_second": round(tokens / train_s, 1),
        "gpu_mem_after_load_gib": round(mem_after_load, 2),
        "peak_gpu_mem_gib": round(peak, 2),
        "eval_loss_before": round(base_eval, 4),
        "eval_loss_after": round(final_eval, 4),
        "probe_instruction": probe,
        "probe_reference": eval_ex[0]["response"],
        "probe_output_before": base_sample,
        "probe_output_after": final_sample,
        "history": history,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print("\n=== summary ===")
    for k in ["gpu", "load_seconds", "train_seconds", "mean_step_seconds", "tokens_per_second",
              "peak_gpu_mem_gib", "eval_loss_before", "eval_loss_after"]:
        print(f"{k:>20}: {results[k]}")
    print(f"{'probe':>20}: {probe}")
    print(f"{'reference':>20}: {results['probe_reference']}")
    print(f"{'before':>20}: {base_sample}")
    print(f"{'after':>20}: {final_sample}")
    print(f"\nfull results: {out / 'results.json'}")


if __name__ == "__main__":
    main()
