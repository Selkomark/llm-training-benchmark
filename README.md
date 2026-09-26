# llm-training-benchmark

Benchmark how well your machine fine-tunes an LLM. It runs a QLoRA fine-tune of
**Qwen3-4B-Instruct** on a small synthetic dataset, entirely inside Docker, and reports
timing, throughput, GPU memory and loss before/after training.

Sized to run on a **4 GB consumer GPU**. Weights come from Hugging Face
(`Qwen/Qwen3-4B-Instruct-2507`, Apache-2.0, ~8 GB, cached in `.cache/hf/`).

## Requirements

- NVIDIA GPU with ≥ 4 GB VRAM and a recent driver
- Docker with the NVIDIA Container Toolkit (`sudo sh scripts/setup-gpu-docker.sh` on
  Debian/Ubuntu; restarts the Docker daemon)
- Optional: a Hugging Face read token for faster downloads

## Quick start

```sh
git clone https://github.com/Selkomark/llm-training-benchmark.git
cd llm-training-benchmark
cp .env.example .env        # optional: add HF_TOKEN=hf_...
./run.sh                    # 3 epochs, defaults below
./run.sh --epochs 1         # any train.py flag can be appended
MODEL_ID=Qwen/Qwen3-1.7B ./run.sh   # try another model
```

## Output

`outputs/results.json` records model load time, total train time, seconds per optimizer
step, tokens/second, peak GPU memory, eval loss before/after, a probe prompt answered
before and after training, and per-step loss history. The LoRA adapter is saved to
`outputs/adapter/`.

## Defaults

| Setting | Value |
|---|---|
| Quantization | 4-bit NF4, double-quant, bf16 compute |
| LoRA | r=16, alpha=32, all attention + MLP projections (33M trainable params) |
| Data | 100 synthetic examples (90 train / 10 eval), seed 42 |
| Sequence length | 256 |
| Batch | 1, gradient accumulation 4, gradient checkpointing |
| Optimizer | AdamW, lr 2e-4, linear decay |

The synthetic data (`src/generate_data.py`) covers arithmetic, unit conversion, sorting,
string manipulation and JSON extraction. Every answer is computed, so it is always correct.
To train on your own data, pass any JSONL file with `instruction` and `response` fields
via `--data`.

## Example result

ASUS ROG Flow X13 laptop, RTX 3050 Ti Laptop GPU (4 GB), torch 2.14, CUDA 13:

| Metric | Value |
|---|---|
| Train time (3 epochs, 69 steps) | 256 s |
| Mean step time | 3.7 s (2.3 s cool, ~4.5 s once thermally throttled) |
| Throughput | 42 tokens/s |
| Peak GPU memory | 3.15 GiB |
| Eval loss before → after | 4.41 → 0.107 |

## Notes for 4 GB GPUs

- `prepare_model_for_kbit_training` is deliberately not used: it upcasts the 389M-parameter
  embedding to fp32 (+1.45 GiB), which does not fit. Gradient checkpointing and input
  gradients are enabled directly instead.
- `gcc` is installed in the image because torch's native Triton kernels compile a small
  helper on first use.

## License

MIT, see [LICENSE](LICENSE).
