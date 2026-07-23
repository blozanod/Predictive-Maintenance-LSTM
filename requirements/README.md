# Per-model requirements (one isolated stack per TSFM backbone)

The five frozen-TSFM backbones **cannot share one Python environment** — their
dependency pins are mutually incompatible, proven on a fresh Colab GPU runtime:

| Backbone | The conflict it forces |
|---|---|
| **Moirai-2** (`uni2ts`) | pins `torch<2.5` → uninstalls Colab's torch 2.10, drops in 2.4.1, which no longer matches the preinstalled `torchvision` → `operator torchvision::nms does not exist` (and that poisoned torch/torchvision then breaks Chronos + TTM's `transformers` import) |
| **MOMENT** (`momentfm`) | hard-pins `numpy==1.25.2` (no Python-3.12 wheel; source build fails) and `huggingface-hub==0.24.0` |
| **Chronos-2 / TTM** | want the *modern* torch 2.10 + transformers ≥4.57 stack |
| **TimesFM 2.5** | fine on the modern stack; only needed the correct HF id |

So each backbone gets its **own file here** and runs in its **own fresh runtime**
(one Colab notebook per model — see `notebooks/verify/`). Install exactly one per
runtime; never combine them.

- `requirements.txt` (repo root) stays the installable **core + Chronos-2** used by the
  CPU test suite and the Chronos campaign — the backbones are deliberately not in it.
- `chronos.txt` / `ttm.txt` / `timesfm.txt` install cleanly on Colab's default stack.
- `moirai.txt` pins `torch==2.4.1` **and** a matching `torchvision==0.19.1` so
  `uni2ts → lightning → torchmetrics → torchvision` imports cleanly.
- `moment.txt` is installed with **`--no-deps`** (its own pins are unbuildable on 3.12);
  the verify notebook does this and relies on the runtime's existing numpy/torch/transformers.

Pins here are best-effort and finalized by running each `notebooks/verify/<model>.ipynb`
on a GPU — a backbone that still fails is reported, not forced (RESEARCH_PLAN §11).
