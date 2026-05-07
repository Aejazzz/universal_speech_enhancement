# Universal Speech Enhancement Policy Learning

Adaptive multi-expert speech enhancement platform with a transformer routing policy and production-oriented FastAPI + React stack.

## Features

- Real expert routing actions: `DeepFilterNet3`, `ResembleEnhance`, `MossFormer2`, `BYPASS`
- WavLM embedding extraction for semantic/acoustic conditioning
- CRNN distortion analyzer on mel spectrograms
- Transformer policy agent with confidence outputs and refinement control
- Quality evaluation with `DNSMOS` (placeholder hook + API point), `UTMOS` (model hook), `PESQ`, `STOI`, `SI-SDR`
- Research-grade plots (waveforms, spectrograms, confidence, routing distribution)
- FastAPI backend with async endpoints and model caching
- React + Tailwind + Framer Motion frontend dashboard
- Training/evaluation/inference scripts, config-driven execution, TensorBoard logging

## Project Layout

See directories:
`backend`, `policy_agent`, `distortion_analyzer`, `enhancement_experts`, `evaluation`, `visualizations`, `frontend`, `scripts`, `configs`, `tests`, `notebooks`.

## Dataset

The system is preconfigured to load datasets from:

`C:\Users\jsm10\OneDrive - Amrita vishwa vidyapeetham\agentic-speech-enhancement\datasets`

No dataset download is performed.

## Windows + CUDA Setup

1. Install Python 3.10+, CUDA toolkit and matching GPU drivers.
2. Create env:
   - `conda env create -f environment.yml`
   - `conda activate universal-speech-enhancement`
   - Recommended: Python `3.10+` for full expert compatibility
   - Keep `torch==2.6.0` and `torchaudio==2.6.0` for DeepFilterNet compatibility
3. Install PyTorch CUDA build if needed:
   - Follow [PyTorch install selector](https://pytorch.org/get-started/locally/)
4. Install frontend:
   - `cd frontend && npm install`

Optional experts:

- `pip install -r requirements-optional.txt`
- `Resemble Enhance` may require Python 3.10+ and source build dependencies.
- For MossFormer2, place checkpoint files under `checkpoints/` or `models/` with names matching `*moss*former*.pt|.pth|.ckpt`, or set `system.mossformer_checkpoint` in `configs/base.yaml`.

## Run Backend

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Run Frontend

```bash
cd frontend
npm run dev
```

## Inference CLI

```bash
python scripts/run_inference.py --input path/to/noisy.wav
```

Outputs are saved into `outputs/` including audio, metrics JSON, CSV, PNG plots and routing logs.

## Training

```bash
python scripts/prepare_dataset.py --dataset-root "C:/Users/jsm10/OneDrive - Amrita vishwa vidyapeetham/agentic-speech-enhancement/datasets"
```

```bash
python scripts/train_policy.py --config configs/base.yaml
```

- Mixed precision enabled
- Checkpointing in `checkpoints/`
- TensorBoard logs in `logs/tensorboard/`
- Early stopping supported
- Reports saved to `outputs/reports/` (`training_history.csv`, `model_performance.csv`, confusion matrices)

### Paired noisy/clean + Parquet

- **Audio pairs (router supervision):** `python scripts/build_paired_manifest.py` discovers `*.noisy/noisy/*.flac` next to `*.clean/clean/*.flac` under `Interspeech 2025 URGENT` and writes `outputs/manifests/paired_train.csv` / `paired_val.csv`. Optional HF metadata columns `hf_ceiling_*` are merged from `HF urgent2025-sqa/data/nonblind_test/*.parquet` (max scores over submissions — *not* raw noisy waveforms).
- **Composite-score oracle labels (with strength sweep):** `python scripts/compute_oracle_labels.py --manifest outputs/manifests/paired_train.csv --output outputs/manifests/paired_train_oracle.csv --strengths 0.6,1.0`. Composite = `0.6 * PESQ_wb + 0.3 * (STOI*5) + 0.1 * tanh(SI_SDR/15)`. Resilient: skips bad files, flushes every 25 rows, **automatic resume** if the output CSV already exists.
- **Tie-breaking ("do no harm"):** `python scripts/rederive_oracle_labels.py --in <oracle_csv> --out <oracle_csv> --margin 0.005` re-derives labels from the existing scores and prefers BYPASS unless an expert improves the composite by `--margin`. Avoids spurious wins from experts that fall back to identity (e.g. when a checkpoint is missing).
- **Full paired train (frozen WavLM + soft labels + cosine LR):** `python scripts/train_policy.py --config configs/full_train_paired.yaml`. Saves `checkpoints/policy_best.pt` and a TensorBoard log under `logs/tensorboard/policy`.
- **End-to-end smoke test:** `python scripts/verify_e2e.py 12` runs the production `EnhancementPipeline` on 12 random val clips, prints routing + DNSMOS deltas, and writes `outputs/reports/e2e_verify.csv`.

## Notes on Third-Party Experts

- `DeepFilterNet3`: direct integration with `deepfilternet` Python package.
- `Resemble Enhance`: integrated via package hook and optional CLI subprocess fallback.
- `MossFormer2`: wrapped as an external model adapter with checkpoint path support from ClearerVoice-Studio artifacts.

This repository keeps wrappers production-friendly and pluggable so you can pin specific pretrained checkpoints for each expert.
