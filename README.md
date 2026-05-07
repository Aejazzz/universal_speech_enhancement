# Universal Speech Enhancement Policy Learning

> Adaptive multi-expert speech enhancement: **9 enhancement experts** (3 neural + 5 classical FFT + bypass) governed by a **frozen-WavLM transformer routing policy** and a **dynamic “speculate-and-measure” router**, served by a production FastAPI backend and a React + Tailwind dashboard.

The headline idea: rather than committing to a single denoising model, the system **enhances each clip with every viable expert at multiple strengths, scores all candidates with no-reference MOS predictors (DNSMOS P.835 + UTMOS22), and emits the winner**. A trained policy provides a fast first-pass advice; the dynamic router is the safety net that prevents any expert from making the input worse.

This repo is the result of that design fully implemented: training pipeline, inference pipeline, evaluation pipeline, REST API, dashboard, and a **100-clip** blind benchmark on the **Interspeech 2025 URGENT** validation set (mean ΔDNSMOS **+0.72**, mean ΔBAK **+1.54**, 100 % top-rank, 0 Pareto-dominated picks — see [`outputs/reports/demo_blind_eval/report.md`](outputs/reports/demo_blind_eval/report.md)).

For a one-page demo cheat sheet (architecture, headline numbers, run instructions, FAQ) see [`DEMO.md`](DEMO.md).

**Presentation:** Google-style slide deck at [`docs/Presentation_Universal_Speech_Enhancement.pptx`](docs/Presentation_Universal_Speech_Enhancement.pptx). Regenerate after updating plots: `pip install python-pptx` then `python scripts/build_presentation.py`.

---

## Table of contents

1. [System architecture](#1-system-architecture)
2. [End-to-end inference flow](#2-end-to-end-inference-flow)
3. [Components in detail](#3-components-in-detail)
4. [Repository layout](#4-repository-layout)
5. [Setup](#5-setup)
6. [Running the system](#6-running-the-system)
7. [Training](#7-training)
8. [Evaluation](#8-evaluation)
9. [Configuration](#9-configuration)
10. [API reference](#10-api-reference)
11. [Results](#11-results)
12. [Implementation notes & design choices](#12-implementation-notes--design-choices)

---

## 1. System architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React + Tailwind)                       │
│  Upload panel │ DNSMOS scorecard │ Candidate leaderboard │ A/B player    │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │ HTTPS multipart/form-data
┌──────────────────────────────▼───────────────────────────────────────────┐
│                       BACKEND (FastAPI / uvicorn)                        │
│   POST /enhance  ─►  EnhancementPipeline.run(input_path)                 │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │
                               ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │                    EnhancementPipeline                           │
   │                                                                  │
   │  1. load_audio   ─► resample to 16 kHz, mono                     │
   │  2. preprocess   ─► DC removal · 60 Hz HPF · -23 LUFS · LBS.1770 │
   │  3. distortion   ─► CRNN over log-mel ─► 6-D feature vector      │
   │  4. policy       ─► Frozen WavLM-base-plus (multi-layer pool)    │
   │                     + residual MLP trunk ─► (expert, strength,   │
   │                     refine, confidence, action probabilities)    │
   │  5. dynamic      ─► IF system.dynamic_routing:                   │
   │     router            for every active expert × strength sweep:  │
   │                          enh = expert.enhance(wave, sr, s)       │
   │                          dns = DNSMOS-P.835(enh)                 │
   │                          utm = UTMOS22(enh)        # if loaded   │
   │                          score = blend(dns, utm)                 │
   │                       choose argmax; "do no harm" margin keeps   │
   │                       BYPASS preferred on near-ties              │
   │                     ELSE: just run policy.expert at policy.s     │
   │  6. (optional) refine = re-run chosen expert with strength+0.1   │
   │  7. soft_limiter at -1 dBFS  (brick-wall)                        │
   │  8. compute_metrics  ─► DNSMOS / UTMOS / PESQ / STOI / SI-SDR    │
   │  9. plots           ─► waveform A/B · 3-panel spectrogram ·      │
   │                       policy probability bars                    │
   │ 10. write artifacts in outputs/<run_id>/                         │
   └──────────────────────────────────────────────────────────────────┘
```

The two-tier routing (trained policy + dynamic router) is the central design decision: the policy is fast and informative, the dynamic router guarantees we never ship an output worse than the input.

---

## 2. End-to-end inference flow

For a single noisy file, the pipeline runs the steps below. Numbers in `<>` are typical wall-clock times on an RTX 4060 Laptop GPU for a ~5 s utterance.

| # | Stage | What happens | Module |
| - | --- | --- | --- |
| 1 | **Load & resample** `<5 ms>` | Reads wav/flac/mp3, resamples to 16 kHz mono. | `backend/app/audio.py` |
| 2 | **Preprocess** `<10 ms>` | DC offset removal · 4th-order Butterworth HPF at 60 Hz · loudness normalization to -23 LUFS using `pyloudnorm` (peak-norm fallback). Returns a `PreprocessReport` with input/output RMS-dB and LUFS. | `backend/app/preprocess.py` |
| 3 | **Distortion analysis** `<8 ms>` | A small CRNN over log-mel emits a 6-D vector (`[snr, reverb, clip, noise, codec, intelligibility]`) used as conditioning input for the policy. A parallel interpretable summary is computed for the dashboard. | `distortion_analyzer/` |
| 4 | **Policy forward** `<60 ms>` | Frozen WavLM-base-plus produces hidden states; we mean-pool layers `{6, 9, 12}` and concatenate. The result is projected, fused with the distortion vector, passed through a residual MLP trunk, and three heads produce: action logits (4 classes), strength (sigmoid), refine (sigmoid). | `policy_agent/model.py` |
| 5 | **Dynamic router** `<6 s>` | When `system.dynamic_routing=true`, every **active** expert is run at strengths `{0.55, 0.80, 0.95, policy_strength}`. Each enhanced waveform is scored by DNSMOS P.835 (SIG/BAK/OVRL) and UTMOS22 (when available). The blended `rank_score` selects the winner with a “do-no-harm” margin in favor of `BYPASS`. | `backend/app/pipeline.py` |
| 6 | **Optional refine** | If `policy.refine=True` and the chosen expert is not `BYPASS`, it is re-applied to its own output at `min(1.0, strength + 0.1)`. | `backend/app/pipeline.py` |
| 7 | **Brick-wall limiter** `<2 ms>` | A peak-based limiter at -1 dBFS guarantees no clipping in the emitted file. | `backend/app/preprocess.py::soft_limiter` |
| 8 | **Metrics** `<300 ms>` | DNSMOS (SIG/BAK/OVRL/P808), UTMOS, plus PESQ-WB / STOI / SI-SDR vs the noisy input. If a clean reference is provided, an additional `vs_clean_reference` block is added. | `evaluation/metrics.py` |
| 9 | **Plots** `<400 ms>` | Waveform A/B, 3-panel spectrogram (orig / enhanced / removed-noise on a shared dB scale), policy-probability bar chart. | `visualizations/plots.py` |
| 10 | **Persist** | Per-run directory `outputs/<run_id>/` with `enhanced.wav`, `metrics.json`, `routing.json`, `summary.csv`, three PNGs. | `backend/app/pipeline.py::run` |

Total typical end-to-end latency: **~7 s** (dominated by the dynamic router running 6–8 active experts × 4 strengths). Setting `system.dynamic_routing=false` collapses the routing cost to a single expert call (< 200 ms).

---

## 3. Components in detail

### 3.1 Enhancement experts (`enhancement_experts/`)

All experts implement the same `EnhancementExpert` interface (`enhance(waveform, sr, strength) -> np.ndarray`). The factory builds the registry below.

| Name | Type | Engine | Notes |
| --- | --- | --- | --- |
| `DeepFilterNet3` | Neural | DeepFilterNet 3 (Schroeter et al., ICASSP ’23) | Public model, downloads on first call. CRN denoiser with deep filtering. |
| `ResembleEnhance` | Neural | `resemble_enhance` package (T2T diffusion) | Optional; gracefully disabled if package not installed. |
| `MossFormer2` | Neural | ClearerVoice-Studio MossFormer2 | Loads checkpoint from `checkpoints/` or `models/` matching `*moss*former*.pt|.pth|.ckpt`; auto-resolves via `_resolve_mossformer_checkpoint`. |
| `WPEDereverb` | Classical | Weighted prediction error dereverb | For reverberant inputs. |
| `NoiseReduce` | Classical | `noisereduce` (Sainburg) non-stationary | Spectral gating. |
| `SpectralGate` | Classical | Sainburg spectral gating, stationary | Robust baseline. |
| `WienerFilter` | Classical | Ephraim–Malah decision-directed (1984) | Strong general-purpose denoiser. |
| `SpectralSubtraction` | Classical | Boll (1979) with over-subtraction + spectral floor | Aggressive musical-noise risk; rank-score punishes it when artifacts dominate. |
| `BYPASS` | Identity | — | Anchors the “do no harm” guarantee. |

**Active-expert filter** (`EnhancementPipeline._expert_is_active`):

- Classical experts and `BYPASS` are always active.
- `DeepFilterNet3` is always active (auto-downloads).
- `WPEDereverb` / `NoiseReduce` are active iff their `available` attribute is true.
- `MossFormer2` is active iff a checkpoint loaded.
- `ResembleEnhance` is active iff the `resemble_enhance` package imports.

This is what keeps the dynamic router honest — dead experts never inflate latency or pollute the candidate set.

### 3.2 Distortion analyzer (`distortion_analyzer/`)

A small CRNN `Conv2d(1→16) → MaxPool → Conv2d(16→32) → AdaptiveAvgPool(8×8) → MLP` mapping a log-mel spectrogram to a 6-D vector. The output is fed into the policy network as conditioning. A separate `summarize_for_dashboard` produces interpretable acoustic features (`snr_db`, `reverb`, `clip`, `noise_level`, `codec`, `intelligibility`) shown in the UI.

### 3.3 Policy agent (`policy_agent/`)

```
waveform ─► WavLM-base-plus (frozen) ─► hidden_states (13 layers × T × 768)
                                       │
                                       ▼  mean-pool over time at layers {6, 9, 12}
                                       │
                                       ▼  concat ─► (B, 3·768)
                                       │
                                       ▼  audio_proj (LN → Linear → GELU → Dropout)
                                                     │
distortion_features (B, 6) ─► distortion_proj (LN → Linear → GELU)
                                                     │
                          concat ─► residual MLP trunk × num_layers
                                                     │
                                ┌────────────────────┼────────────────────┐
                                ▼                    ▼                    ▼
                         action_head           strength_head         refine_head
                         (Linear→4)          (Linear→1, sigmoid)   (Linear→1, sigmoid)
```

**Why frozen WavLM + a tiny head**: WavLM already encodes phonetic/acoustic structure across its layers (cf. SUPERB). Mean-pooling layers `{mid, ¾, last}` matches common practice for non-ASR downstream tasks. The trainable head is tiny (~2.67 M params), trains in minutes on a laptop GPU, and avoids overfitting on a few-thousand-clip dataset.

`ACTIONS = ["DeepFilterNet3", "ResembleEnhance", "MossFormer2", "BYPASS"]`. The trained policy picks among these four; the dynamic router is then free to override with a classical expert (or vice-versa) based on measured DNSMOS/UTMOS.

### 3.4 Preprocessing & limiter (`backend/app/preprocess.py`)

- **DC offset removal** (subtract mean).
- **High-pass at 60 Hz** (4th-order Butterworth SOS) — kills mains hum and rumble that no speech-band model needs.
- **Loudness normalization** to -23 LUFS using `pyloudnorm` (ITU-R BS.1770); falls back to peak-normalization at -3 dBFS if the meter is unavailable.
- **Soft limiter (brick-wall)** at -1 dBFS on the way out so no expert ships clipped audio.

A `PreprocessReport` records input/output RMS-dB and LUFS for the dashboard.

### 3.5 Dynamic router (`backend/app/pipeline.py::_dynamic_select`)

Pseudocode:

```python
candidates = [BYPASS_at_0]
for expert in active_experts:           # priority order
    for s in {0.55, 0.80, 0.95, policy_strength}:
        enh = expert.enhance(wave, sr, s)
        # scoring is done on the same signal that will be limited later
        enh_for_scoring = soft_limiter(enh, -1 dBFS)
        dns = DNSMOS_P835(enh_for_scoring)             # SIG, BAK, OVRL
        utm = UTMOS22(enh_for_scoring)                 # may be None
        if utmos_is_reliable():
            rank = 0.45*OVRL + 0.20*SIG + 0.15*BAK + 0.20*UTMOS
        else:
            rank = 0.60*OVRL + 0.25*SIG + 0.15*BAK
        candidates.append({expert, strength: s, ovrl, sig, bak, utm, rank})

# do-no-harm margin (= 0.02 OVRL): non-BYPASS must beat BYPASS by at least margin
best = argmax(c.rank − (0 if c.expert=="BYPASS" else 0.02))
```

Crucially:

- **One limiter pass total**: scoring uses a limited copy, but the *cached* audio for each candidate is unlimited so the final emit applies the limiter exactly once.
- **UTMOS reliability gating**: `evaluation.metrics.utmos_is_reliable()` reports whether the actual UTMOS22 hub model loaded. When it didn’t, the heuristic fallback is **excluded from the routing blend** — we don’t pretend a heuristic is a MOS predictor.
- **Telemetry**: every candidate (≤ 25–32 per clip) shows up in `routing.dynamic_candidates` in the API response, including expert, strength, all DNSMOS components, UTMOS, and rank score. The dashboard renders this as a leaderboard with a crown for the winner.

### 3.6 Metrics (`evaluation/metrics.py`)

- `dnsmos_full`: full P.835 + P.808 breakdown via the official `speechmos` ONNX models (`sig_mos`, `bak_mos`, `ovrl_mos`, `p808_mos`); deterministic heuristic fallback if `speechmos` is broken.
- `utmos_score`: UTMOS22-strong via `tarepan/SpeechMOS` torch.hub. Caches the model after first load. Falls back to a coarse RMS+spectral-flatness heuristic if loading fails.
- `utmos_is_reliable()`: true iff the real predictor is loaded.
- `compute_metrics`: enhanced/original/improvement scorecard (DNSMOS components, P808, UTMOS) plus `similarity_vs_noisy_input` (PESQ-WB, STOI, SI-SDR vs the input) and an optional `vs_clean_reference` block when a clean reference is supplied.

### 3.7 Backend (`backend/app/`)

- `main.py` — FastAPI app. `GET /health`, `POST /enhance` (multipart upload of `.wav/.flac/.mp3`), `/outputs` static mount. CORS open by default for local dev.
- `pipeline.py` — `EnhancementPipeline` orchestrating everything above.
- `schemas.py` — Pydantic response models: `RoutingDecision`, `MetricsResult`, `EnhancementResponse`. The routing payload exposes the **full** dynamic candidate list, the policy advice, the decision reason, the preprocessing report, and timings.
- `audio.py` — `load_audio`, `save_audio` (libsndfile-backed; resamples to 16 kHz mono).
- `config.py` — Pydantic-validated YAML loader.

### 3.8 Frontend (`frontend/`)

React 18 + Vite 6 + Tailwind 3 + Framer Motion 11.

- Upload panel (single file).
- DNSMOS scorecard with animated bars and delta pills.
- Candidate leaderboard with crown for the winner, every dynamic candidate visible.
- Preprocessing block (LUFS in/out, RMS-dB, DC offset).
- Audio A/B player (input vs enhanced).
- API base URL is set via `VITE_API_BASE_URL` (default `:8001`).

---

## 4. Repository layout

```
universal_speech_enhancement/
├── backend/                      FastAPI service
│   ├── app/
│   │   ├── main.py               /enhance, /health, /outputs static mount
│   │   ├── pipeline.py           EnhancementPipeline (the heart)
│   │   ├── preprocess.py         BS.1770 preprocess + soft_limiter
│   │   ├── audio.py              load_audio / save_audio
│   │   ├── schemas.py            Pydantic response models
│   │   └── config.py             YAML config loader
│   └── API.md                    REST API reference (verbose)
├── policy_agent/
│   ├── model.py                  TransformerPolicyAgent (frozen WavLM + heads)
│   └── train.py                  Routing-policy trainer (weighted CE + soft labels)
├── distortion_analyzer/
│   ├── model.py                  Tiny CRNN producing 6-D routing features
│   └── routing_features.py       Interpretable distortion summary for UI
├── enhancement_experts/
│   ├── base.py                   EnhancementExpert ABC
│   ├── factory.py                build_experts(...)
│   ├── deepfilternet.py · mossformer.py · resemble.py
│   ├── wiener.py · spectral_gate.py · spectral_subtraction.py
│   ├── noisereduce_expert.py · wpe_dereverb.py · bypass.py
├── evaluation/
│   └── metrics.py                DNSMOS, UTMOS, PESQ, STOI, SI-SDR, deltas
├── visualizations/
│   └── plots.py                  Waveform A/B · spectrogram · policy probs
├── datasets/
│   ├── loader.py                 Routing manifest loader (with class weights)
│   ├── paired_discovery.py       Walks Interspeech 2025 URGENT noisy/clean dirs
│   └── parquet_enrichment.py     Adds HF urgent2025-sqa ceiling scores
├── frontend/                     React + Vite + Tailwind + Framer Motion
│   ├── src/{App.jsx, api.js, main.jsx, index.css}
│   ├── tailwind.config.js · postcss.config.js · vite.config.js
│   └── package.json
├── scripts/                      Operational entry points
│   ├── prepare_dataset.py        Build training/val routing manifests
│   ├── build_paired_manifest.py  Discover paired clips into CSVs
│   ├── compute_oracle_labels.py  Composite-score sweep → oracle labels
│   ├── rederive_oracle_labels.py "Do-no-harm" margin re-derivation
│   ├── train_policy.py           Trainer wrapper
│   ├── evaluate_dataset.py       Dataset-level evaluation
│   ├── eval_blind_batch.py       Blind-test folder evaluator (resume-capable)
│   ├── run_inference.py          Single-file CLI
│   ├── verify_e2e.py             Manifest-driven smoke test
│   ├── verify_dynamic_routing.py Routing-only check
│   ├── demo_smoke.py             Dataset-free smoke test (added for the demo)
│   ├── start_demo.ps1            One-shot launcher (Windows)
│   ├── plot_training_results.py · plot_eval_aggregate.py
│   └── setup_windows.ps1
├── configs/
│   ├── base.yaml                 Inference + dynamic routing defaults
│   ├── train_stage1.yaml         Stage-1 training config
│   └── full_train_paired.yaml    Frozen WavLM + soft labels + cosine LR
├── tests/
│   └── test_health.py            Backend health check (pytest)
├── notebooks/                    Jupyter walkthroughs
├── outputs/                      Run artifacts (gitignored)
│   ├── <run_id>/                 enhanced.wav, plots, metrics.json, routing.json
│   ├── manifests/                Generated training/val manifests
│   └── reports/                  Aggregated reports + demo bundle
├── checkpoints/                  Policy + (optional) MossFormer ckpts (gitignored)
├── DEMO.md                       Demo cheat-sheet (architecture, numbers, run)
├── README.md                     This file
├── requirements.txt · requirements-optional.txt
├── environment.yml               Conda env spec
├── setup.py
└── .gitattributes / .gitignore
```

---

## 5. Setup

### 5.1 Prerequisites

- Python **3.9+** (3.10+ recommended for full expert compatibility, especially `resemble_enhance`).
- Node.js **18+** and npm 9+.
- For GPU acceleration: NVIDIA CUDA toolkit + drivers compatible with PyTorch 2.6.

### 5.2 Python environment

```bash
# Option A: conda
conda env create -f environment.yml
conda activate universal-speech-enhancement

# Option B: venv + pip
python -m venv .venv
. .venv/Scripts/activate    # Windows
# . .venv/bin/activate      # Linux/macOS
pip install -r requirements.txt
pip install -r requirements-optional.txt   # MossFormer/Resemble extras
```

Pin notes:

- Keep `torch==2.6.0` and `torchaudio==2.6.0` for DeepFilterNet 3 compatibility.
- For CUDA, install the matching PyTorch wheel from <https://pytorch.org/get-started/locally/>.

### 5.3 Frontend

```bash
cd frontend
npm install
```

### 5.4 Dataset

The system is preconfigured to read from:

```
C:/Users/jsm10/OneDrive - Amrita vishwa vidyapeetham/agentic-speech-enhancement/datasets
```

Override with `system.dataset_root` in `configs/base.yaml`. Expected layout (Interspeech 2025 URGENT):

```
Interspeech 2025 URGENT/
├── official validation set/
│   ├── validation.noisy/noisy/*.flac      (1000 clips)
│   └── validation.clean/clean/*.flac      (1000 clips)
└── blind_test set/
    └── blind_test.noisy/noisy/*.flac      (900 clips)
```

(Optional) `HF urgent2025-sqa/data/nonblind_test/*.parquet` adds `hf_ceiling_*` metadata columns when building manifests.

### 5.5 Checkpoints

- **Policy** — `checkpoints/policy_best.pt` is auto-loaded by `EnhancementPipeline`. If missing, the router runs with random-init weights and a warning; the **dynamic router still works** because it scores candidates with DNSMOS/UTMOS, so you’ll get sensible enhancement even before training.
- **MossFormer2** (optional) — drop a `*moss*former*.pt|.pth|.ckpt` under `checkpoints/` or `models/`, or set `system.mossformer_checkpoint`.

---

## 6. Running the system

### 6.1 One-shot demo launcher (Windows)

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\start_demo.ps1
```

Starts the backend on `:8001` and Vite frontend on `:5173`, polls `/health` until ready, and wires `VITE_API_BASE_URL` correctly.

### 6.2 Manual

```bash
# terminal 1 — backend
uvicorn backend.app.main:app --host 0.0.0.0 --port 8001
# (or :8000 — set VITE_API_BASE_URL accordingly)

# terminal 2 — frontend
cd frontend
VITE_API_BASE_URL=http://127.0.0.1:8001 npm run dev
```

Open <http://127.0.0.1:5173>, drop a `.wav/.flac/.mp3`, click **Enhance Audio**.

### 6.3 CLI inference (no UI)

```bash
python scripts/run_inference.py --input path/to/noisy.wav
```

Writes everything (audio, metrics, plots, routing log) into `outputs/<run_id>/`.

### 6.4 Dataset-free smoke test

```bash
python scripts/demo_smoke.py 8
```

Pulls 8 random clips from the validation noisy/clean folders, runs the production pipeline, prints `expert / strength / ΔDNSMOS / ΔSIG / ΔBAK / ΔUTMOS` per clip and the routing distribution.

---

## 7. Training

### 7.1 Build manifests

```bash
# 1. discover paired noisy/clean files
python scripts/build_paired_manifest.py
#   → outputs/manifests/paired_train.csv, paired_val.csv

# 2. enrich with composite-score oracle labels (sweeps strength)
python scripts/compute_oracle_labels.py \
    --manifest outputs/manifests/paired_train.csv \
    --output  outputs/manifests/paired_train_oracle.csv \
    --strengths 0.6,1.0
# composite = 0.6·PESQ_wb + 0.3·(STOI·5) + 0.1·tanh(SI_SDR/15)
# resilient: skips bad files, flushes every 25 rows, auto-resume if output exists

# 3. (optional) "do no harm" tie-break: prefer BYPASS unless margin is exceeded
python scripts/rederive_oracle_labels.py \
    --in outputs/manifests/paired_train_oracle.csv \
    --out outputs/manifests/paired_train_oracle.csv \
    --margin 0.005
```

### 7.2 Train the policy

```bash
python scripts/train_policy.py --config configs/full_train_paired.yaml
```

- Frozen WavLM encoder + small residual head (~2.67 M trainable params).
- Mixed precision, gradient clipping, cosine LR, early stopping.
- Inverse-frequency class weights with median fallback for absent classes.
- TensorBoard logs under `logs/tensorboard/policy`.
- `checkpoints/policy_best.pt` and `policy_last.pt` written.
- Reports: `outputs/reports/training_history.csv`, `model_performance.csv`, confusion matrices.

### 7.3 Plot training curves

```bash
python scripts/plot_training_results.py
```

Produces loss curves, accuracy plots, and the consolidated `training_dashboard.png`.

---

## 8. Evaluation

### 8.1 Blind batch (recommended)

```powershell
python scripts/eval_blind_batch.py `
    --input-dir "<path-to>/blind_test set/blind_test.noisy/noisy" `
    --glob "*.flac" `
    --output-dir outputs/reports/my_blind_eval `
    --resume
```

Writes:

- `summary.csv` — per-clip telemetry (chosen expert, ΔDNSMOS/SIG/BAK/UTMOS, candidates, decision reason, …).
- `per_file.jsonl` — same data line-by-line for streaming.
- `leaderboard_experts.csv` — per-expert files-served / mean deltas / top-rank match rate.
- `top20_dnsmos_gain.csv` — biggest wins.
- `report.md` — drop-in markdown summary with the leaderboard table.

`--resume` makes it idempotent: only unprocessed files run, and the report is regenerated from `summary.csv`.

### 8.2 Manifest-driven evaluation

```bash
python scripts/evaluate_dataset.py --config configs/base.yaml
```

Runs over the validation manifest with full metrics. Aggregations via `scripts/plot_eval_aggregate.py`.

### 8.3 Pytest

```bash
python -m pytest tests/ -q
```

Currently a backend-health smoke test; designed to expand.

---

## 9. Configuration

`configs/base.yaml` (default) — inference + dynamic routing.

```yaml
system:
  device: cuda                # cuda | cpu
  mixed_precision: true       # AMP for the policy forward
  sample_rate: 16000
  dataset_root: "<path>/datasets"
  output_root: "outputs"
  mossformer_checkpoint: ""   # auto-resolved if blank
  policy_checkpoint: "checkpoints/policy_best.pt"
  dynamic_routing: true       # turn off to bypass speculate-and-measure

policy:
  wavlm_name: "microsoft/wavlm-base-plus"
  hidden_dim: 384
  num_heads: 8                # legacy field; trunk is MLP, kept for compat
  num_layers: 3               # depth of the residual MLP trunk
  dropout: 0.30
  num_actions: 4

training:
  batch_size: 8
  epochs: 30
  lr: 1.0e-4
  weight_decay: 1.0e-5
  early_stopping_patience: 5
  gradient_clip_norm: 1.0
  distributed: false
  train_manifest: "outputs/manifests/train.csv"
  val_manifest:   "outputs/manifests/val.csv"
```

`configs/full_train_paired.yaml` overrides the manifest paths, increases epochs, enables soft labels, and uses cosine LR.

---

## 10. API reference

Backend (FastAPI on `:8001` by default; see also [`backend/API.md`](backend/API.md)).

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /enhance`

Multipart form:

- `file` — `.wav` / `.flac` / `.mp3`.

200 response (`EnhancementResponse`):

```jsonc
{
  "id": "ac81633e",
  "input_path": "outputs/uploads/clip.flac",
  "output_audio_path": "outputs/ac81633e/enhanced.wav",
  "metrics_path":      "outputs/ac81633e/metrics.json",
  "routing_log_path":  "outputs/ac81633e/routing.json",
  "csv_report_path":   "outputs/ac81633e/summary.csv",
  "plots": [
    "outputs/ac81633e/waveform.png",
    "outputs/ac81633e/spectrogram.png",
    "outputs/ac81633e/policy_probs.png"
  ],
  "routing": {
    "expert": "WienerFilter",
    "strength": 0.95,
    "refine": false,
    "confidence": 0.42,
    "probabilities": {"DeepFilterNet3": 0.31, "ResembleEnhance": 0.24, "MossFormer2": 0.03, "BYPASS": 0.42},
    "reason": "snr=...",
    "policy_advice": {"expert": "BYPASS", "strength": 0.62, "refine": false, "confidence": 0.42},
    "dynamic_candidates": [ /* ≤ ~25 candidates with rank_score, ovrl, sig, bak, utmos */ ],
    "decision_reason": "WienerFilter@0.95 rank=2.261 (OVRL=1.980, UTMOS=3.299, utmos_reliable=False) beat BYPASS rank=1.122",
    "dynamic_routing": true,
    "preprocess": { "dc_offset_removed": ..., "rms_db_in": ..., "rms_db_out": ..., "loudness_lufs_in": ..., "loudness_lufs_out": ..., "high_pass_hz": 60.0, "target_lufs": -23.0 },
    "timings": { "load_ms": 5.4, "policy_ms": 60.1, "enhance_ms": 5870.2, "metrics_ms": 312.0, "total_ms": 6248.7, "audio_seconds": 5.41, "rtf": 1.155 },
    "distortion_summary": { "snr_db": ..., "reverb": ..., "clip": ..., "noise_level": ..., "codec": ..., "intelligibility": ... }
  },
  "metrics": {
    "original": { "dnsmos": 1.10, "dnsmos_sig": 1.17, "dnsmos_bak": 1.13, "dnsmos_p808": 2.21, "utmos": 3.33, "pesq": null, "stoi": null, "si_sdr": null },
    "enhanced": { "dnsmos": 1.98, "dnsmos_sig": 2.76, "dnsmos_bak": 2.55, "dnsmos_p808": 2.54, "utmos": 3.30, "pesq": 1.30, "stoi": 0.85, "si_sdr": 4.65 },
    "improvement": { "dnsmos": +0.88, "dnsmos_sig": +1.59, "dnsmos_bak": +1.42, "dnsmos_p808": +0.34, "utmos": -0.03 },
    "similarity_vs_noisy_input": { "pesq": 1.30, "stoi": 0.85, "si_sdr": 4.65 }
  }
}
```

Static files mount: `GET /outputs/<run_id>/<file>` serves any artifact written by the pipeline (the dashboard uses this to play `enhanced.wav` and embed the PNGs).

---

## 11. Results

### 11.1 Blind eval — 100 clips from the URGENT validation set

Generated by `scripts/eval_blind_batch.py` and saved at `outputs/reports/demo_blind_eval/`.

| metric | value |
| --- | ---: |
| Files processed | **100** |
| Mean ΔDNSMOS (OVRL) | **+0.72** |
| Mean ΔSIG | **+0.56** |
| Mean ΔBAK | **+1.54** |
| Mean ΔUTMOS22 | **−0.60** *(heuristic fallback; UTMOS not in routing blend)* |
| Chosen-is-top-rank rate | **100.00 %** |
| Pareto-dominated selections | **0** |

Routing distribution:

| chosen expert | files | mean ΔDNSMOS | mean ΔSIG | mean ΔBAK |
| --- | ---: | ---: | ---: | ---: |
| DeepFilterNet3 | 57 | +0.74 | +0.47 | +1.60 |
| NoiseReduce | 26 | +0.58 | +0.27 | +1.40 |
| WienerFilter | 11 | +1.01 | +1.51 | +1.64 |
| SpectralSubtraction | 4 | +0.55 | +0.84 | +1.38 |
| SpectralGate | 2 | +0.89 | +1.14 | +1.49 |
| BYPASS | 0 | — | — | — |

Every chosen output beat `BYPASS` at the rank-score level — the dynamic router never made an input worse. UTMOS22 hub model fell back on this machine (a name clash between our `speechmos` ONNX package and `tarepan/SpeechMOS`’s internal `speechmos.utmos22`), so the router automatically dropped UTMOS from the blend; the negative ΔUTMOS22 column is the heuristic fallback talking, not a regression.

### 11.2 Single-clip example (`fileid_102.flac`)

| signal | DNSMOS OVRL | SIG | BAK | UTMOS22 | PESQ-WB vs noisy | STOI vs noisy | SI-SDR vs noisy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 1.10 | 1.17 | 1.13 | 3.33 | — | — | — |
| Enhanced (`WienerFilter @ 0.95`) | 1.98 | 2.76 | 2.55 | 3.30 | 1.30 | 0.85 | +4.65 dB |
| **Δ** | **+0.88** | **+1.59** | **+1.42** | -0.03 | — | — | — |

Decision reason emitted by the API: `WienerFilter@0.95 rank=2.261 (OVRL=1.980, UTMOS=3.299, utmos_reliable=False) beat BYPASS rank=1.122`.

---

## 12. Implementation notes & design choices

- **Why a dynamic router on top of a trained policy?** A trained classifier on noisy speech is right most of the time but quietly catastrophic on the long-tail. The dynamic router is a measurement-based safety harness: it never picks an output that didn’t actually outperform `BYPASS` on DNSMOS, so the worst-case is “we returned the input.”
- **Why `0.45·OVRL + 0.20·SIG + 0.15·BAK + 0.20·UTMOS22`?** OVRL is the umbrella P.835 score; SIG is the speech-quality component (penalises over-suppression); BAK rewards background removal; UTMOS22 is a different-architecture, MOS-trained predictor that catches artifacts (musical noise) DNSMOS misses. When UTMOS isn’t loaded, we redistribute weights as `0.60·OVRL + 0.25·SIG + 0.15·BAK` rather than trust a heuristic.
- **Single-pass limiting.** The router scores a limited copy but **caches the unlimited audio**, so `EnhancementPipeline.run()` applies the brick-wall limiter exactly once. This avoids subtle level discrepancies between “what we scored” and “what we shipped.”
- **`utmos_is_reliable()` gate.** Code-path explicit. The decision-reason text always reports `utmos_reliable=...` so debugging is one log line away.
- **“Do-no-harm” margin = 0.02 OVRL.** Mirrors `scripts/rederive_oracle_labels.py`. Stops the router from picking an expert that won by 0.005 over `BYPASS` on a measurement that fluctuates by more than that.
- **Active-expert filter.** Cheap, but it’s what keeps latency honest. When MossFormer2/Resemble aren’t installed, they don’t show up in the candidate list at all.
- **Frozen WavLM.** Trains in minutes on a laptop GPU, doesn’t over-fit a ~few-thousand-clip oracle dataset, and matches well-known SUPERB-style downstream practice.
- **Composite oracle score for training labels.** `0.6·PESQ_wb + 0.3·(STOI·5) + 0.1·tanh(SI_SDR/15)` — chosen so the three components live on roughly the same scale and SI-SDR can’t dominate via outliers.
- **Windows-friendly throughout.** The launcher, the demo smoke, and the eval scripts all work in PowerShell. CRLF normalization is handled via `.gitattributes`.

---

## License & attribution

Third-party model weights retain their original licenses:

- **DeepFilterNet 3** — Schroeter et al., LGPL.
- **WavLM-base-plus** — Microsoft, MIT.
- **DNSMOS / P.808** — Microsoft DNS-Challenge, MIT.
- **UTMOS22** — Saeki et al. (Interspeech 2022), via tarepan/SpeechMOS hub config.
- **MossFormer2** — ClearerVoice-Studio, model license per their repo.
- **noisereduce** — Sainburg et al., MIT.

The wrapper code in this repository is intentionally pluggable so you can pin specific pretrained checkpoints per expert without touching the routing logic.
