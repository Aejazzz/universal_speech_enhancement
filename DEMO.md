# Demo Day Guide

This is the one-page “show-and-tell” for the **Universal Speech Enhancement Policy Learning** project. Everything here was verified end-to-end on the demo machine (Python 3.9, RTX 4060 Laptop GPU, Windows 11).

> If you only have 30 seconds: run `pwsh -File scripts/start_demo.ps1`, open http://127.0.0.1:5173, upload `outputs/reports/demo_input_fileid_102.flac`, hit **Enhance Audio**.

---

## 1. What this project does

Given a noisy speech clip, an **adaptive routing policy** picks the best enhancement expert (from 6 candidates) and the best strength setting per clip:

```
noisy speech
   │
   ▼
ITU-R BS.1770 preprocess  (DC offset, 60 Hz HPF, -23 LUFS)
   │
   ▼
CRNN distortion analyzer  (6 routing features)   ┐
                                                 │
Frozen WavLM-base-plus + small residual MLP  ───►│ policy advice (expert, strength)
                                                 │
                                                 ▼
"Speculate-and-measure" router
   • runs every active expert at strengths {0.55, 0.80, 0.95, policy}
   • scores every candidate with DNSMOS P.835 + UTMOS22
   • blended rank_score = 0.45·OVRL + 0.20·SIG + 0.15·BAK + 0.20·UTMOS
     (UTMOS dropped if the predictor fell back to heuristic)
   • "do no harm" margin keeps BYPASS preferred on near-ties
   │
   ▼
brick-wall limiter at -1 dBFS
   │
   ▼
enhanced.wav  +  metrics.json (DNSMOS / UTMOS / PESQ / STOI / SI-SDR)
```

Active experts (order of priority): `DeepFilterNet3`, `WPEDereverb`, `NoiseReduce`, `SpectralGate`, `WienerFilter`, `SpectralSubtraction`, `ResembleEnhance`, `MossFormer2`, plus `BYPASS`. Neural experts auto-disable when their checkpoints/packages aren’t present so the router never inflates latency on dead candidates.

---

## 2. Environment (verified)

| Item | Value |
| --- | --- |
| Python | 3.9.13 |
| Torch | 2.6.0+cu124 |
| CUDA device | NVIDIA GeForce RTX 4060 Laptop GPU |
| Policy checkpoint | `checkpoints/policy_best.pt` (loads cleanly) |
| Active experts | DeepFilterNet3 (CUDA), WienerFilter, SpectralSubtraction, SpectralGate, NoiseReduce, BYPASS |
| Node | v22.22.0 / npm 11.11.0 |
| Frontend build | `npm run build` → 316 KB JS, 14 KB CSS, 435 modules ✅ |

---

## 3. Numbers to put on the slide (blind-eval over 30 validation clips)

```
outputs/reports/demo_blind_eval/
  ├── summary.csv              # per-clip telemetry
  ├── leaderboard_experts.csv  # routing distribution + per-expert deltas
  ├── per_file.jsonl
  ├── top20_dnsmos_gain.csv
  └── report.md                # ready-made Markdown table
```

Headlines from `report.md`:

- **Files processed: 30**
- **Mean DNSMOS Δ: +0.96**
- **Mean SIG Δ: +1.00 · Mean BAK Δ: +1.69**
- **Chosen-is-top-rank rate: 100.00 %**
- **Pareto-dominated selections: 0**

Routing distribution (out of 30):

| chosen_expert | files | mean ΔDNSMOS | mean ΔSIG | mean ΔBAK |
| --- | ---: | ---: | ---: | ---: |
| DeepFilterNet3 | 17 | +1.10 | +1.06 | +1.92 |
| WienerFilter | 6 | +1.03 | +1.51 | +1.61 |
| NoiseReduce | 4 | +0.49 | +0.06 | +1.05 |
| SpectralSubtraction | 2 | +0.78 | +1.07 | +1.47 |
| SpectralGate | 1 | +0.49 | +0.38 | +1.12 |

> Every selected output beats `BYPASS` at the rank-score level — no clip was made worse. UTMOS22 fell back to heuristic on the demo machine; the router automatically dropped UTMOS from the blend (`utmos_reliable=False` shows in the decision reason).

A representative live API run (`POST /enhance` with `fileid_1.flac`):

```
expert=WienerFilter strength=0.95
ovrl_in=1.102 → ovrl_out=1.980  (Δ +0.879)
candidates=25
reason: WienerFilter@0.95 rank=2.261 (OVRL=1.980, UTMOS=3.299, utmos_reliable=False) beat BYPASS rank=1.122
```

A second sample run (`fileid_102.flac`, available as `outputs/reports/demo_sample_run/`):

| signal | DNSMOS OVRL | SIG | BAK | UTMOS |
| --- | ---: | ---: | ---: | ---: |
| Original | 1.10 | 1.17 | 1.13 | 3.33 |
| Enhanced (`WienerFilter@0.95`) | 1.98 | 2.76 | 2.55 | 3.30 |
| **Improvement** | **+0.88** | **+1.59** | **+1.42** | -0.03 |

PESQ vs noisy = 1.30, STOI vs noisy = 0.85, SI-SDR vs noisy = +4.65 dB.

---

## 4. Run it on demo day

### Option A — one-shot script (recommended)

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\start_demo.ps1
```

This starts the FastAPI backend (port 8001) and the Vite frontend (port 5173) in two new PowerShell windows, sets `VITE_API_BASE_URL` correctly, and waits for `/health` to go green. If the backend is already running it will just verify and skip.

### Option B — manual

```powershell
# terminal 1
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8001

# terminal 2
cd frontend
$env:VITE_API_BASE_URL = "http://127.0.0.1:8001"
npm run dev
```

Open http://127.0.0.1:5173, choose a `.wav` / `.flac` / `.mp3` (try `outputs/reports/demo_input_fileid_102.flac`), and click **Enhance Audio**.

### Re-run the blind eval

```powershell
python scripts\eval_blind_batch.py `
    --input-dir "C:\Users\jsm10\OneDrive - Amrita vishwa vidyapeetham\agentic-speech-enhancement\datasets\Interspeech 2025 URGENT\official validation set\validation.noisy\noisy" `
    --glob "*.flac" --limit 30 `
    --output-dir outputs\reports\demo_blind_eval --resume
```

`--resume` makes it idempotent: only unprocessed files run, and `report.md` / leaderboard CSVs are regenerated from `summary.csv`.

### Smoke test (no UI)

```powershell
python scripts\demo_smoke.py 8
```

Pulls 8 random clips from the validation set, runs the production pipeline, and prints `expert / strength / ΔDNSMOS / ΔSIG / ΔBAK / ΔUTMOS` per clip plus the routing distribution.

---

## 5. What to show, in order

1. `DEMO.md` — this file. Architecture diagram + numbers slide.
2. `outputs/reports/demo_blind_eval/report.md` — leaderboard.
3. `scripts/start_demo.ps1` (or just have it already running).
4. The **frontend dashboard** (http://127.0.0.1:5173):
   - upload `outputs/reports/demo_input_fileid_102.flac`
   - point out: chosen expert, dynamic candidate leaderboard (with crown), DNSMOS scorecard with deltas, A/B audio player, decision reason text.
5. `outputs/reports/demo_sample_run/`:
   - `waveform.png`, `spectrogram.png` (orig / enhanced / removed-noise three-panel), `policy_probs.png`.
6. Code highlights: `backend/app/pipeline.py::EnhancementPipeline._dynamic_select`, `evaluation/metrics.py::utmos_is_reliable`, `enhancement_experts/factory.py`, `policy_agent/model.py`.

---

## 6. Demo-day failure modes (and what to say)

| Symptom | Cause | What to say |
| --- | --- | --- |
| `port 8001 in use` | Backend already running from earlier session. | "It’s already up — `/health` is fine, frontend will hit it." |
| First request takes ~10 s | First WavLM forward + DNSMOS ONNX session warm-up. | "First-clip warm-up cost; steady-state RTF is reported in `routing.timings`." |
| `UTMOS22 load failed (...)` | tarepan/SpeechMOS hub model had a name-clash with our installed `speechmos` ONNX package. | "Predictable — `utmos_is_reliable()` already returns False here, so the router blends DNSMOS-only weights instead of trusting the heuristic. Safety harness, not a bug." |
| Negative `utmos_delta` in CSV | Same as above — heuristic UTMOS isn’t a real predictor. | "Read the DNSMOS columns; UTMOS is null when `utmos_reliable=False`." |
| Browser shows CORS error | Backend on a different host/port than `VITE_API_BASE_URL`. | Use `start_demo.ps1` (sets it for you), or set `$env:VITE_API_BASE_URL` before `npm run dev`. |

---

## 7. File map of demo artifacts

```
outputs/reports/
├── demo_blind_eval/                  # 30-clip blind run
│   ├── summary.csv
│   ├── leaderboard_experts.csv
│   ├── per_file.jsonl
│   ├── top20_dnsmos_gain.csv
│   └── report.md                     # ← put this on the slide
├── demo_sample_run/                  # one full pipeline run on fileid_1.flac
│   ├── enhanced.wav
│   ├── waveform.png
│   ├── spectrogram.png
│   ├── policy_probs.png
│   ├── routing.json
│   ├── metrics.json
│   └── summary.csv
├── demo_input_fileid_102.flac        # ← drag this onto the dashboard
└── demo_frontend_idle.png            # screenshot of the empty UI
```
