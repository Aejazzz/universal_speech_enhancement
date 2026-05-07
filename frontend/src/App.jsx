import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { API_BASE_URL, enhanceAudio } from "./api";

const METRICS_HELP =
  "dnsmos/utmos: no-reference proxies on input vs output. pesq/stoi/si_sdr on the enhanced row are vs the *noisy input* " +
  "(waveform similarity), not vs clean speech — a big SI-SDR drop often means the signal changed shape, not that it sounds worse. " +
  "Original row leaves pesq/stoi/si_sdr blank (not self-identity vs noisy). improvement uses null for pesq/stoi/si_sdr deltas; " +
  "see similarity_vs_noisy_input for the same similarity triple. When you pass a clean reference to the API/CLI, vs_clean_reference adds true intrusive metrics.";

function JsonBlock({ value }) {
  return (
    <pre className="mt-3 overflow-x-auto rounded-xl bg-slate-950 p-4 text-left text-xs leading-relaxed text-emerald-200/95">
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const originalAudioUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);

  const outputUrl = (runId, name) => `${API_BASE_URL}/outputs/${runId}/${name}`;

  const onSubmit = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const data = await enhanceAudio(file);
      setResult(data);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-6 md:p-10">
      <div className="mx-auto max-w-6xl space-y-8">
        <header>
          <h1 className="text-3xl font-bold tracking-tight text-white md:text-4xl">
            Agentic Universal Speech Enhancement Policy Learning
          </h1>
          <p className="mt-2 text-sm text-slate-400">
            Upload noisy speech → routed enhancement → dashboards, plots, and structured metrics below.
          </p>
        </header>

        <section className="rounded-2xl border border-slate-700 bg-slate-900/80 p-5 shadow-lg">
          <h2 className="mb-4 text-lg font-semibold text-slate-100">Upload Panel</h2>
          <div className="flex flex-wrap items-center gap-4">
            <input
              type="file"
              accept=".wav,.mp3,.flac,audio/*"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="block text-sm text-slate-300 file:mr-3 file:rounded-lg file:border-0 file:bg-slate-700 file:px-4 file:py-2 file:text-sm file:font-medium file:text-slate-100"
            />
            {file && (
              <span className="text-sm font-medium text-emerald-300">
                {file.name}
              </span>
            )}
            <button
              onClick={onSubmit}
              disabled={!file || loading}
              className="rounded-xl bg-emerald-500 px-5 py-2.5 font-semibold text-black shadow transition hover:bg-emerald-400 disabled:pointer-events-none disabled:opacity-40"
            >
              {loading ? "Enhancing..." : "Enhance Audio"}
            </button>
          </div>
        </section>

        {result && (
          <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="space-y-8">
            <section className="rounded-2xl border border-slate-700 bg-slate-900/80 p-5">
              <h2 className="mb-4 text-lg font-semibold">Audio Playback</h2>
              <div className="grid gap-6 md:grid-cols-2">
                <div>
                  <p className="mb-2 text-sm font-medium text-slate-400">Original</p>
                  {originalAudioUrl && (
                    <audio controls className="w-full rounded-lg" src={originalAudioUrl} />
                  )}
                </div>
                <div>
                  <p className="mb-2 text-sm font-medium text-slate-400">Enhanced</p>
                  <audio controls className="w-full rounded-lg" src={outputUrl(result.id, "enhanced.wav")} />
                </div>
              </div>
            </section>

            <section className="rounded-2xl border border-slate-700 bg-slate-900/80 p-5">
              <h2 className="mb-4 text-lg font-semibold">Enhancement Dashboard</h2>
              <div className="space-y-2 text-slate-200">
                <p>
                  <span className="text-slate-500">Expert:</span>{" "}
                  <span className="font-semibold text-white">{result.routing.expert}</span>
                </p>
                <p>
                  <span className="text-slate-500">Strength:</span>{" "}
                  {Number(result.routing.strength).toFixed(3)}
                </p>
                <p>
                  <span className="text-slate-500">Refinement:</span>{" "}
                  {String(result.routing.refine)}
                </p>
                <p>
                  <span className="text-slate-500">Confidence:</span>{" "}
                  {Number(result.routing.confidence).toFixed(4)}
                </p>
                <p className="pt-2 text-sm text-slate-300">
                  <span className="text-slate-500">Reason:</span> {result.routing.reason}
                </p>
              </div>

              <p className="mt-6 text-xs font-semibold uppercase tracking-wide text-slate-500">
                Routing probabilities (JSON)
              </p>
              <JsonBlock value={result.routing.probabilities} />
            </section>

            <section className="rounded-2xl border border-slate-700 bg-slate-900/80 p-5">
              <h2 className="mb-3 text-lg font-semibold">Metrics</h2>
              <p className="text-sm leading-relaxed text-slate-400">{METRICS_HELP}</p>
              <JsonBlock value={result.metrics} />
            </section>

            <section className="rounded-2xl border border-slate-700 bg-slate-900/80 p-5">
              <h2 className="mb-4 text-lg font-semibold">Visualizations</h2>
              <div className="grid gap-4 md:grid-cols-3">
                <figure>
                  <img
                    className="rounded-xl border border-slate-700"
                    src={outputUrl(result.id, "waveform.png")}
                    alt="Waveform comparison"
                  />
                  <figcaption className="mt-2 text-center text-xs text-slate-500">
                    Waveform comparison
                  </figcaption>
                </figure>
                <figure>
                  <img
                    className="rounded-xl border border-slate-700"
                    src={outputUrl(result.id, "spectrogram.png")}
                    alt="Spectrogram comparison"
                  />
                  <figcaption className="mt-2 text-center text-xs text-slate-500">
                    Spectrogram comparison
                  </figcaption>
                </figure>
                <figure>
                  <img
                    className="rounded-xl border border-slate-700"
                    src={outputUrl(result.id, "policy_probs.png")}
                    alt="Policy probabilities"
                  />
                  <figcaption className="mt-2 text-center text-xs text-slate-500">
                    Policy probabilities
                  </figcaption>
                </figure>
              </div>
            </section>
          </motion.div>
        )}
      </div>
    </div>
  );
}
