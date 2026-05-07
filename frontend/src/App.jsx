import { useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { API_BASE_URL, enhanceAudio } from "./api";

const METRICS_HELP =
  "DNSMOS P.835 (SIG/BAK/OVRL) and P.808 are reference-free MOS predictors trained by Microsoft on the DNS challenge data — higher is better. " +
  "PESQ/STOI/SI-SDR on the enhanced row are computed against the noisy input itself (waveform similarity), so a big SI-SDR drop usually means " +
  "the enhancer changed the signal a lot, not that it sounds worse. With a clean reference, vs_clean_reference adds the standard intrusive metrics.";

const fmt = (v, d = 3) =>
  v === null || v === undefined || Number.isNaN(Number(v)) ? "—" : Number(v).toFixed(d);

const fmtSigned = (v, d = 3) => {
  const x = Number(v);
  if (v === null || v === undefined || Number.isNaN(x)) return "—";
  return (x >= 0 ? "+" : "") + x.toFixed(d);
};

function MetricBar({ label, value, max = 5, color = "emerald" }) {
  const pct = Math.max(0, Math.min(100, (Number(value) / max) * 100));
  const colorClass = {
    emerald: "from-emerald-400 to-emerald-600",
    sky: "from-sky-400 to-sky-600",
    violet: "from-violet-400 to-violet-600",
    amber: "from-amber-400 to-amber-600",
  }[color];
  return (
    <div>
      <div className="flex items-baseline justify-between text-xs text-slate-300">
        <span>{label}</span>
        <span className="font-mono text-white">{fmt(value)}</span>
      </div>
      <div className="mt-1 h-2 overflow-hidden rounded-full bg-slate-800">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className={`h-full bg-gradient-to-r ${colorClass}`}
        />
      </div>
    </div>
  );
}

function DeltaPill({ value, suffix = "" }) {
  const x = Number(value);
  if (value === null || value === undefined || Number.isNaN(x)) {
    return <span className="rounded-md bg-slate-700/60 px-1.5 py-0.5 text-xs text-slate-400">—</span>;
  }
  const positive = x > 0;
  const cls = positive
    ? "bg-emerald-500/20 text-emerald-300"
    : x < 0
    ? "bg-rose-500/20 text-rose-300"
    : "bg-slate-700/60 text-slate-300";
  return (
    <span className={`rounded-md px-1.5 py-0.5 text-xs font-mono ${cls}`}>
      {fmtSigned(x)}
      {suffix}
    </span>
  );
}

function ScoreCard({ title, original, enhanced, improvement, color }) {
  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
      <div className="mb-3 flex items-baseline justify-between">
        <h3 className="text-sm font-semibold text-slate-200">{title}</h3>
        <DeltaPill value={improvement} />
      </div>
      <MetricBar label="Original" value={original} color="amber" />
      <div className="h-2" />
      <MetricBar label="Enhanced" value={enhanced} color={color} />
    </div>
  );
}

function CandidateRow({ candidate, isWinner }) {
  return (
    <div
      className={`flex items-center justify-between rounded-lg border px-3 py-2 text-sm ${
        isWinner
          ? "border-emerald-500/60 bg-emerald-500/10"
          : "border-slate-700 bg-slate-900/40"
      }`}
    >
      <div className="flex items-center gap-2">
        {isWinner && <span className="text-xs">👑</span>}
        <span className="font-mono text-slate-200">{candidate.expert}</span>
        <span className="text-xs text-slate-500">@ {Number(candidate.strength).toFixed(2)}</span>
      </div>
      <div className="flex items-center gap-3 font-mono text-xs">
        {candidate.rank_score !== undefined && (
          <span className="text-violet-300">
            rank <span className="font-semibold">{fmt(candidate.rank_score)}</span>
          </span>
        )}
        <span className="text-slate-400">
          OVRL <span className="text-white">{fmt(candidate.dnsmos)}</span>
        </span>
        {candidate.dnsmos_sig !== undefined && (
          <span className="text-slate-400">
            SIG <span className="text-white">{fmt(candidate.dnsmos_sig)}</span>
          </span>
        )}
        {candidate.dnsmos_bak !== undefined && (
          <span className="text-slate-400">
            BAK <span className="text-white">{fmt(candidate.dnsmos_bak)}</span>
          </span>
        )}
        {candidate.utmos !== undefined && (
          <span className="text-slate-400">
            UTMOS <span className="text-white">{fmt(candidate.utmos)}</span>
          </span>
        )}
      </div>
    </div>
  );
}

function JsonBlock({ value, height = "max-h-72" }) {
  return (
    <pre
      className={`mt-3 overflow-auto rounded-xl bg-slate-950 p-4 text-left text-xs leading-relaxed text-emerald-200/95 ${height}`}
    >
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showJson, setShowJson] = useState(false);
  const originalAudioUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);

  const outputUrl = (runId, name) => `${API_BASE_URL}/outputs/${runId}/${name}`;

  const onSubmit = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const data = await enhanceAudio(file);
      setResult(data);
    } catch (err) {
      setError(err?.message || "Enhancement failed");
    } finally {
      setLoading(false);
    }
  };

  const routing = result?.routing;
  const metrics = result?.metrics;
  const candidates = routing?.dynamic_candidates || [];
  const sortedCandidates = [...candidates].sort(
    (a, b) =>
      Number(b.rank_score ?? b.dnsmos ?? 0) - Number(a.rank_score ?? a.dnsmos ?? 0)
  );
  const winnerKey = routing
    ? `${routing.expert}|${Number(routing.strength).toFixed(2)}`
    : null;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-6 md:p-10">
      <div className="mx-auto max-w-7xl space-y-8">
        <header>
          <motion.h1
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-gradient-to-r from-emerald-300 via-sky-300 to-violet-300 bg-clip-text text-3xl font-bold tracking-tight text-transparent md:text-4xl"
          >
            Agentic Universal Speech Enhancement Policy Learning
          </motion.h1>
          <p className="mt-2 text-sm text-slate-400">
            Upload noisy speech → adaptive routing across 6 experts (3 neural + 3 classical FFT) →
            ITU-R BS.1770 loudness norm + brick-wall limit → DNSMOS / PESQ / STOI / SI-SDR scoring.
          </p>
        </header>

        <section className="rounded-2xl border border-slate-700 bg-slate-900/80 p-5 shadow-lg">
          <h2 className="mb-4 text-lg font-semibold text-slate-100">Upload Panel</h2>
          <div className="flex flex-wrap items-center gap-4">
            <input
              type="file"
              accept=".wav,.mp3,.flac,audio/*"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="block text-sm text-slate-300 file:mr-3 file:rounded-lg file:border-0 file:bg-slate-700 file:px-4 file:py-2 file:text-sm file:font-medium file:text-slate-100 hover:file:bg-slate-600"
            />
            {file && (
              <span className="text-sm font-medium text-emerald-300">{file.name}</span>
            )}
            <button
              onClick={onSubmit}
              disabled={!file || loading}
              className="rounded-xl bg-emerald-500 px-5 py-2.5 font-semibold text-black shadow transition hover:bg-emerald-400 disabled:pointer-events-none disabled:opacity-40"
            >
              {loading ? "Enhancing…" : "Enhance Audio"}
            </button>
          </div>
          {error && (
            <p className="mt-3 rounded-lg bg-rose-500/10 px-3 py-2 text-sm text-rose-300">
              {error}
            </p>
          )}
        </section>

        <AnimatePresence>
          {result && routing && metrics && (
            <motion.div
              key={result.id}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="space-y-8"
            >
              {/* Audio Playback */}
              <section className="rounded-2xl border border-slate-700 bg-slate-900/80 p-5">
                <h2 className="mb-4 text-lg font-semibold">Audio A/B</h2>
                <div className="grid gap-6 md:grid-cols-2">
                  <div>
                    <p className="mb-2 text-sm font-medium text-slate-400">Original (uploaded)</p>
                    {originalAudioUrl && (
                      <audio controls className="w-full rounded-lg" src={originalAudioUrl} />
                    )}
                  </div>
                  <div>
                    <p className="mb-2 text-sm font-medium text-emerald-400">
                      Enhanced ({routing.expert} @ {Number(routing.strength).toFixed(2)})
                    </p>
                    <audio
                      controls
                      className="w-full rounded-lg"
                      src={outputUrl(result.id, "enhanced.wav")}
                    />
                  </div>
                </div>
              </section>

              {/* Routing decision */}
              <section className="rounded-2xl border border-slate-700 bg-slate-900/80 p-5">
                <div className="mb-4 flex flex-wrap items-baseline justify-between gap-3">
                  <h2 className="text-lg font-semibold">Routing Decision</h2>
                  <div className="flex items-center gap-2">
                    {routing.timings && (
                      <span className="rounded-md bg-slate-800 px-2 py-0.5 text-xs font-mono text-slate-300">
                        {routing.timings.total_ms} ms total · RTF {routing.timings.rtf} ·{" "}
                        {routing.timings.audio_seconds}s audio
                      </span>
                    )}
                    {routing.dynamic_routing && (
                      <span className="rounded-md bg-violet-500/20 px-2 py-0.5 text-xs font-medium text-violet-300">
                        dynamic speculate-and-measure
                      </span>
                    )}
                  </div>
                </div>
                <div className="grid gap-4 md:grid-cols-3">
                  <div className="rounded-xl border border-emerald-500/40 bg-emerald-500/10 p-4">
                    <p className="text-xs uppercase tracking-wide text-emerald-300/80">
                      Final Choice
                    </p>
                    <p className="mt-1 text-2xl font-bold text-white">{routing.expert}</p>
                    <p className="mt-1 text-sm text-slate-300">
                      strength {Number(routing.strength).toFixed(2)}
                      {routing.refine ? " • +refine" : ""}
                    </p>
                  </div>
                  <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
                    <p className="text-xs uppercase tracking-wide text-slate-400">
                      Policy Advisory
                    </p>
                    <p className="mt-1 text-xl font-semibold text-slate-200">
                      {routing.policy_advice?.expert ?? "—"}
                    </p>
                    <p className="mt-1 text-sm text-slate-400">
                      confidence{" "}
                      {fmt(routing.policy_advice?.confidence ?? routing.confidence, 3)}
                    </p>
                  </div>
                  <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
                    <p className="text-xs uppercase tracking-wide text-slate-400">Why</p>
                    <p className="mt-1 line-clamp-3 text-sm text-slate-200">
                      {routing.decision_reason || routing.reason}
                    </p>
                  </div>
                </div>

                {sortedCandidates.length > 0 && (
                  <div className="mt-5">
                    <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                      Candidate leaderboard ({sortedCandidates.length} ran)
                    </p>
                    <div className="space-y-1.5">
                      {sortedCandidates.slice(0, 24).map((c, i) => {
                        const key = `${c.expert}|${Number(c.strength).toFixed(2)}`;
                        return (
                          <CandidateRow
                            key={`${c.expert}-${c.strength}-${i}`}
                            candidate={c}
                            isWinner={key === winnerKey}
                          />
                        );
                      })}
                    </div>
                  </div>
                )}

                {routing.timings && (
                  <div className="mt-5 grid grid-cols-2 gap-2 rounded-xl bg-slate-950/40 p-3 text-xs font-mono text-slate-400 md:grid-cols-5">
                    <div>load <span className="text-white">{routing.timings.load_ms} ms</span></div>
                    <div>policy <span className="text-white">{routing.timings.policy_ms} ms</span></div>
                    <div>enhance <span className="text-white">{routing.timings.enhance_ms} ms</span></div>
                    <div>metrics <span className="text-white">{routing.timings.metrics_ms} ms</span></div>
                    <div>total <span className="text-emerald-300">{routing.timings.total_ms} ms</span></div>
                  </div>
                )}
              </section>

              {/* DNSMOS scorecards */}
              <section className="rounded-2xl border border-slate-700 bg-slate-900/80 p-5">
                <h2 className="mb-4 text-lg font-semibold">DNSMOS P.835 / P.808 Scorecards</h2>
                <div className="grid gap-4 md:grid-cols-4">
                  <ScoreCard
                    title="OVRL (overall)"
                    original={metrics.original?.dnsmos}
                    enhanced={metrics.enhanced?.dnsmos}
                    improvement={metrics.improvement?.dnsmos}
                    color="emerald"
                  />
                  <ScoreCard
                    title="SIG (signal)"
                    original={metrics.original?.dnsmos_sig}
                    enhanced={metrics.enhanced?.dnsmos_sig}
                    improvement={metrics.improvement?.dnsmos_sig}
                    color="sky"
                  />
                  <ScoreCard
                    title="BAK (background)"
                    original={metrics.original?.dnsmos_bak}
                    enhanced={metrics.enhanced?.dnsmos_bak}
                    improvement={metrics.improvement?.dnsmos_bak}
                    color="violet"
                  />
                  <ScoreCard
                    title="P.808 MOS"
                    original={metrics.original?.dnsmos_p808}
                    enhanced={metrics.enhanced?.dnsmos_p808}
                    improvement={metrics.improvement?.dnsmos_p808}
                    color="emerald"
                  />
                </div>
                <div className="mt-4 grid gap-4 md:grid-cols-2">
                  <ScoreCard
                    title="UTMOS"
                    original={metrics.original?.utmos}
                    enhanced={metrics.enhanced?.utmos}
                    improvement={metrics.improvement?.utmos}
                    color="amber"
                  />
                  <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4 text-sm text-slate-300">
                    <p className="text-xs uppercase tracking-wide text-slate-500">
                      vs noisy similarity
                    </p>
                    <div className="mt-2 grid grid-cols-3 gap-2 font-mono text-xs">
                      <div>
                        PESQ <span className="text-white">{fmt(metrics.enhanced?.pesq)}</span>
                      </div>
                      <div>
                        STOI <span className="text-white">{fmt(metrics.enhanced?.stoi)}</span>
                      </div>
                      <div>
                        SI-SDR{" "}
                        <span className="text-white">{fmt(metrics.enhanced?.si_sdr, 2)} dB</span>
                      </div>
                    </div>
                  </div>
                </div>
                <p className="mt-3 text-xs leading-relaxed text-slate-500">{METRICS_HELP}</p>
              </section>

              {/* Preprocessing & distortion */}
              <section className="grid gap-4 md:grid-cols-2">
                <div className="rounded-2xl border border-slate-700 bg-slate-900/80 p-5">
                  <h2 className="mb-3 text-lg font-semibold">Preprocessing</h2>
                  {routing.preprocess ? (
                    <ul className="space-y-1 text-sm text-slate-300">
                      <li>
                        DC offset removed:{" "}
                        <span className="font-mono text-white">
                          {fmt(routing.preprocess.dc_offset_removed, 5)}
                        </span>
                      </li>
                      <li>
                        High-pass:{" "}
                        <span className="font-mono text-white">
                          {fmt(routing.preprocess.high_pass_hz, 0)} Hz
                        </span>
                      </li>
                      <li>
                        RMS:{" "}
                        <span className="font-mono text-white">
                          {fmt(routing.preprocess.rms_db_in, 1)} → {fmt(routing.preprocess.rms_db_out, 1)} dB
                        </span>
                      </li>
                      <li>
                        Loudness:{" "}
                        <span className="font-mono text-white">
                          {fmt(routing.preprocess.loudness_lufs_in, 1)} →{" "}
                          {fmt(routing.preprocess.loudness_lufs_out, 1)} LUFS (target{" "}
                          {fmt(routing.preprocess.target_lufs, 1)})
                        </span>
                      </li>
                    </ul>
                  ) : (
                    <p className="text-sm text-slate-500">No preprocessing report.</p>
                  )}
                </div>
                <div className="rounded-2xl border border-slate-700 bg-slate-900/80 p-5">
                  <h2 className="mb-3 text-lg font-semibold">Distortion Analysis</h2>
                  {routing.distortion_summary ? (
                    <div className="grid grid-cols-2 gap-3 text-sm text-slate-300">
                      {Object.entries(routing.distortion_summary).map(([k, v]) => (
                        <div key={k} className="flex items-baseline justify-between">
                          <span className="text-slate-500">{k}</span>
                          <span className="font-mono text-white">{fmt(v)}</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-slate-500">No distortion summary.</p>
                  )}
                </div>
              </section>

              {/* Visualizations */}
              <section className="rounded-2xl border border-slate-700 bg-slate-900/80 p-5">
                <h2 className="mb-4 text-lg font-semibold">Visualizations</h2>
                <figure className="mb-4">
                  <img
                    className="w-full rounded-xl border border-slate-700"
                    src={outputUrl(result.id, "spectrogram.png")}
                    alt="Spectrogram triptych: original / enhanced / removed-noise"
                  />
                  <figcaption className="mt-2 text-center text-xs text-slate-500">
                    Spectrograms (shared dB scale): original · enhanced · removed-noise
                  </figcaption>
                </figure>
                <div className="grid gap-4 md:grid-cols-2">
                  <figure>
                    <img
                      className="rounded-xl border border-slate-700"
                      src={outputUrl(result.id, "waveform.png")}
                      alt="Waveform comparison"
                    />
                    <figcaption className="mt-2 text-center text-xs text-slate-500">
                      Waveform
                    </figcaption>
                  </figure>
                  <figure>
                    <img
                      className="rounded-xl border border-slate-700"
                      src={outputUrl(result.id, "policy_probs.png")}
                      alt="Policy probabilities"
                    />
                    <figcaption className="mt-2 text-center text-xs text-slate-500">
                      Trained-policy advisory probabilities
                    </figcaption>
                  </figure>
                </div>
              </section>

              {/* Raw JSON */}
              <section className="rounded-2xl border border-slate-700 bg-slate-900/80 p-5">
                <button
                  onClick={() => setShowJson((s) => !s)}
                  className="text-sm font-semibold text-slate-300 hover:text-white"
                >
                  {showJson ? "▾ Hide raw response" : "▸ Show raw response (full JSON)"}
                </button>
                {showJson && <JsonBlock value={result} height="max-h-[36rem]" />}
              </section>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
