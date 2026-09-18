/**
 * Chapter VI — the live instrument. The field behind it IS the probability cloud; these
 * frosted panels sit over it, which is the product's own metaphor made literal.
 * Never auto-downloads: weights are fetched only after an explicit click (skill Part 5).
 */
import { useMemo, useState } from 'react';
import { config } from '../config';
import { Timeline } from '../render/Timeline';
import { activeNodes, activeText } from '../state/branchStore';
import type { useInference } from '../state/useInference';
import type { PromptMode } from '../worker/protocol';

/**
 * Chips chosen to demonstrate distribution SHAPE, not model capability (spec §9.2).
 * Most are completions: a chat template turns "The capital of France is" into a
 * question, and you get an assistant reply instead of the one-spike distribution.
 */
const PROMPTS: ReadonlyArray<{ label: string; text: string; mode: PromptMode; note: string }> = [
  { label: 'Certain', text: 'The capital of France is', mode: 'complete', note: 'one spike — near total confidence' },
  { label: 'Open', text: 'The rain began just after midnight, and', mode: 'complete', note: 'a wide, flat field' },
  { label: 'Confidently wrong', text: 'The first person to walk on Mars was named', mode: 'complete', note: 'sharp, confident, false' },
  { label: 'Chat', text: 'Name three colours.', mode: 'chat', note: 'instruct template — it answers instead of continuing' },
];

const HAS_WEBGPU = typeof navigator !== 'undefined' && 'gpu' in navigator;
const LOW_MEMORY =
  typeof navigator !== 'undefined' && ((navigator as unknown as { deviceMemory?: number }).deviceMemory ?? 8) < 4;

type Inference = ReturnType<typeof useInference>;

export function Instrument({ inference, active }: { inference: Inference; active: boolean }) {
  const { state, temperature, load, generate, fork, cancel, setTemperature, reset } = inference;
  const [draft, setDraft] = useState(PROMPTS[0].text);
  const [mode, setMode] = useState<PromptMode>(PROMPTS[0].mode);

  const nodes = useMemo(() => activeNodes(state.tree), [state.tree]);
  const text = useMemo(() => activeText(state.tree), [state.tree]);
  const busy = state.phase === 'generating';
  const loaded = state.phase === 'ready' || state.phase === 'generating' || state.phase === 'done';
  const pct = state.loadProgress ? Math.round(state.loadProgress.progress) : 0;

  return (
    <section className="chapter instrument" data-chapter="instrument" inert={!active} aria-hidden={!active} aria-label="Live instrument">
      <div className="glass composer">
        <p className="kicker">VI · live</p>

        {state.phase === 'cold' && (
          <div className="stack">
            <h2 className="panel-title">Load the model</h2>
            <p className="body small">
              Fetches <strong>{config.modelId.split('/')[1]}</strong> ({config.dtype}) and runs it on
              your {HAS_WEBGPU ? 'GPU' : 'CPU'}. Cached by the browser afterwards — you pay this once.
            </p>
            {!HAS_WEBGPU && (
              <p className="notice">
                No WebGPU here. It will run on the CPU through WebAssembly — correct, but slow.
              </p>
            )}
            {LOW_MEMORY && (
              <p className="notice">This device reports under 4 GB of memory. It may struggle.</p>
            )}
            <button type="button" className="primary" onClick={load}>
              Load the model
            </button>
          </div>
        )}

        {state.phase === 'loading' && (
          <div className="stack">
            <h2 className="panel-title">Downloading weights</h2>
            <div className="progress" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
              <i style={{ width: `${pct}%` }} />
            </div>
            <p className="source">{state.loadProgress?.file ?? 'starting'} · {pct}%</p>
          </div>
        )}

        {state.error && (
          <div className="stack" role="alert">
            <h2 className="panel-title danger">{state.error.code}</h2>
            <p className="body small">{state.error.message}</p>
            {state.error.recoverable && (
              <button type="button" className="primary" onClick={load}>
                Try again
              </button>
            )}
          </div>
        )}

        {loaded && (
          <div className="stack">
            <label className="field-label" htmlFor="prompt">
              Prompt
            </label>
            <textarea id="prompt" value={draft} rows={2} onChange={(e) => setDraft(e.target.value)} disabled={busy} />
            <div className="chips">
              {PROMPTS.map((p) => (
                <button
                  key={p.label}
                  type="button"
                  className="chip"
                  title={p.note}
                  disabled={busy}
                  onClick={() => {
                    setDraft(p.text);
                    setMode(p.mode);
                  }}
                >
                  {p.label}
                </button>
              ))}
            </div>
            <fieldset className="modes" disabled={busy}>
              <legend>mode</legend>
              <label>
                <input type="radio" name="mode" checked={mode === 'complete'} onChange={() => setMode('complete')} />
                <span>continue my text</span>
              </label>
              <label>
                <input type="radio" name="mode" checked={mode === 'chat'} onChange={() => setMode('chat')} />
                <span>answer it (chat)</span>
              </label>
            </fieldset>
            <div className="actions">
              <button
                type="button"
                className="primary"
                onClick={() => generate(draft, mode)}
                disabled={busy || draft.trim().length === 0}
              >
                Generate
              </button>
              {busy && (
                <button type="button" onClick={cancel}>
                  Stop
                </button>
              )}
              {!busy && nodes.length > 0 && (
                <button type="button" onClick={reset}>
                  Clear
                </button>
              )}
            </div>
            <label className="temp">
              <span>
                temperature <strong>{temperature.toFixed(2)}</strong>
              </span>
              <input
                type="range"
                min={0}
                max={2}
                step={0.05}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                aria-label="Temperature"
              />
            </label>
            <p className="source">
              {state.device}
              {state.stats ? ` · ${state.stats.tokensPerSecond} tok/s · ${state.stats.reason}` : ''}
            </p>
          </div>
        )}
      </div>

      {loaded && (
        <>
          <div className="glass readout">
            <p className="field-label">Output</p>
            <p className="output" aria-live="polite">
              {text || <span className="muted">generate something, then click any token to fork it</span>}
            </p>
          </div>
          <div className="glass timeline-dock">
            <Timeline nodes={nodes} onFork={fork} disabled={busy} />
          </div>
        </>
      )}
    </section>
  );
}
