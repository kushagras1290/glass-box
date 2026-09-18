/**
 * Narrative chapter layers (I–V, VII). Opacity and motion come from per-chapter CSS vars
 * written by the scroll engine, so none of this re-renders while scrolling. Inactive
 * layers are `inert`: invisible text must not be focusable or announced (Part 5).
 */
import recorded from '../data/recorded.json';
import { recordedDistribution } from './director';
import { CHAPTERS, INSTRUMENT_INDEX, type ChapterId } from './chapters';
import { useStory } from './storyStore';

const CERTAIN = recordedDistribution('certain', 1);
const PARIS_PCT = (CERTAIN[0].p * 100).toFixed(1);
const VOCAB = 49_152;

function Layer({
  id,
  active,
  className = '',
  children,
}: {
  id: ChapterId;
  active: boolean;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <section className={`chapter ${className}`} data-chapter={id} inert={!active} aria-hidden={!active}>
      {children}
    </section>
  );
}

function TemperatureReadout() {
  const temperature = useStory((s) => s.temperature);
  const top = recordedDistribution('open', temperature).slice(0, 6);
  const verdict =
    temperature < 0.08
      ? 'Frozen. It will always pick the likeliest word — deterministic, and dull.'
      : temperature < 0.6
        ? 'Cool. The favourite still dominates; the rest barely register.'
        : temperature < 1.1
          ? 'Warm. Real alternatives are in play. This is where writing lives.'
          : 'Hot. Everything is plausible, and most of it is nonsense.';

  return (
    <div className="temp-readout">
      <p className="temp-value" aria-live="off">
        <span>T</span> {temperature.toFixed(2)}
      </p>
      <p className="temp-verdict">{verdict}</p>
      <ol className="temp-bars" aria-label="Top candidates at this temperature">
        {top.map((cell) => (
          <li key={cell.tokenId}>
            <span className="bar-text">{cell.text}</span>
            <span className="bar-track" aria-hidden="true">
              <i style={{ width: `${Math.max(0.5, cell.p * 100)}%` }} />
            </span>
            <span className="bar-pct">{(cell.p * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

export function Chapters({ jumpTo }: { jumpTo: (index: number) => void }) {
  const activeIndex = useStory((s) => s.activeIndex);
  const is = (id: ChapterId) => CHAPTERS[activeIndex].id === id;

  return (
    <>
      <Layer id="dormant" active={is('dormant')} className="chapter-centre">
        <p className="kicker">SmolLM2 · 360M parameters · runs in this tab</p>
        <h1 className="title">Glass Box</h1>
        <p className="lede">
          A language model is about to run inside your browser. Before it does — here is what it
          is actually doing.
        </p>
      </Layer>

      <Layer id="premise" active={is('premise')} className="chapter-left">
        <p className="kicker">II · recorded</p>
        <h2 className="headline">It never writes a word.</h2>
        <p className="body">
          It writes a <em>distribution</em> over every token it knows, and one is drawn from it.
          Given “{recorded.distributions.certain.prompt}”, this model puts{' '}
          <strong>{PARIS_PCT}%</strong> on “Paris”.
        </p>
        <p className="source">
          recorded from {recorded.model.split('/')[1]} · {recorded.dtype} · T = 1 · probabilities
          renormalised over the top {recorded.topK} of {VOCAB.toLocaleString()} tokens
        </p>
      </Layer>

      <Layer id="shatter" active={is('shatter')} className="chapter-centre">
        <p className="kicker">III · recorded</p>
        <h2 className="headline">It doesn’t read letters.</h2>
        <div className="shatter" aria-label={`"${recorded.shatter.text}" is ${recorded.shatter.tokens.length} tokens`}>
          {recorded.shatter.tokens.map((token) => (
            <span className="shard" key={token.id}>
              <span className="shard-text">{token.text}</span>
              <span className="shard-id">{token.id}</span>
            </span>
          ))}
        </div>
        <p className="body narrow">
          It reads tokens — {recorded.shatter.tokens.length} of them here. To the model each one is
          only a number, one of {VOCAB.toLocaleString()}.
        </p>
      </Layer>

      <Layer id="temperature" active={is('temperature')} className="chapter-left">
        <p className="kicker">IV · recorded · your scroll is the dial</p>
        <h2 className="headline">Temperature is the only dial.</h2>
        <p className="prompt-line">“{recorded.distributions.open.prompt} …”</p>
        <TemperatureReadout />
      </Layer>

      <Layer id="handover" active={is('handover')} className="chapter-left">
        <p className="kicker">V</p>
        <h2 className="headline">Everything above was recorded.</h2>
        <p className="lede">
          From here it’s live — your prompt, running on your GPU. Nothing leaves this tab, because
          there is nowhere for it to go.
        </p>
      </Layer>

      <Layer id="coda" active={is('coda')} className="chapter-centre">
        <h2 className="headline">Every word it wrote was one draw.</h2>
        <p className="lede">You just watched the dice.</p>
        <button type="button" className="ghost" onClick={() => jumpTo(INSTRUMENT_INDEX)}>
          back to the instrument
        </button>
        <p className="credits">
          Built by Kushagra Singh · {recorded.model} · transformers.js on WebGPU · no server ·{' '}
          <a href="https://github.com/kushagras1290/glass-box" target="_blank" rel="noopener noreferrer">
            source
          </a>
        </p>
      </Layer>
    </>
  );
}
