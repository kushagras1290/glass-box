/**
 * The constant frame — cinematic-web §3.7. It never moves while the field transforms
 * behind it, so seven chapters of change don't read as disorientation. The numerals are
 * also the accessible table of contents: every chapter is reachable by keyboard.
 */
import { CHAPTERS, INSTRUMENT_INDEX } from './chapters';
import { useStory } from './storyStore';
import type { Device } from '../worker/protocol';

interface Props {
  readonly jumpTo: (index: number) => void;
  readonly device: Device | null;
  readonly busy: boolean;
}

export function Frame({ jumpTo, device, busy }: Props) {
  const activeIndex = useStory((s) => s.activeIndex);

  return (
    <>
      <header className="frame">
        <button type="button" className="brand" onClick={() => jumpTo(0)}>
          Glass Box
        </button>
        <nav className="chapter-nav" aria-label="Chapters">
          {CHAPTERS.map((chapter, i) => (
            <button
              key={chapter.id}
              type="button"
              aria-current={i === activeIndex ? 'step' : undefined}
              aria-label={`Chapter ${chapter.numeral}: ${chapter.title}`}
              onClick={() => jumpTo(i)}
            >
              {chapter.numeral}
            </button>
          ))}
        </nav>
        <div className="frame-status">
          {activeIndex < INSTRUMENT_INDEX ? (
            <button type="button" className="skip" onClick={() => jumpTo(INSTRUMENT_INDEX)}>
              skip to the instrument →
            </button>
          ) : (
            <span className="status">
              <i className={busy ? 'dot live' : device ? 'dot' : 'dot idle'} aria-hidden="true" />
              {device ? `${device}${busy ? ' · thinking' : ''}` : 'not loaded'}
            </span>
          )}
        </div>
      </header>
      <div className="rail" aria-hidden="true">
        <i />
      </div>
    </>
  );
}
