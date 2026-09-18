/**
 * Composition. The document never scrolls (cinematic-web §3.1): a fixed surface owns
 * scrolling, a sticky stage holds the field + chapter layers + constant frame, and the
 * chapter spacers below provide the travel.
 *
 * The stage is sticky-with-transform rather than position:fixed on purpose: a fixed
 * element's containing block is the viewport, which removes it from the surface's scroll
 * chain — wheeling over the instrument would then do nothing.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { FieldCanvas } from './render/FieldCanvas';
import { activeNodes } from './state/branchStore';
import { useInference } from './state/useInference';
import { CHAPTERS, INSTRUMENT_INDEX } from './story/chapters';
import { Chapters } from './story/ChapterLayers';
import type { LiveInputs } from './story/director';
import { Frame } from './story/Frame';
import { useStory } from './story/storyStore';
import { useScrollStory } from './story/useScrollStory';
import { Instrument } from './ui/Instrument';
import './styles.css';

function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const list = window.matchMedia(query);
    const onChange = () => setMatches(list.matches);
    list.addEventListener('change', onChange);
    return () => list.removeEventListener('change', onChange);
  }, [query]);
  return matches;
}

export default function App() {
  const inference = useInference();
  const { state } = inference;
  const surfaceRef = useRef<HTMLDivElement>(null);
  const spacerRefs = useRef<(HTMLElement | null)[]>([]);
  const { jumpTo } = useScrollStory(surfaceRef, spacerRefs);
  const activeIndex = useStory((s) => s.activeIndex);
  const compact = useMediaQuery('(max-width: 900px)');

  const lastNode = useMemo(() => {
    const nodes = activeNodes(state.tree);
    return nodes.length > 0 ? nodes[nodes.length - 1] : null;
  }, [state.tree]);

  const live: LiveInputs = useMemo(
    () => ({
      cloud: state.cloud,
      chosenId: lastNode?.tokenId ?? null,
      phase: state.phase,
      loadProgress: state.loadProgress ? state.loadProgress.progress / 100 : 0,
      compact,
    }),
    [state.cloud, lastNode, state.phase, state.loadProgress, compact],
  );

  const busy = state.phase === 'generating';
  useEffect(() => {
    document.title = busy ? 'Glass Box · thinking…' : 'Glass Box';
  }, [busy]);

  return (
    <div className="surface" ref={surfaceRef}>
      <div className="stage">
        <FieldCanvas live={live} />
        <Chapters jumpTo={jumpTo} />
        <Instrument inference={inference} active={activeIndex === INSTRUMENT_INDEX} />
        <Frame jumpTo={jumpTo} device={state.device} busy={busy} />
      </div>
      <main className="spacers" aria-hidden="true">
        {CHAPTERS.map((chapter, i) => (
          <div
            key={chapter.id}
            className="spacer"
            ref={(el) => {
              spacerRefs.current[i] = el;
            }}
            style={{ height: `${chapter.length * 100}vh` }}
          />
        ))}
      </main>
    </div>
  );
}
