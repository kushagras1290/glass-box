/**
 * The scroll engine — cinematic-web §3.1 (decoupled surface), §3.2 (score),
 * §3.3 (URL from scroll), plus keyboard paging.
 *
 * The document never scrolls. A fixed surface owns scrolling; chapter spacers inside it
 * provide the travel. State is DERIVED from scroll position every frame, never played as
 * a timeline, so reverse scroll, refresh mid-page and deep links reconstruct exactly.
 *
 * Chapter ranges are measured from the spacers rather than computed from vh, because on
 * mobile `100vh` and the visible viewport disagree and the measured value is the truth.
 */
import { useCallback, useEffect, useRef, type RefObject } from 'react';
import { CHAPTERS, FADE_VIEWPORTS, TEMPERATURE_INDEX, temperatureAt } from './chapters';
import { storyStore } from './storyStore';

interface Range {
  readonly start: number;
  readonly end: number;
}

const clamp01 = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v);

function isInteractiveTarget(target: EventTarget | null): boolean {
  return Boolean(
    (target as Element | null)?.closest?.('input, textarea, select, button, a, [contenteditable="true"], [role="dialog"]'),
  );
}

export function useScrollStory(
  surfaceRef: RefObject<HTMLDivElement | null>,
  spacerRefs: RefObject<(HTMLElement | null)[]>,
) {
  const rangesRef = useRef<Range[]>([]);
  const activeRef = useRef(0);

  const measure = useCallback(() => {
    rangesRef.current = CHAPTERS.map((_, i) => {
      const el = spacerRefs.current?.[i];
      if (!el) return { start: 0, end: 0 };
      return { start: el.offsetTop, end: el.offsetTop + el.offsetHeight };
    });
  }, [spacerRefs]);

  const jumpTo = useCallback(
    (index: number, behavior: ScrollBehavior = 'smooth') => {
      const surface = surfaceRef.current;
      if (!surface) return;
      if (rangesRef.current.length === 0) measure();
      const range = rangesRef.current[index];
      if (!range) return;
      const middle = (range.start + range.end) / 2;
      const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      surface.scrollTo({
        top: Math.max(0, middle - surface.clientHeight / 2),
        behavior: reduced ? 'instant' : behavior,
      });
    },
    [measure, surfaceRef],
  );

  useEffect(() => {
    const surface = surfaceRef.current;
    if (!surface) return undefined;
    let ticking = false;

    const update = () => {
      ticking = false;
      if (rangesRef.current.length === 0) measure();
      const viewport = Math.max(1, surface.clientHeight);
      const center = surface.scrollTop + viewport / 2;
      const fade = viewport * FADE_VIEWPORTS;

      const visibility: number[] = [];
      const local: number[] = [];
      let activeIndex = 0;

      rangesRef.current.forEach((range, i) => {
        const span = Math.max(1, range.end - range.start);
        visibility.push(clamp01((center - range.start) / fade) * clamp01((range.end - center) / fade));
        local.push(clamp01((center - range.start) / span));
        if (center >= range.start) activeIndex = i;
      });

      const temperature = temperatureAt(local[TEMPERATURE_INDEX] ?? 0);

      // Per-chapter CSS vars: DOM layers read these, so chapter text needs no React render.
      CHAPTERS.forEach((chapter, i) => {
        surface.style.setProperty(`--v-${chapter.id}`, visibility[i].toFixed(4));
        surface.style.setProperty(`--p-${chapter.id}`, local[i].toFixed(4));
      });
      const total = Math.max(1, surface.scrollHeight - viewport);
      surface.style.setProperty('--progress', clamp01(surface.scrollTop / total).toFixed(4));

      storyStore.set({ activeIndex, visibility, local, temperature });

      if (activeIndex !== activeRef.current) {
        activeRef.current = activeIndex;
        // replaceState, never pushState: scrolling must not poison the back button.
        const hash = CHAPTERS[activeIndex].hash;
        const next = `${window.location.pathname}${window.location.search}${hash ? `#${hash}` : ''}`;
        if (next !== `${window.location.pathname}${window.location.search}${window.location.hash}`) {
          window.history.replaceState(null, '', next);
        }
      }
    };

    const request = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(update);
    };

    const remeasure = () => {
      measure();
      request();
    };

    const onKey = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (isInteractiveTarget(event.target)) return;
      if (document.querySelector('[aria-modal="true"]')) return;
      const page = surface.clientHeight * 0.85;
      const delta: Record<string, number> = { PageDown: page, PageUp: -page, ArrowDown: 120, ArrowUp: -120, ' ': event.shiftKey ? -page : page };
      if (event.key in delta) {
        event.preventDefault();
        surface.scrollBy({ top: delta[event.key], behavior: 'smooth' });
      } else if (event.key === 'Home' || event.key === 'End') {
        event.preventDefault();
        surface.scrollTo({ top: event.key === 'Home' ? 0 : surface.scrollHeight, behavior: 'smooth' });
      }
    };

    measure();

    // Deep link: map an incoming hash back to its chapter (§3.3). On first load this is an
    // instant jump; afterwards it handles same-document hash navigation (an in-page link,
    // an edited URL, Back to a hash) — which never reloads the page, so without a
    // hashchange listener those would silently do nothing. replaceState does not fire
    // hashchange, so the engine's own URL writes cannot loop back through here.
    const chapterForHash = (): number => {
      const incoming = window.location.hash.slice(1);
      return incoming ? CHAPTERS.findIndex((c) => c.hash === incoming) : -1;
    };
    const linked = chapterForHash();
    if (linked > 0) {
      requestAnimationFrame(() => {
        jumpTo(linked, 'instant');
        request();
      });
    }
    const onHashChange = () => {
      const index = chapterForHash();
      jumpTo(index >= 0 ? index : 0);
    };

    update();
    surface.addEventListener('scroll', request, { passive: true });
    window.addEventListener('resize', remeasure);
    window.addEventListener('load', remeasure);
    window.addEventListener('keydown', onKey);
    window.addEventListener('hashchange', onHashChange);
    document.fonts?.ready.then(remeasure).catch(() => undefined);
    const observer = new ResizeObserver(remeasure);
    spacerRefs.current?.forEach((el) => el && observer.observe(el));

    return () => {
      surface.removeEventListener('scroll', request);
      window.removeEventListener('resize', remeasure);
      window.removeEventListener('load', remeasure);
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('hashchange', onHashChange);
      observer.disconnect();
    };
  }, [jumpTo, measure, spacerRefs, surfaceRef]);

  return { jumpTo };
}
