/**
 * Director — turns (story state + live inference state) into what the field shows.
 * Pure: no DOM, no canvas. The scene eases toward whatever this returns, so chapter
 * transitions are continuous even though the target switches discretely.
 */
import recorded from '../data/recorded.json';
import type { Phase } from '../state/useInference';
import { softmaxOver } from '../worker/sampler';
import type { Candidate } from '../worker/protocol';
import { CHAPTERS, TEMP_MAX, temperatureAccent, type RGB } from './chapters';
import type { StorySnapshot } from './storyStore';

export interface FieldCell {
  readonly tokenId: number;
  readonly text: string;
  readonly p: number;
}

export interface FieldTarget {
  readonly cells: readonly FieldCell[];
  readonly chosenId: number | null;
  /** 0..1 intensity of the single central point (dormant / loading / home) */
  readonly point: number;
  readonly focusX: number;
  readonly focusY: number;
  /** radius multiplier; > 1 scatters the field outward (the handover dissolve) */
  readonly spread: number;
  /** global opacity multiplier */
  readonly dim: number;
  readonly accent: RGB;
  readonly showPercent: boolean;
}

export interface LiveInputs {
  readonly cloud: readonly Candidate[];
  readonly chosenId: number | null;
  readonly phase: Phase;
  readonly loadProgress: number;
  readonly compact: boolean;
}

type RecordedKey = keyof typeof recorded.distributions;

/** Real recorded logits → probabilities at temperature T, via the SAME sampler the engine uses. */
export function recordedDistribution(key: RecordedKey, temperature: number): FieldCell[] {
  const source = recorded.distributions[key].candidates;
  const scored = softmaxOver(
    source.map((c) => ({ id: c.id, logit: c.logit })),
    temperature,
  );
  return scored.map((s, i) => ({ tokenId: s.id, text: source[i].text, p: s.p }));
}

const CERTAIN_AT_ONE = recordedDistribution('certain', 1);
const CERTAIN_TOP = recorded.distributions.certain.candidates[0].id;
const OPEN_TOP = recorded.distributions.open.candidates[0].id;
const WARM: RGB = [233, 174, 123];
/** Where the lone point rests in the centred chapters: below the copy, never on it. */
const HOME_Y = 0.78;

/**
 * On a phone the copy and the field share one narrow column, so labelled cells collide
 * with the text. There the DOM readout carries the numbers legibly and the field steps
 * back to atmosphere (skill Part 5: simpler rig on mobile, never a worse read).
 */
const TEXT_CHAPTERS = new Set(['premise', 'temperature', 'handover']);
const COMPACT_TEXT_DIM = 0.3;

export function direct(story: StorySnapshot, live: LiveInputs): FieldTarget {
  const target = directRaw(story, live);
  if (!live.compact || !TEXT_CHAPTERS.has(CHAPTERS[story.activeIndex].id)) return target;
  return { ...target, dim: target.dim * COMPACT_TEXT_DIM, showPercent: false };
}

function directRaw(story: StorySnapshot, live: LiveInputs): FieldTarget {
  const chapter = CHAPTERS[story.activeIndex];
  const local = story.local[story.activeIndex] ?? 0;
  const base = {
    focusX: 0.5,
    focusY: 0.5,
    spread: 1,
    dim: 1,
    accent: chapter.accent,
    showPercent: false,
  };

  switch (chapter.id) {
    // The point sits BELOW the centred copy — it is the scroll cue, not a mark over the text.
    // Coda uses the same position so the ending rhymes with the opening (§3.9).
    case 'dormant':
      return { ...base, focusY: HOME_Y, cells: [], chosenId: null, point: 0.6 };

    case 'premise':
      return {
        ...base,
        focusX: live.compact ? 0.5 : 0.62,
        cells: CERTAIN_AT_ONE,
        chosenId: CERTAIN_TOP,
        point: 0,
        showPercent: true,
      };

    case 'shatter':
      // The tokens are the subject here; the field steps back entirely.
      return { ...base, focusY: HOME_Y, cells: [], chosenId: null, point: 0, dim: 0.7 };

    case 'temperature':
      return {
        ...base,
        focusX: live.compact ? 0.5 : 0.64,
        cells: recordedDistribution('open', story.temperature),
        chosenId: OPEN_TOP,
        point: 0,
        accent: temperatureAccent(story.temperature),
        showPercent: true,
      };

    case 'handover':
      // The recorded world dissolves outward while a warm live point is born.
      return {
        ...base,
        focusX: live.compact ? 0.5 : 0.64,
        cells: recordedDistribution('open', TEMP_MAX),
        chosenId: null,
        spread: 1 + local * 1.5,
        dim: Math.max(0, 1 - local * 1.15),
        point: local,
        accent: WARM,
      };

    case 'instrument': {
      const focus = live.compact ? { focusX: 0.5, focusY: 0.46 } : { focusX: 0.68, focusY: 0.44 };
      if (live.phase === 'loading') {
        return { ...base, ...focus, cells: [], chosenId: null, point: 0.2 + live.loadProgress * 0.8, accent: WARM };
      }
      if (live.cloud.length > 0) {
        return {
          ...base,
          ...focus,
          spread: live.compact ? 0.72 : 0.8,
          cells: live.cloud.map((c) => ({ tokenId: c.tokenId, text: c.text, p: c.p })),
          chosenId: live.chosenId,
          point: 0,
          accent: WARM,
          showPercent: true,
        };
      }
      return { ...base, ...focus, cells: [], chosenId: null, point: 0.4, accent: WARM };
    }

    case 'coda':
      // Come home: the single point again, in the same place, quieter than where we began.
      return { ...base, focusY: HOME_Y, cells: [], chosenId: null, point: 0.45 - local * 0.2, dim: 0.9 };
  }
}
