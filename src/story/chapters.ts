/**
 * The score — cinematic-web §3.2. Chapter lengths are deliberately unequal: a long
 * opening dwell, a quick beat, the extended centrepiece (IV), a short exhale, a long
 * close. Uniform lengths read as a conveyor belt.
 *
 *   I 1.4 · II 1.05 · III 2.1 · IV 4.2 · V 1.26 · VI 2.1 · VII 1.4   (viewport heights)
 *
 * I–IV are RECORDED (real data from this model, see src/data/recorded.json).
 * V is the world change (§3.8): recorded → live.
 * VII rhymes with I and returns the URL to root (§3.9).
 */

export type ChapterId =
  | 'dormant'
  | 'premise'
  | 'shatter'
  | 'temperature'
  | 'handover'
  | 'instrument'
  | 'coda';

export type RGB = readonly [number, number, number];

export interface Chapter {
  readonly id: ChapterId;
  readonly numeral: string;
  readonly title: string;
  /** scroll length in viewport heights */
  readonly length: number;
  /** URL hash while this chapter is active; null means the root URL (§3.3, §3.9) */
  readonly hash: string | null;
  /** chapter accent; 'temperature' overrides this per frame from the live dial (§3.10) */
  readonly accent: RGB;
}

export const CHAPTERS: readonly Chapter[] = [
  { id: 'dormant', numeral: 'I', title: 'Dormant', length: 1.4, hash: null, accent: [168, 196, 204] },
  { id: 'premise', numeral: 'II', title: 'A distribution', length: 1.05, hash: 'premise', accent: [125, 219, 210] },
  { id: 'shatter', numeral: 'III', title: 'Tokens', length: 2.1, hash: 'tokens', accent: [125, 200, 235] },
  { id: 'temperature', numeral: 'IV', title: 'Temperature', length: 4.2, hash: 'temperature', accent: [120, 210, 235] },
  { id: 'handover', numeral: 'V', title: 'Handover', length: 1.26, hash: 'live', accent: [233, 174, 123] },
  { id: 'instrument', numeral: 'VI', title: 'Instrument', length: 2.1, hash: 'instrument', accent: [233, 174, 123] },
  { id: 'coda', numeral: 'VII', title: 'Coda', length: 1.4, hash: null, accent: [168, 196, 204] },
];

export const INSTRUMENT_INDEX = CHAPTERS.findIndex((c) => c.id === 'instrument');
export const TEMPERATURE_INDEX = CHAPTERS.findIndex((c) => c.id === 'temperature');

/** Highest temperature the scroll dial reaches in chapter IV. */
export const TEMP_MAX = 1.6;

/** Fraction of a viewport over which chapter text fades at each edge. */
export const FADE_VIEWPORTS = 0.35;

const clamp01 = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v);

/**
 * Chapter IV maps scroll to temperature. It holds at T=0 for the first 10% so the reader
 * sees the single collapsed spike, then sweeps to TEMP_MAX, then holds at the top.
 */
export function temperatureAt(localProgress: number): number {
  return TEMP_MAX * clamp01((localProgress - 0.1) / 0.78);
}

/** Temperature → colour temperature: cold teal at 0, violet mid, hot amber at the top. */
const TEMP_STOPS: ReadonlyArray<{ t: number; rgb: RGB }> = [
  { t: 0, rgb: [120, 210, 235] },
  { t: 0.8, rgb: [168, 140, 255] },
  { t: TEMP_MAX, rgb: [240, 170, 110] },
];

export function temperatureAccent(t: number): RGB {
  if (t <= TEMP_STOPS[0].t) return TEMP_STOPS[0].rgb;
  for (let i = 1; i < TEMP_STOPS.length; i++) {
    const a = TEMP_STOPS[i - 1];
    const b = TEMP_STOPS[i];
    if (t <= b.t) {
      const k = (t - a.t) / (b.t - a.t);
      return [
        Math.round(a.rgb[0] + (b.rgb[0] - a.rgb[0]) * k),
        Math.round(a.rgb[1] + (b.rgb[1] - a.rgb[1]) * k),
        Math.round(a.rgb[2] + (b.rgb[2] - a.rgb[2]) * k),
      ];
    }
  }
  return TEMP_STOPS[TEMP_STOPS.length - 1].rgb;
}
