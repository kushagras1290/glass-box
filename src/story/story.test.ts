import { describe, expect, it } from 'vitest';
import recorded from '../data/recorded.json';
import { CHAPTERS, TEMP_MAX, TEMPERATURE_INDEX, temperatureAccent, temperatureAt } from './chapters';
import { direct, recordedDistribution, type LiveInputs } from './director';
import type { StorySnapshot } from './storyStore';

const live: LiveInputs = { cloud: [], chosenId: null, phase: 'cold', loadProgress: 0, compact: false };

function snapshotAt(index: number, local = 0.5, temperature = 0): StorySnapshot {
  return {
    activeIndex: index,
    visibility: CHAPTERS.map((_, i) => (i === index ? 1 : 0)),
    local: CHAPTERS.map((_, i) => (i === index ? local : 0)),
    temperature,
  };
}

describe('the score', () => {
  it('is deliberately unequal — the centrepiece is the longest chapter', () => {
    const lengths = CHAPTERS.map((c) => c.length);
    expect(new Set(lengths).size).toBeGreaterThan(3);
    expect(Math.max(...lengths)).toBe(CHAPTERS[TEMPERATURE_INDEX].length);
  });

  it('opens and closes on the root URL so the journey comes home', () => {
    expect(CHAPTERS[0].hash).toBeNull();
    expect(CHAPTERS[CHAPTERS.length - 1].hash).toBeNull();
  });

  it('has unique hashes for every deep-linkable chapter', () => {
    const hashes = CHAPTERS.map((c) => c.hash).filter(Boolean);
    expect(new Set(hashes).size).toBe(hashes.length);
  });
});

describe('temperature dial', () => {
  it('holds at 0, sweeps, and clamps at the maximum', () => {
    expect(temperatureAt(0)).toBe(0);
    expect(temperatureAt(0.1)).toBe(0);
    expect(temperatureAt(1)).toBe(TEMP_MAX);
    expect(temperatureAt(0.5)).toBeGreaterThan(0);
    expect(temperatureAt(0.5)).toBeLessThan(TEMP_MAX);
  });

  it('maps cold to cool colour and hot to warm colour', () => {
    const [coldR, , coldB] = temperatureAccent(0);
    const [hotR, , hotB] = temperatureAccent(TEMP_MAX);
    expect(coldB).toBeGreaterThan(coldR);
    expect(hotR).toBeGreaterThan(hotB);
  });
});

describe('recorded data', () => {
  it('collapses to a single certain token at T = 0', () => {
    const cells = recordedDistribution('open', 0);
    expect(cells[0].p).toBe(1);
    expect(cells.slice(1).every((c) => c.p === 0)).toBe(true);
  });

  it('puts Paris first for the certain prompt — the real recorded result', () => {
    const cells = recordedDistribution('certain', 1);
    expect(cells[0].text).toBe('·Paris');
    expect(cells[0].p).toBeGreaterThan(0.8);
  });

  it('is genuinely less certain for the open prompt than the certain one', () => {
    expect(recordedDistribution('open', 1)[0].p).toBeLessThan(recordedDistribution('certain', 1)[0].p);
  });

  it('was recorded from the configured model', () => {
    expect(recorded.model).toBe('HuggingFaceTB/SmolLM2-360M-Instruct');
  });
});

describe('director', () => {
  it('shows the lone point and no cells when dormant', () => {
    const t = direct(snapshotAt(0), live);
    expect(t.cells).toHaveLength(0);
    expect(t.point).toBeGreaterThan(0);
  });

  it('drives the field from the scroll temperature in chapter IV', () => {
    const frozen = direct(snapshotAt(TEMPERATURE_INDEX, 0.05, 0), live);
    const hot = direct(snapshotAt(TEMPERATURE_INDEX, 0.95, TEMP_MAX), live);
    expect(frozen.cells[0].p).toBe(1);
    expect(hot.cells[0].p).toBeLessThan(0.5);
  });

  it('steps the field back behind the copy on a phone', () => {
    const desktop = direct(snapshotAt(TEMPERATURE_INDEX, 0.5, 1), live);
    const phone = direct(snapshotAt(TEMPERATURE_INDEX, 0.5, 1), { ...live, compact: true });
    expect(phone.dim).toBeLessThan(desktop.dim);
    expect(phone.showPercent).toBe(false);
  });

  it('renders the live cloud in the instrument once tokens arrive', () => {
    const instrument = CHAPTERS.findIndex((c) => c.id === 'instrument');
    const cloud = [{ tokenId: 1, text: '·x', raw: ' x', p: 0.9, logit: 1 }];
    const t = direct(snapshotAt(instrument), { ...live, phase: 'generating', cloud, chosenId: 1 });
    expect(t.cells).toHaveLength(1);
    expect(t.chosenId).toBe(1);
  });
});
