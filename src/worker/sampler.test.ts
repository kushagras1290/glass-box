import { describe, expect, it } from 'vitest';
import { applyTopP, makeRng, sampleFrom, softmaxOver, topKFromLogits } from './sampler';

describe('softmaxOver', () => {
  it('does not overflow to NaN on large logits (max-subtraction)', () => {
    const out = softmaxOver([{ id: 0, logit: 1000 }, { id: 1, logit: 1001 }], 1);
    expect(out.every((c) => Number.isFinite(c.p))).toBe(true);
    expect(out[0].p + out[1].p).toBeCloseTo(1, 10);
    expect(out[1].p).toBeGreaterThan(out[0].p);
  });

  it('treats temperature 0 as argmax with p = 1, without dividing by zero', () => {
    const out = softmaxOver([{ id: 5, logit: 3 }, { id: 9, logit: 1 }], 0);
    expect(out.map((c) => c.p)).toEqual([1, 0]);
  });

  it('sharpens as temperature falls and flattens as it rises', () => {
    const cands = [{ id: 0, logit: 2 }, { id: 1, logit: 1 }, { id: 2, logit: 0 }];
    const cold = softmaxOver(cands, 0.2)[0].p;
    const warm = softmaxOver(cands, 1)[0].p;
    const hot = softmaxOver(cands, 2)[0].p;
    expect(cold).toBeGreaterThan(warm);
    expect(warm).toBeGreaterThan(hot);
  });

  it('preserves input order so callers can zip results with their own metadata', () => {
    const out = softmaxOver([{ id: 7, logit: 0 }, { id: 3, logit: 5 }], 1);
    expect(out.map((c) => c.id)).toEqual([7, 3]);
  });
});

describe('topKFromLogits', () => {
  it('matches a full sort on a realistic vocabulary without sorting it', () => {
    const rng = makeRng(42);
    const logits = new Float32Array(49_152);
    for (let i = 0; i < logits.length; i++) logits[i] = rng() * 30 - 10;

    const got = topKFromLogits(logits, 24);
    const expected = Array.from(logits)
      .map((logit, id) => ({ id, logit }))
      .sort((a, b) => b.logit - a.logit)
      .slice(0, 24);

    expect(got).toHaveLength(24);
    expect(got.map((c) => c.id)).toEqual(expected.map((c) => c.id));
  });

  it('returns descending order and clamps k to the array length', () => {
    const got = topKFromLogits(new Float32Array([0.1, 0.9, 0.5]), 10);
    expect(got.map((c) => c.id)).toEqual([1, 2, 0]);
  });

  it('returns nothing for k <= 0', () => {
    expect(topKFromLogits(new Float32Array([1, 2]), 0)).toEqual([]);
  });
});

describe('applyTopP', () => {
  it('keeps the smallest prefix reaching the mass, then renormalises', () => {
    const cands = [
      { id: 0, logit: 0, p: 0.5 },
      { id: 1, logit: 0, p: 0.3 },
      { id: 2, logit: 0, p: 0.2 },
    ];
    const out = applyTopP(cands, 0.75);
    expect(out.map((c) => c.id)).toEqual([0, 1]);
    expect(out.reduce((s, c) => s + c.p, 0)).toBeCloseTo(1, 10);
  });

  it('is the identity at topP = 1', () => {
    const cands = [{ id: 0, logit: 0, p: 1 }];
    expect(applyTopP(cands, 1)).toBe(cands);
  });
});

describe('sampleFrom + makeRng', () => {
  it('is deterministic for a given seed — share links reproduce exactly', () => {
    const cands = softmaxOver(
      Array.from({ length: 12 }, (_, id) => ({ id, logit: id * 0.3 })),
      1,
    );
    const draw = (seed: number) => {
      const rng = makeRng(seed);
      return Array.from({ length: 20 }, () => sampleFrom(cands, rng).id);
    };
    expect(draw(7)).toEqual(draw(7));
    expect(draw(7)).not.toEqual(draw(8));
  });

  it('draws in proportion to probability', () => {
    const cands = [
      { id: 0, logit: 0, p: 0.8 },
      { id: 1, logit: 0, p: 0.2 },
    ];
    const rng = makeRng(1);
    let zeros = 0;
    const n = 20_000;
    for (let i = 0; i < n; i++) if (sampleFrom(cands, rng).id === 0) zeros++;
    expect(zeros / n).toBeGreaterThan(0.77);
    expect(zeros / n).toBeLessThan(0.83);
  });

  it('throws a RangeError on an empty pool rather than returning undefined', () => {
    expect(() => sampleFrom([], Math.random)).toThrow(RangeError);
  });
});
