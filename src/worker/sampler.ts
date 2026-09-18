/**
 * Pure sampling functions — spec §6.7. No DOM, no model: unit-testable in Node.
 *
 * Two non-negotiables encoded here:
 *  - top-k must not sort the full vocabulary (49k–152k floats per step)
 *  - softmax must subtract the max logit or fp32 overflows to NaN
 */

export interface RawCandidate {
  readonly id: number;
  readonly logit: number;
}

export interface ScoredCandidate extends RawCandidate {
  readonly p: number;
}

/**
 * Bounded-scan top-k. O(V·log k) with a tiny k, versus O(V·log V) for a full sort.
 * Returns descending by logit.
 */
export function topKFromLogits(logits: Float32Array, k: number): RawCandidate[] {
  if (k <= 0) return [];
  const size = Math.min(k, logits.length);
  const heap: RawCandidate[] = [];

  for (let i = 0; i < logits.length; i++) {
    const logit = logits[i];
    if (heap.length < size) {
      heap.push({ id: i, logit });
      if (heap.length === size) heap.sort((a, b) => a.logit - b.logit);
    } else if (logit > heap[0].logit) {
      heap[0] = { id: i, logit };
      // Re-establish the min at index 0 with a single linear pass — cheaper than a
      // full sort and k is small enough that this dominates nothing.
      let minIdx = 0;
      for (let j = 1; j < heap.length; j++) if (heap[j].logit < heap[minIdx].logit) minIdx = j;
      if (minIdx !== 0) {
        const tmp = heap[0];
        heap[0] = heap[minIdx];
        heap[minIdx] = tmp;
      }
    }
  }
  return heap.sort((a, b) => b.logit - a.logit);
}

/** Max-subtracted softmax. temperature === 0 ⇒ argmax with p = 1 (no division by zero). */
export function softmaxOver(cands: RawCandidate[], temperature: number): ScoredCandidate[] {
  if (cands.length === 0) return [];
  if (temperature <= 0) {
    return cands.map((c, i) => ({ ...c, p: i === 0 ? 1 : 0 }));
  }
  let max = -Infinity;
  for (const c of cands) if (c.logit > max) max = c.logit;

  const exps = cands.map((c) => Math.exp((c.logit - max) / temperature));
  let sum = 0;
  for (const e of exps) sum += e;
  if (sum === 0 || !Number.isFinite(sum)) {
    return cands.map((c, i) => ({ ...c, p: i === 0 ? 1 : 0 }));
  }
  return cands.map((c, i) => ({ ...c, p: exps[i] / sum }));
}

/** Nucleus filter. Keeps the smallest prefix whose mass ≥ topP, then renormalises. */
export function applyTopP(cands: ScoredCandidate[], topP: number): ScoredCandidate[] {
  if (topP >= 1 || cands.length === 0) return cands;
  const kept: ScoredCandidate[] = [];
  let mass = 0;
  for (const c of cands) {
    kept.push(c);
    mass += c.p;
    if (mass >= topP) break;
  }
  if (mass <= 0) return cands;
  return kept.map((c) => ({ ...c, p: c.p / mass }));
}

/** Inverse-CDF sample. `rng` is injected so tests and ?seed= links are deterministic. */
export function sampleFrom(cands: ScoredCandidate[], rng: () => number): ScoredCandidate {
  if (cands.length === 0) throw new RangeError('sampleFrom: empty candidate list');
  const roll = rng();
  let acc = 0;
  for (const c of cands) {
    acc += c.p;
    if (roll <= acc) return c;
  }
  return cands[cands.length - 1];
}

/** mulberry32 — small, fast, seedable. Same seed ⇒ same run (spec §9.3 share links). */
export function makeRng(seed?: number): () => number {
  if (seed === undefined) return Math.random;
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
