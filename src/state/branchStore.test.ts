import { describe, expect, it } from 'vitest';
import type { Candidate } from '../worker/protocol';
import {
  activeNodes,
  activeText,
  appendToken,
  assertInvariants,
  emptyTree,
  ghostNodes,
  prefixTokenIds,
  pruneGhosts,
  truncateForFork,
  type BranchTree,
} from './branchStore';

const cand = (tokenId: number, raw: string, p = 0.5): Candidate => ({
  tokenId,
  raw,
  text: raw.replace(/ /g, '·'),
  p,
  logit: 0,
});

function build(words: string[]): BranchTree {
  return words.reduce(
    (tree, word, step) =>
      appendToken(tree, { step, chosen: cand(step + 100, word), candidates: [cand(step + 100, word)], temperature: 0.8 }),
    emptyTree,
  );
}

describe('appendToken', () => {
  it('builds a linear active path that satisfies every invariant', () => {
    const tree = build([' the', ' rain', ' fell']);
    expect(activeText(tree)).toBe(' the rain fell');
    expect(() => assertInvariants(tree)).not.toThrow();
  });

  it('drops a token for a step it already holds (a stale late message)', () => {
    const tree = build([' a', ' b']);
    const again = appendToken(tree, { step: 1, chosen: cand(9, ' x'), candidates: [], temperature: 1 });
    expect(again).toBe(tree);
  });

  it('never mutates the previous tree', () => {
    const before = build([' a']);
    const snapshot = JSON.stringify(before);
    appendToken(before, { step: 1, chosen: cand(2, ' b'), candidates: [], temperature: 1 });
    expect(JSON.stringify(before)).toBe(snapshot);
  });
});

describe('forking', () => {
  it('replays exactly the prefix before the fork point', () => {
    const tree = build([' the', ' rain', ' fell', ' hard']);
    expect(prefixTokenIds(tree, 2)).toEqual([100, 101]);
  });

  /**
   * Regression for the bug where the engine numbered fork steps from 0: the store then
   * discarded the forced token and the next ones as stale, and the timeline showed text
   * the model never produced in that order. Fork steps are absolute, so the forced token
   * arrives as step k and must land at position k.
   */
  it('places the forced token at the fork position with an absolute step number', () => {
    const original = build([' the', ' rain', ' fell', ' hard']);
    const k = 2;
    let tree = truncateForFork(original, k);
    tree = appendToken(tree, { step: k, chosen: cand(777, ' stopped'), candidates: [], temperature: 0.8, chosenByUser: true });
    tree = appendToken(tree, { step: k + 1, chosen: cand(778, ' suddenly'), candidates: [], temperature: 0.8 });

    const nodes = activeNodes(tree);
    expect(nodes.map((n) => n.raw)).toEqual([' the', ' rain', ' stopped', ' suddenly']);
    expect(nodes[k].chosenByUser).toBe(true);
    expect(nodes[k].tokenId).toBe(777);
    expect(() => assertInvariants(tree)).not.toThrow();
  });

  it('would silently lose the forced token if steps restarted at 0 — the old bug', () => {
    const tree = truncateForFork(build([' the', ' rain', ' fell']), 2);
    const buggy = appendToken(tree, { step: 0, chosen: cand(777, ' stopped'), candidates: [], temperature: 0.8, chosenByUser: true });
    expect(activeText(buggy)).toBe(' the rain');
  });

  it('keeps the abandoned branch as a ghost instead of deleting it', () => {
    const original = build([' the', ' rain', ' fell']);
    let tree = truncateForFork(original, 1);
    tree = appendToken(tree, { step: 1, chosen: cand(900, ' snow'), candidates: [], temperature: 1, chosenByUser: true });
    const ghosts = ghostNodes(tree).map((n) => n.raw);
    expect(ghosts).toEqual(expect.arrayContaining([' rain', ' fell']));
    expect(activeText(tree)).toBe(' the snow');
  });
});

describe('pruneGhosts', () => {
  it('caps retained ghosts and keeps the active path intact', () => {
    let tree = build([' a', ' b', ' c', ' d', ' e']);
    for (let i = 0; i < 4; i++) {
      tree = truncateForFork(tree, 1);
      tree = appendToken(tree, { step: 1, chosen: cand(500 + i, ` x${i}`), candidates: [], temperature: 1 });
    }
    const pruned = pruneGhosts(tree, 2);
    expect(ghostNodes(pruned)).toHaveLength(2);
    expect(activeText(pruned)).toBe(activeText(tree));
    expect(() => assertInvariants(pruned)).not.toThrow();
  });
});
