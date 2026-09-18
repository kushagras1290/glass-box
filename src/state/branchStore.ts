/**
 * Branch tree — spec §5. Single source of truth; everything rendered derives from it.
 * Forking never mutates or deletes: it appends a child and re-points the active path,
 * so the tree becomes the visitor's history of interrogation.
 */
import type { Candidate } from '../worker/protocol';

export interface BranchNode {
  readonly id: string;
  readonly parentId: string | null;
  readonly childIds: string[];
  readonly step: number;
  readonly tokenId: number;
  readonly text: string;
  readonly raw: string;
  readonly p: number;
  readonly candidates: Candidate[];
  readonly chosenByUser: boolean;
  readonly temperature: number;
}

export interface BranchTree {
  readonly nodes: Readonly<Record<string, BranchNode>>;
  readonly rootIds: readonly string[];
  readonly activePath: readonly string[];
  readonly promptTokenIds: readonly number[];
  readonly promptTexts: readonly string[];
}

export const emptyTree: BranchTree = {
  nodes: {},
  rootIds: [],
  activePath: [],
  promptTokenIds: [],
  promptTexts: [],
};

let counter = 0;
function nextId(): string {
  counter += 1;
  return `n${counter.toString(36)}`;
}

export function setPrompt(tree: BranchTree, tokenIds: number[], texts: string[]): BranchTree {
  return { ...tree, promptTokenIds: tokenIds, promptTexts: texts };
}

/** Append a generated token to the end of the active path. */
export function appendToken(
  tree: BranchTree,
  input: {
    step: number;
    chosen: Candidate;
    candidates: Candidate[];
    temperature: number;
    chosenByUser?: boolean;
  },
): BranchTree {
  // A late token for a step we already hold means a stale run; ignore it.
  if (input.step < tree.activePath.length) return tree;

  const parentId = tree.activePath.length > 0 ? tree.activePath[tree.activePath.length - 1] : null;
  const id = nextId();
  const node: BranchNode = {
    id,
    parentId,
    childIds: [],
    step: input.step,
    tokenId: input.chosen.tokenId,
    text: input.chosen.text,
    raw: input.chosen.raw,
    p: input.chosen.p,
    candidates: input.candidates,
    chosenByUser: input.chosenByUser ?? false,
    temperature: input.temperature,
  };

  const nodes: Record<string, BranchNode> = { ...tree.nodes, [id]: node };
  if (parentId) {
    const parent = nodes[parentId];
    nodes[parentId] = { ...parent, childIds: [...parent.childIds, id] };
  }

  return {
    ...tree,
    nodes,
    rootIds: parentId ? tree.rootIds : [...tree.rootIds, id],
    activePath: [...tree.activePath, id],
  };
}

/**
 * Truncate the active path so the next appended token lands at `step`.
 * Old nodes are retained — they render as ghost branches.
 */
export function truncateForFork(tree: BranchTree, step: number): BranchTree {
  return { ...tree, activePath: tree.activePath.slice(0, step) };
}

/** Token ids along the active path up to (not including) `step` — the fork replay prefix. */
export function prefixTokenIds(tree: BranchTree, step: number): number[] {
  return tree.activePath.slice(0, step).map((id) => tree.nodes[id].tokenId);
}

export function activeNodes(tree: BranchTree): BranchNode[] {
  return tree.activePath.map((id) => tree.nodes[id]).filter(Boolean);
}

export function activeText(tree: BranchTree): string {
  return activeNodes(tree)
    .map((n) => n.raw)
    .join('');
}

/** Ghost branches: nodes that exist but are not on the active path. */
export function ghostNodes(tree: BranchTree): BranchNode[] {
  const active = new Set(tree.activePath);
  return Object.values(tree.nodes).filter((n) => !active.has(n.id));
}

/** Cap retained branches so a long forking session cannot grow unbounded (spec §11.2). */
export function pruneGhosts(tree: BranchTree, maxRetained: number): BranchTree {
  const ghosts = ghostNodes(tree);
  if (ghosts.length <= maxRetained) return tree;

  const doomed = new Set(
    ghosts
      .sort((a, b) => a.step - b.step)
      .slice(0, ghosts.length - maxRetained)
      .map((n) => n.id),
  );
  const nodes: Record<string, BranchNode> = {};
  for (const [id, node] of Object.entries(tree.nodes)) {
    if (doomed.has(id)) continue;
    nodes[id] = { ...node, childIds: node.childIds.filter((c) => !doomed.has(c)) };
  }
  return {
    ...tree,
    nodes,
    rootIds: tree.rootIds.filter((id) => !doomed.has(id)),
  };
}

/** Dev-only invariant check (spec §5). */
export function assertInvariants(tree: BranchTree): void {
  const { activePath, nodes, rootIds } = tree;
  if (activePath.length === 0) return;
  if (!rootIds.includes(activePath[0])) {
    throw new RangeError('invariant 1: activePath[0] is not a root');
  }
  for (let i = 1; i < activePath.length; i++) {
    if (nodes[activePath[i]].parentId !== activePath[i - 1]) {
      throw new RangeError(`invariant 2 broken at index ${i}`);
    }
  }
  for (let i = 0; i < activePath.length; i++) {
    if (nodes[activePath[i]].step !== i) {
      throw new RangeError(`invariant 3 broken at index ${i}`);
    }
  }
}
