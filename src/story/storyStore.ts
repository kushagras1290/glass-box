/**
 * Scroll-derived story state, held OUTSIDE React.
 *
 * Scroll fires every frame; routing it through React state would re-render the whole
 * app 60 times a second. Instead the scroll handler writes here, the canvas reads it
 * imperatively, and React components subscribe only to the primitive they need.
 *
 * Selectors passed to useStory MUST return primitives (or otherwise stable values):
 * useSyncExternalStore compares with Object.is, so a fresh object per call loops forever.
 */
import { useSyncExternalStore } from 'react';
import { CHAPTERS } from './chapters';

export interface StorySnapshot {
  readonly activeIndex: number;
  readonly visibility: readonly number[];
  readonly local: readonly number[];
  readonly temperature: number;
}

type Listener = () => void;

let snapshot: StorySnapshot = {
  activeIndex: 0,
  visibility: CHAPTERS.map((_, i) => (i === 0 ? 1 : 0)),
  local: CHAPTERS.map(() => 0),
  temperature: 0,
};
const listeners = new Set<Listener>();

export const storyStore = {
  get(): StorySnapshot {
    return snapshot;
  },
  set(next: StorySnapshot): void {
    snapshot = next;
    listeners.forEach((listener) => listener());
  },
  subscribe(listener: Listener): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
};

export function useStory<T extends string | number | boolean | null>(selector: (s: StorySnapshot) => T): T {
  return useSyncExternalStore(storyStore.subscribe, () => selector(storyStore.get()));
}
