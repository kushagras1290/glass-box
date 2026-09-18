/**
 * Mounts the field and feeds it. Two inputs drive the target: scroll-derived story state
 * (read imperatively from the store, never through React renders) and live inference
 * state (props). Either changing recomputes the target via the pure director.
 */
import { useEffect, useRef } from 'react';
import { config } from '../config';
import { direct, type LiveInputs } from '../story/director';
import { storyStore } from '../story/storyStore';
import { FieldScene } from './fieldScene';

interface Props {
  readonly live: LiveInputs;
}

export function FieldCanvas({ live }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sceneRef = useRef<FieldScene | null>(null);
  const liveRef = useRef<LiveInputs>(live);
  liveRef.current = live;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const scene = new FieldScene(canvas, { dprCap: config.dprCap, reducedMotion });
    sceneRef.current = scene;

    const push = () => scene.setTarget(direct(storyStore.get(), liveRef.current));
    push();
    const unsubscribe = storyStore.subscribe(push);
    const observer = new ResizeObserver(() => scene.resize());
    observer.observe(canvas);

    return () => {
      unsubscribe();
      observer.disconnect();
      scene.destroy();
      sceneRef.current = null;
    };
  }, []);

  useEffect(() => {
    sceneRef.current?.setTarget(direct(storyStore.get(), live));
  }, [live]);

  return <canvas className="field" ref={canvasRef} aria-hidden="true" />;
}
