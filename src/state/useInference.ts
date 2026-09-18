/** React binding over the worker — owns the tree and the run lifecycle. */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { config } from '../config';
import { log } from '../log';
import type {
  Candidate,
  Device,
  GenParams,
  MainToWorker,
  PromptMode,
  WorkerToMain,
} from '../worker/protocol';
import {
  appendToken,
  emptyTree,
  prefixTokenIds,
  pruneGhosts,
  setPrompt,
  truncateForFork,
  type BranchTree,
} from './branchStore';

export type Phase = 'cold' | 'loading' | 'ready' | 'generating' | 'done' | 'error';

export interface LoadProgress {
  readonly file: string;
  readonly progress: number;
  readonly loaded: number;
  readonly total: number;
}

export interface RunStats {
  readonly tokensPerSecond: number;
  readonly totalMs: number;
  readonly reason: 'eos' | 'length' | 'cancelled';
}

export interface InferenceState {
  phase: Phase;
  device: Device | null;
  loadProgress: LoadProgress | null;
  loadMs: number | null;
  tree: BranchTree;
  cloud: Candidate[];
  error: { code: string; message: string; recoverable: boolean } | null;
  stats: RunStats | null;
  prompt: string;
}

const INITIAL: InferenceState = {
  phase: 'cold',
  device: null,
  loadProgress: null,
  loadMs: null,
  tree: emptyTree,
  cloud: [],
  error: null,
  stats: null,
  prompt: '',
};

export function useInference() {
  const workerRef = useRef<Worker | null>(null);
  const runIdRef = useRef<string | null>(null);
  const promptRef = useRef<string>('');
  const modeRef = useRef<PromptMode>('complete');
  const [state, setState] = useState<InferenceState>(INITIAL);
  /** Latest committed tree, readable from event handlers without a stale closure. */
  const treeRef = useRef(state.tree);
  treeRef.current = state.tree;
  const [temperature, setTemperatureState] = useState(config.defaultTemperature);
  const temperatureRef = useRef(temperature);
  temperatureRef.current = temperature;

  const send = useCallback((message: MainToWorker) => {
    workerRef.current?.postMessage(message);
  }, []);

  useEffect(() => {
    const worker = new Worker(new URL('../worker/inference.worker.ts', import.meta.url), {
      type: 'module',
    });
    workerRef.current = worker;

    worker.onmessage = (event: MessageEvent<WorkerToMain>) => {
      const msg = event.data;
      switch (msg.type) {
        case 'LOAD_PROGRESS':
          setState((s) => ({
            ...s,
            phase: 'loading',
            loadProgress: { file: msg.file, progress: msg.progress, loaded: msg.loaded, total: msg.total },
          }));
          return;

        case 'READY':
          log('info', 'ui.ready', { device: msg.device, loadMs: msg.loadMs });
          setState((s) => ({ ...s, phase: 'ready', device: msg.device, loadMs: msg.loadMs, loadProgress: null }));
          return;

        case 'PROMPT_TOKENS':
          if (msg.runId !== runIdRef.current) return;
          setState((s) => ({ ...s, tree: setPrompt(s.tree, msg.tokenIds, msg.texts) }));
          return;

        case 'TOKEN': {
          if (msg.runId !== runIdRef.current) return;
          setState((s) => ({
            ...s,
            phase: 'generating',
            cloud: msg.candidates,
            tree: appendToken(s.tree, {
              step: msg.step,
              chosen: msg.chosen,
              candidates: msg.candidates,
              temperature: msg.temperature,
              chosenByUser: msg.forced,
            }),
          }));
          return;
        }

        case 'DONE':
          if (msg.runId !== runIdRef.current) return;
          runIdRef.current = null;
          setState((s) => ({
            ...s,
            phase: 'done',
            stats: { tokensPerSecond: msg.tokensPerSecond, totalMs: msg.totalMs, reason: msg.reason },
            tree: pruneGhosts(s.tree, config.maxRetainedBranches),
          }));
          return;

        case 'ERROR':
          runIdRef.current = null;
          setState((s) => ({
            ...s,
            phase: 'error',
            error: { code: msg.code, message: msg.message, recoverable: msg.recoverable },
          }));
          return;
      }
    };

    worker.onerror = (event) => {
      setState((s) => ({
        ...s,
        phase: 'error',
        error: { code: 'INFERENCE', message: event.message || 'worker crashed', recoverable: false },
      }));
    };

    return () => {
      worker.postMessage({ type: 'DISPOSE' } satisfies MainToWorker);
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  const params = useCallback(
    (seed?: number): GenParams => ({
      temperature: temperatureRef.current,
      topK: config.topK,
      topP: config.defaultTopP,
      maxNewTokens: config.maxNewTokens,
      seed,
    }),
    [],
  );

  const load = useCallback(() => {
    setState((s) => ({ ...s, phase: 'loading', error: null }));
    send({
      type: 'LOAD',
      modelId: config.modelId,
      dtype: config.dtype,
      timeoutMs: config.modelLoadTimeoutMs,
    });
  }, [send]);

  const generate = useCallback(
    (prompt: string, mode: PromptMode, seed?: number) => {
      const runId = `r${Date.now().toString(36)}`;
      runIdRef.current = runId;
      promptRef.current = prompt;
      modeRef.current = mode;
      setState((s) => ({
        ...s,
        phase: 'generating',
        prompt,
        tree: emptyTree,
        cloud: [],
        stats: null,
        error: null,
      }));
      send({ type: 'GENERATE', runId, prompt, mode, params: params(seed) });
    },
    [params, send],
  );

  /**
   * Fork at `step` by forcing `tokenId`; everything before `step` is replayed.
   *
   * The message is sent OUTSIDE the state updater. Updaters must be pure: React
   * StrictMode invokes them twice in development, which sent FORK twice and started
   * two concurrent decode loops on the same model.
   */
  const fork = useCallback(
    (step: number, tokenId: number) => {
      const runId = `r${Date.now().toString(36)}f`;
      runIdRef.current = runId;
      const prefix = prefixTokenIds(treeRef.current, step);
      send({
        type: 'FORK',
        runId,
        prompt: promptRef.current,
        mode: modeRef.current,
        prefixTokenIds: prefix,
        forcedTokenId: tokenId,
        params: params(),
      });
      setState((s) => ({ ...s, phase: 'generating', cloud: [], stats: null, tree: truncateForFork(s.tree, step) }));
    },
    [params, send],
  );

  const cancel = useCallback(() => {
    if (runIdRef.current) send({ type: 'CANCEL', runId: runIdRef.current });
  }, [send]);

  const setTemperature = useCallback(
    (value: number) => {
      setTemperatureState(value);
      send({ type: 'SET_TEMPERATURE', temperature: value });
    },
    [send],
  );

  const reset = useCallback(() => {
    runIdRef.current = null;
    setState((s) => ({ ...INITIAL, phase: s.device ? 'ready' : 'cold', device: s.device, loadMs: s.loadMs }));
  }, []);

  return useMemo(
    () => ({ state, temperature, load, generate, fork, cancel, setTemperature, reset }),
    [state, temperature, load, generate, fork, cancel, setTemperature, reset],
  );
}
