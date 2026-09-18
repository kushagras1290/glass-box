/** Worker entry + message router — spec §3.1, §7. Never throws across the boundary. */
import { toGlassBoxError, ProtocolError } from '../errors';
import { log } from '../log';
import { InferenceEngine } from './engine';
import type { MainToWorker, PromptMode, WorkerToMain } from './protocol';

const engine = new InferenceEngine();
let activeRunId: string | null = null;
/**
 * The in-flight run. Runs are serialised: a new GENERATE/FORK cancels the previous run
 * and WAITS for it to stop before starting, so two decode loops never drive the same
 * ONNX session concurrently.
 */
let currentRun: Promise<void> | null = null;

function post(message: WorkerToMain): void {
  (self as unknown as Worker).postMessage(message);
}

function fail(runId: string | null, err: unknown): void {
  const mapped = toGlassBoxError(err);
  log('error', 'worker.error', { code: mapped.code, message: mapped.message, runId });
  post({
    type: 'ERROR',
    runId,
    code: mapped.code,
    message: mapped.message,
    recoverable: mapped.recoverable,
  });
}

async function handleLoad(msg: Extract<MainToWorker, { type: 'LOAD' }>): Promise<void> {
  const loadMs = await engine.load(msg.modelId, msg.dtype, msg.timeoutMs, (file, progress, loaded, total) => {
    post({ type: 'LOAD_PROGRESS', file, progress, loaded, total });
  });
  post({
    type: 'READY',
    modelId: msg.modelId,
    device: engine.resolvedDevice,
    vocabSize: engine.resolvedVocabSize,
    loadMs: Math.round(loadMs),
  });
}

async function execute(
  runId: string,
  prompt: string,
  mode: PromptMode,
  params: Extract<MainToWorker, { type: 'GENERATE' }>['params'],
  prefixTokenIds: number[],
  forcedTokenId: number | null,
): Promise<void> {
  activeRunId = runId;
  const built = engine.buildPrompt(prompt, mode);

  const { reason, totalMs, steps } = await engine.run(
    built,
    params,
    {
      onPromptTokens: ({ promptTokenIds, promptTexts }) => {
        if (activeRunId !== runId) return;
        post({ type: 'PROMPT_TOKENS', runId, tokenIds: promptTokenIds, texts: promptTexts });
      },
      onStep: (event) => {
        if (activeRunId !== runId) return; // drop late steps from a superseded run
        post({
          type: 'TOKEN',
          runId,
          step: event.step,
          chosen: event.chosen,
          candidates: event.candidates,
          temperature: event.temperature,
          forced: event.forced,
          msSinceLast: event.msSinceLast,
        });
      },
    },
    prefixTokenIds,
    forcedTokenId,
  );

  if (activeRunId !== runId) return;
  post({
    type: 'DONE',
    runId,
    reason,
    totalMs: Math.round(totalMs),
    tokensPerSecond: steps > 0 ? Math.round((steps / totalMs) * 1000 * 10) / 10 : 0,
  });
  activeRunId = null;
}

self.onmessage = async (event: MessageEvent<MainToWorker>): Promise<void> => {
  const msg = event.data;
  try {
    switch (msg.type) {
      case 'LOAD':
        await handleLoad(msg);
        return;
      case 'GENERATE':
      case 'FORK': {
        if (currentRun) {
          engine.cancel();
          await currentRun.catch(() => undefined);
        }
        const run =
          msg.type === 'GENERATE'
            ? execute(msg.runId, msg.prompt, msg.mode, msg.params, [], null)
            : execute(msg.runId, msg.prompt, msg.mode, msg.params, msg.prefixTokenIds, msg.forcedTokenId);
        currentRun = run;
        try {
          await run;
        } finally {
          if (currentRun === run) currentRun = null;
        }
        return;
      }
      case 'SET_TEMPERATURE':
        engine.setTemperature(msg.temperature);
        return;
      case 'CANCEL':
        if (activeRunId === msg.runId) engine.cancel();
        return;
      case 'DISPOSE':
        await engine.dispose();
        return;
      default: {
        const unknownType = (msg as { type?: string }).type ?? 'undefined';
        throw new ProtocolError(`unhandled message type: ${unknownType}`);
      }
    }
  } catch (err) {
    fail('runId' in msg ? (msg as { runId: string }).runId : null, err);
    activeRunId = null;
  }
};
