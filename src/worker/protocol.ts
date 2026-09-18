/** Worker message contract — spec §7. Imported by BOTH sides; no stringly-typed messages. */
import type { ErrorCode } from '../errors';

export interface Candidate {
  readonly tokenId: number;
  readonly text: string; // decoded, whitespace already made visible
  readonly raw: string; // decoded, verbatim
  readonly p: number; // post-temperature probability 0..1
  readonly logit: number;
}

export interface GenParams {
  readonly temperature: number;
  readonly topK: number;
  readonly topP: number;
  readonly maxNewTokens: number;
  readonly seed?: number;
}

export type Device = 'webgpu' | 'wasm';

/**
 * 'complete' feeds the prompt verbatim — the model simply continues the text, which is
 * what actually demonstrates next-token prediction. 'chat' wraps it in the instruct
 * template, which turns a completion prompt into a question and produces an assistant
 * reply instead of a continuation.
 */
export type PromptMode = 'complete' | 'chat';

export type MainToWorker =
  | { type: 'LOAD'; modelId: string; dtype: string; timeoutMs: number }
  | { type: 'GENERATE'; runId: string; prompt: string; mode: PromptMode; params: GenParams }
  | {
      type: 'FORK';
      runId: string;
      prompt: string;
      mode: PromptMode;
      prefixTokenIds: number[];
      forcedTokenId: number;
      params: GenParams;
    }
  | { type: 'SET_TEMPERATURE'; temperature: number }
  | { type: 'CANCEL'; runId: string }
  | { type: 'DISPOSE' };

export type WorkerToMain =
  | { type: 'LOAD_PROGRESS'; file: string; progress: number; loaded: number; total: number }
  | {
      type: 'READY';
      modelId: string;
      device: Device;
      vocabSize: number;
      loadMs: number;
    }
  | { type: 'PROMPT_TOKENS'; runId: string; tokenIds: number[]; texts: string[] }
  | {
      type: 'TOKEN';
      runId: string;
      step: number;
      chosen: Candidate;
      candidates: Candidate[];
      temperature: number;
      forced: boolean;
      msSinceLast: number;
    }
  | {
      type: 'DONE';
      runId: string;
      reason: 'eos' | 'length' | 'cancelled';
      totalMs: number;
      tokensPerSecond: number;
    }
  | { type: 'ERROR'; runId: string | null; code: ErrorCode; message: string; recoverable: boolean };

/** Whitespace made visible so token boundaries are legible in the UI (spec §2.2). */
export function visibleText(raw: string): string {
  return raw.replace(/\n/g, '⏎').replace(/\t/g, '⇥').replace(/ /g, '·');
}
