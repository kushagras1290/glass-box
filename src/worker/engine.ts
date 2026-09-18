/**
 * Inference engine — spec §6. Runs inside the worker; owns the decode loop.
 *
 * We deliberately do NOT use generate(): forking is a KV-cache rewind and the
 * probability cloud needs per-step logits. Both are impossible through that API.
 *
 * KV-cache contract, established empirically in the Phase 0 trace (docs/phase0-results.md):
 *   - the ONNX export EMITS the cache as flat `present.<layer>.<key|value>` outputs
 *   - it CONSUMES it as ONE nested object under the single input name `past_key_values`
 *   - spreading `past_key_values.N.key` at the top level is silently ignored and the
 *     model then runs with an empty cache, producing fluent-looking garbage
 *   - `position_ids` must be supplied explicitly for cached steps
 */
import { AutoTokenizer, AutoModelForCausalLM, Tensor, env } from '@huggingface/transformers';
import {
  InferenceError,
  ModelLoadError,
  ModelTimeoutError,
  TokenizeError,
  toGlassBoxError,
} from '../errors';
import { log } from '../log';
import type { Candidate, Device, GenParams, PromptMode } from './protocol';
import { visibleText } from './protocol';
import { applyTopP, makeRng, sampleFrom, softmaxOver, topKFromLogits } from './sampler';

env.allowLocalModels = false;

interface LogitsTensor {
  readonly dims: readonly number[];
  readonly data: Float32Array;
}
type ModelOutput = Record<string, unknown> & { logits: LogitsTensor };
type CallableModel = ((inputs: Record<string, unknown>) => Promise<ModelOutput>) & {
  dispose?: () => Promise<void>;
};
interface LoadedTokenizer {
  (text: string): Promise<{ input_ids: { dims: number[]; data: BigInt64Array } }>;
  decode(ids: number[], options?: Record<string, unknown>): string;
  apply_chat_template(messages: unknown[], options: Record<string, unknown>): string | unknown;
}

export interface StepEvent {
  readonly step: number;
  readonly chosen: Candidate;
  readonly candidates: Candidate[];
  readonly temperature: number;
  /** true when this token was substituted by the visitor rather than sampled. */
  readonly forced: boolean;
  readonly msSinceLast: number;
}

export interface RunHandle {
  readonly promptTokenIds: number[];
  readonly promptTexts: string[];
}

export interface EngineCallbacks {
  onPromptTokens(handle: RunHandle): void;
  onStep(event: StepEvent): void;
}

/** present.N.key → past_key_values.N.key, returned as the single nested object. */
function extractPastKeyValues(output: Record<string, unknown>): Record<string, unknown> {
  const past: Record<string, unknown> = {};
  for (const key of Object.keys(output)) {
    if (key.startsWith('present.')) {
      past[key.replace('present.', 'past_key_values.')] = output[key];
    }
  }
  return past;
}

function lastLogitRow(logits: LogitsTensor): Float32Array {
  const seq = logits.dims[1];
  const vocab = logits.dims[2];
  return logits.data.subarray((seq - 1) * vocab, seq * vocab) as Float32Array;
}

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new ModelTimeoutError(`${label} exceeded ${ms}ms`)), ms);
    promise.then(
      (value) => { clearTimeout(timer); resolve(value); },
      (err) => { clearTimeout(timer); reject(err); },
    );
  });
}

export class InferenceEngine {
  private tokenizer: LoadedTokenizer | null = null;
  private model: CallableModel | null = null;
  private eosTokenIds: Set<number> = new Set();
  private vocabSize = 0;
  private device: Device = 'wasm';
  private cancelled = false;
  private liveTemperature: number | null = null;

  get ready(): boolean {
    return this.tokenizer !== null && this.model !== null;
  }
  get resolvedDevice(): Device {
    return this.device;
  }
  get resolvedVocabSize(): number {
    return this.vocabSize;
  }

  async load(
    modelId: string,
    dtype: string,
    timeoutMs: number,
    onProgress: (file: string, progress: number, loaded: number, total: number) => void,
  ): Promise<number> {
    const started = performance.now();
    this.device = (globalThis.navigator as unknown as { gpu?: unknown })?.gpu ? 'webgpu' : 'wasm';
    log('info', 'model.load.start', { modelId, dtype, device: this.device });

    try {
      this.tokenizer = (await withTimeout(
        AutoTokenizer.from_pretrained(modelId) as Promise<unknown>,
        timeoutMs,
        'tokenizer load',
      )) as LoadedTokenizer;

      this.model = (await withTimeout(
        AutoModelForCausalLM.from_pretrained(modelId, {
          dtype: dtype as 'q4',
          device: this.device,
          progress_callback: (p: { status?: string; file?: string; progress?: number; loaded?: number; total?: number }) => {
            if (p.status === 'progress') {
              onProgress(p.file ?? '', p.progress ?? 0, p.loaded ?? 0, p.total ?? 0);
            }
          },
        }) as Promise<unknown>,
        timeoutMs,
        'model load',
      )) as CallableModel;
    } catch (err) {
      const mapped = toGlassBoxError(err, 'MODEL_LOAD');
      throw mapped.code === 'MODEL_TIMEOUT' ? mapped : new ModelLoadError(mapped.message, err);
    }

    // EOS ids: generation_config first, then the chat-template terminator.
    const genConfig = (this.model as unknown as { generation_config?: { eos_token_id?: number | number[] } })
      .generation_config;
    const rawEos = genConfig?.eos_token_id;
    if (typeof rawEos === 'number') this.eosTokenIds.add(rawEos);
    else if (Array.isArray(rawEos)) for (const id of rawEos) this.eosTokenIds.add(id);
    const imEnd = (this.tokenizer as unknown as { model?: { tokens_to_ids?: Map<string, number> } })?.model
      ?.tokens_to_ids?.get('<|im_end|>');
    if (typeof imEnd === 'number') this.eosTokenIds.add(imEnd);

    const loadMs = performance.now() - started;
    log('info', 'model.load.done', { loadMs: Math.round(loadMs), eos: [...this.eosTokenIds] });
    return loadMs;
  }

  cancel(): void {
    this.cancelled = true;
  }

  setTemperature(temperature: number): void {
    this.liveTemperature = temperature;
  }

  buildPrompt(userText: string, mode: PromptMode): string {
    if (!this.tokenizer) throw new InferenceError('buildPrompt before load');
    if (mode === 'complete') return userText;
    try {
      return this.tokenizer.apply_chat_template([{ role: 'user', content: userText }], {
        tokenize: false,
        add_generation_prompt: true,
      }) as string;
    } catch (err) {
      throw new TokenizeError('apply_chat_template failed', err);
    }
  }

  /**
   * Decode loop. `forcedFirstTokenId` substitutes the first sampled token — that is
   * how a fork begins. `prefixTokenIds` replays an existing branch before continuing.
   */
  async run(
    prompt: string,
    params: GenParams,
    callbacks: EngineCallbacks,
    prefixTokenIds: number[] = [],
    forcedFirstTokenId: number | null = null,
  ): Promise<{ reason: 'eos' | 'length' | 'cancelled'; totalMs: number; steps: number }> {
    if (!this.model || !this.tokenizer) throw new InferenceError('run before load');
    this.cancelled = false;
    this.liveTemperature = null;
    const startedAt = performance.now();
    const rng = makeRng(params.seed);

    let encoded;
    try {
      encoded = await this.tokenizer(prompt);
    } catch (err) {
      throw new TokenizeError('prompt tokenization failed', err);
    }

    const promptIds = Array.from(encoded.input_ids.data).map(Number);
    callbacks.onPromptTokens({
      promptTokenIds: promptIds,
      promptTexts: promptIds.map((id) => visibleText(this.decodeOne(id))),
    });

    // Prefill over prompt + any replayed branch prefix, in one batched pass.
    const seedIds = [...promptIds, ...prefixTokenIds];
    const seedLen = seedIds.length;
    let output: ModelOutput;
    try {
      output = await this.model({
        input_ids: new Tensor('int64', BigInt64Array.from(seedIds.map(BigInt)), [1, seedLen]),
        attention_mask: new Tensor('int64', BigInt64Array.from({ length: seedLen }, () => 1n), [1, seedLen]),
      });
    } catch (err) {
      throw toGlassBoxError(err, 'INFERENCE');
    }
    this.vocabSize = output.logits.dims[2];

    let past = extractPastKeyValues(output);
    let row = lastLogitRow(output.logits);
    const generated: number[] = [];
    let lastTokenAt = performance.now();
    let reason: 'eos' | 'length' | 'cancelled' = 'length';

    // Steps are ABSOLUTE positions in the generated sequence. A fork replays N prefix
    // tokens, so its first new token is step N — not 0. Numbering from 0 made the store
    // discard the forced token and the next ones as "stale", so the timeline showed text
    // the model never produced in that order.
    const stepOffset = prefixTokenIds.length;

    for (let i = 0; i < params.maxNewTokens; i++) {
      if (this.cancelled) { reason = 'cancelled'; break; }
      const step = stepOffset + i;

      const temperature = this.liveTemperature ?? params.temperature;
      const ranked = softmaxOver(topKFromLogits(row, params.topK), temperature);
      const pool = applyTopP(ranked, params.topP);

      const forced = i === 0 && forcedFirstTokenId !== null
        ? ranked.find((c) => c.id === forcedFirstTokenId) ?? {
            id: forcedFirstTokenId,
            logit: Number.NaN,
            p: Number.NaN,
          }
        : null;
      const picked = forced ?? sampleFrom(pool, rng);

      // Stop BEFORE emitting: an end-of-sequence token is a decision to stop talking,
      // not a word. Emitting it would put "<|im_end|>" into the visible output.
      if (this.eosTokenIds.has(picked.id)) { reason = 'eos'; break; }

      const candidates: Candidate[] = ranked.map((c) => this.toCandidate(c.id, c.p, c.logit));
      const chosen = this.toCandidate(picked.id, picked.p, picked.logit);

      const now = performance.now();
      callbacks.onStep({
        step,
        chosen,
        candidates,
        temperature,
        forced: forced !== null,
        msSinceLast: Math.round(now - lastTokenAt),
      });
      lastTokenAt = now;

      generated.push(picked.id);
      const total = seedLen + generated.length;

      try {
        output = await this.model({
          input_ids: new Tensor('int64', BigInt64Array.from([BigInt(picked.id)]), [1, 1]),
          attention_mask: new Tensor('int64', BigInt64Array.from({ length: total }, () => 1n), [1, total]),
          position_ids: new Tensor('int64', BigInt64Array.from([BigInt(total - 1)]), [1, 1]),
          past_key_values: past,
        });
      } catch (err) {
        throw toGlassBoxError(err, 'INFERENCE');
      }
      past = extractPastKeyValues(output);
      row = lastLogitRow(output.logits);

      // Yield so CANCEL / SET_TEMPERATURE messages are processed promptly.
      await Promise.resolve();
    }

    const totalMs = performance.now() - startedAt;
    log('info', 'run.done', { reason, steps: generated.length, totalMs: Math.round(totalMs) });
    return { reason, totalMs, steps: generated.length };
  }

  private decodeOne(id: number): string {
    if (!this.tokenizer) return '';
    try {
      return this.tokenizer.decode([id]);
    } catch {
      return '�';
    }
  }

  private toCandidate(id: number, p: number, logit: number): Candidate {
    const raw = this.decodeOne(id);
    return { tokenId: id, raw, text: visibleText(raw), p, logit };
  }

  async dispose(): Promise<void> {
    await this.model?.dispose?.();
    this.model = null;
    this.tokenizer = null;
  }
}
