/**
 * Phase 0 spike — the gate defined in GLASS_BOX_TECHNICAL_SPEC.md §4.
 *
 * Proves or kills six assertions before any product code is written:
 *   A1 model loads in-browser (WebGPU preferred)
 *   A2 forward pass returns logits [1, seq, vocab]
 *   A3 logits.data is a readable Float32Array
 *   A4 output carries reusable past_key_values
 *   A5 feeding past_key_values back yields coherent continuation
 *   A6 tokenizer.apply_chat_template exists and emits a sane instruct prompt
 *
 * Results are mirrored onto window.__SPIKE__ so a driver can read them.
 */
import {
  AutoTokenizer,
  AutoModelForCausalLM,
  Tensor,
  env,
} from '@huggingface/transformers';

interface SpikeResult {
  done: boolean;
  failed: string | null;
  assertions: Record<string, boolean | string | null>;
  metrics: Record<string, number | string>;
  topFive: Array<{ id: number; p: number; text: string }>;
  continuation: string | null;
}

const result: SpikeResult = {
  done: false,
  failed: null,
  assertions: { A1: null, A2: null, A3: null, A4: null, A5: null, A6: null },
  metrics: {},
  topFive: [],
  continuation: null,
};
(window as unknown as { __SPIKE__: SpikeResult }).__SPIKE__ = result;

const el = document.getElementById('out') as HTMLPreElement;
const log = (...parts: unknown[]): void => {
  const line = parts.map(String).join(' ');
  el.textContent += '\n' + line;
  console.log(line);
};

env.allowLocalModels = false;

const MODEL_ID = 'HuggingFaceTB/SmolLM2-360M-Instruct';

interface RawCandidate { id: number; logit: number }

/** Bounded-heap top-k so we never sort the full vocabulary (spec §6.7). */
function topKFromLogits(logits: Float32Array, k: number): RawCandidate[] {
  const best: RawCandidate[] = [];
  for (let i = 0; i < logits.length; i++) {
    const logit = logits[i];
    if (best.length < k) {
      best.push({ id: i, logit });
      best.sort((a, b) => a.logit - b.logit);
    } else if (logit > best[0].logit) {
      best[0] = { id: i, logit };
      best.sort((a, b) => a.logit - b.logit);
    }
  }
  return best.reverse();
}

/** Max-subtracted softmax — without the subtraction fp32 overflows to NaN (spec §6.7). */
function softmax(cands: RawCandidate[], temperature = 1): Array<RawCandidate & { p: number }> {
  if (temperature === 0) return cands.map((c, i) => ({ ...c, p: i === 0 ? 1 : 0 }));
  const max = Math.max(...cands.map((c) => c.logit));
  const exps = cands.map((c) => Math.exp((c.logit - max) / temperature));
  const sum = exps.reduce((a, b) => a + b, 0);
  return cands.map((c, i) => ({ ...c, p: exps[i] / sum }));
}

function lastRow(logits: { dims: readonly number[]; data: Float32Array }): Float32Array {
  const seq = logits.dims[1];
  const vocab = logits.dims[2];
  return logits.data.subarray((seq - 1) * vocab, seq * vocab) as Float32Array;
}

async function run(): Promise<void> {
  const hasGPU = Boolean((navigator as unknown as { gpu?: unknown }).gpu);
  const device: 'webgpu' | 'wasm' = hasGPU ? 'webgpu' : 'wasm';
  log('webgpu available:', String(hasGPU), '→ device:', device);
  result.metrics.device = device;

  // ── A6 ──────────────────────────────────────────────────────────────────
  const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
  const hasTemplate = typeof (tokenizer as { apply_chat_template?: unknown }).apply_chat_template === 'function';
  result.assertions.A6 = hasTemplate;
  log(hasTemplate ? 'A6 OK — apply_chat_template present' : 'A6 FAIL — no apply_chat_template');

  // ── A1 ──────────────────────────────────────────────────────────────────
  const loadStart = performance.now();
  let lastPct = -1;
  const model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
    dtype: 'q4',
    device,
    progress_callback: (p: { status?: string; file?: string; progress?: number }) => {
      if (p.status === 'progress' && typeof p.progress === 'number') {
        const pct = Math.floor(p.progress / 10) * 10;
        if (pct !== lastPct) { lastPct = pct; log(`  ${p.file ?? ''} ${pct}%`); }
      }
    },
  });
  const loadMs = performance.now() - loadStart;
  result.assertions.A1 = true;
  result.metrics.loadMs = Math.round(loadMs);
  log(`A1 OK — model loaded in ${(loadMs / 1000).toFixed(1)}s`);

  const prompt = hasTemplate
    ? (tokenizer.apply_chat_template([{ role: 'user', content: 'Name three colours.' }], {
        tokenize: false,
        add_generation_prompt: true,
      }) as string)
    : 'User: Name three colours.\nAssistant:';
  log('prompt:', JSON.stringify(prompt).slice(0, 160));

  const inputs = await tokenizer(prompt);
  const promptLen = inputs.input_ids.dims[1] as number;
  log('prompt tokens:', String(promptLen));

  // ── A2 / A3 ─────────────────────────────────────────────────────────────
  const prefillStart = performance.now();
  const out = await model({ ...inputs });
  const prefillMs = performance.now() - prefillStart;
  result.metrics.prefillMs = Math.round(prefillMs);
  log('forward keys:', Object.keys(out).join(', '));

  const logits = out.logits as unknown as { dims: readonly number[]; data: Float32Array };
  const dimsOk = logits?.dims?.length === 3;
  result.assertions.A2 = dimsOk;
  log(dimsOk ? 'A2 OK — logits dims ' + JSON.stringify(logits.dims) : 'A2 FAIL — no 3D logits');

  const dataOk = logits.data instanceof Float32Array;
  result.assertions.A3 = dataOk;
  log(`A3 ${dataOk ? 'OK' : 'FAIL'} — data ${logits.data?.constructor?.name} len ${logits.data?.length}`);
  result.metrics.vocabSize = logits.dims[2];
  result.metrics.prefillMsPerToken = Math.round(prefillMs / promptLen);

  const five = softmax(topKFromLogits(lastRow(logits), 5));
  result.topFive = five.map((c) => ({ id: c.id, p: c.p, text: tokenizer.decode([c.id]) }));
  log('\nTOP-5 NEXT TOKENS:');
  for (const c of result.topFive) {
    log(`  ${(c.p * 100).toFixed(1).padStart(5)}%  id=${String(c.id).padStart(6)}  ${JSON.stringify(c.text)}`);
  }

  // ── A4 / A5 ─────────────────────────────────────────────────────────────
  // The ONNX export emits the KV cache as flat `present.N.key` / `present.N.value`
  // outputs and consumes it as `past_key_values.N.key` / `.value` inputs. There is no
  // single `past_key_values` object — the names must be remapped every step.
  const cacheKeys = Object.keys(out).filter((k) => k.startsWith('present.'));
  const hasPkv = cacheKeys.length > 0;
  result.assertions.A4 = hasPkv;
  result.metrics.cacheTensors = cacheKeys.length;
  log('\nA4', hasPkv ? `OK — ${cacheKeys.length} cache tensors (present.*)` : 'FAIL — no cache tensors');

  if (!hasPkv) {
    result.assertions.A5 = false;
    result.done = true;
    return;
  }

  /** present.N.key → past_key_values.N.key (and .value), for the next forward pass. */
  const toPastKeyValues = (source: Record<string, unknown>): Record<string, unknown> => {
    const mapped: Record<string, unknown> = {};
    for (const key of Object.keys(source)) {
      if (key.startsWith('present.')) {
        mapped[key.replace('present.', 'past_key_values.')] = source[key];
      }
    }
    return mapped;
  };

  const generated: number[] = [];
  let past = toPastKeyValues(out as unknown as Record<string, unknown>);
  let nextId = result.topFive[0].id;
  const stepTimes: number[] = [];

  for (let step = 0; step < 16; step++) {
    generated.push(nextId);
    const total = promptLen + generated.length;
    const stepStart = performance.now();
    const stepOut = await model({
      input_ids: new Tensor('int64', BigInt64Array.from([BigInt(nextId)]), [1, 1]),
      attention_mask: new Tensor('int64', BigInt64Array.from({ length: total }, () => 1n), [1, total]),
      position_ids: new Tensor('int64', BigInt64Array.from([BigInt(total - 1)]), [1, 1]),
      ...past,
    });
    stepTimes.push(performance.now() - stepStart);
    past = toPastKeyValues(stepOut as unknown as Record<string, unknown>);
    const stepLogits = stepOut.logits as unknown as { dims: readonly number[]; data: Float32Array };
    nextId = topKFromLogits(lastRow(stepLogits), 1)[0].id;
  }

  const continuation = tokenizer.decode(generated);
  result.continuation = continuation;
  // Coherence proxy: real text, not repeated padding or a single repeating token.
  const distinct = new Set(generated).size;
  const coherent = continuation.trim().length > 0 && distinct > 3;
  result.assertions.A5 = coherent;

  stepTimes.sort((a, b) => a - b);
  result.metrics.stepMsP50 = Math.round(stepTimes[Math.floor(stepTimes.length / 2)]);
  result.metrics.stepMsP95 = Math.round(stepTimes[Math.floor(stepTimes.length * 0.95)]);
  result.metrics.tokensPerSecond = Math.round(1000 / (stepTimes.reduce((a, b) => a + b, 0) / stepTimes.length));

  log('A5', coherent ? 'OK' : 'FAIL', '— continuation:', JSON.stringify(continuation));
  log(`\nstep ms p50=${result.metrics.stepMsP50} p95=${result.metrics.stepMsP95} · ${result.metrics.tokensPerSecond} tok/s`);
  result.done = true;
}

run()
  .then(() => log('\nSPIKE COMPLETE'))
  .catch((err: unknown) => {
    result.failed = err instanceof Error ? err.message : String(err);
    result.done = true;
    log('\nFAILED:', result.failed);
    console.error(err);
  });
