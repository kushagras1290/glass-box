/**
 * Phase 0b — decisive comparison for the degenerate-continuation problem.
 *
 * Runs the SAME prompt three ways and prints all three:
 *   1. library generate() greedy   — reference implementation, threads KV internally
 *   2. manual KV loop, greedy      — our decode loop (spec §6.4)
 *   3. manual loop without position_ids — isolates whether position_ids is the culprit
 *
 * If (1) is coherent and (2) is not, our threading is wrong.
 * If (1) is also degenerate, the q4 weights are the problem, not the loop.
 */
import { AutoTokenizer, AutoModelForCausalLM, Tensor, env } from '@huggingface/transformers';

interface Probe {
  done: boolean;
  failed: string | null;
  generateRef: string | null;
  manualWithPositions: string | null;
  manualWithoutPositions: string | null;
  dtype: string;
  matches: boolean | null;
}

const probe: Probe = {
  done: false,
  failed: null,
  generateRef: null,
  manualWithPositions: null,
  manualWithoutPositions: null,
  dtype: 'q4',
  matches: null,
};
(window as unknown as { __PROBE__: Probe }).__PROBE__ = probe;

const el = document.getElementById('out') as HTMLPreElement;
const log = (...p: unknown[]): void => {
  const line = p.map(String).join(' ');
  el.textContent += '\n' + line;
  console.log(line);
};

env.allowLocalModels = false;

const MODEL_ID = 'HuggingFaceTB/SmolLM2-360M-Instruct';
const DTYPE = (new URLSearchParams(location.search).get('dtype') ?? 'q4') as 'q4' | 'q8' | 'fp16' | 'fp32';
const STEPS = 16;

function argmax(row: Float32Array): number {
  let bestId = 0;
  let bestVal = -Infinity;
  for (let i = 0; i < row.length; i++) if (row[i] > bestVal) { bestVal = row[i]; bestId = i; }
  return bestId;
}
function lastRow(t: { dims: readonly number[]; data: Float32Array }): Float32Array {
  const seq = t.dims[1];
  const vocab = t.dims[2];
  return t.data.subarray((seq - 1) * vocab, seq * vocab) as Float32Array;
}
const toPast = (src: Record<string, unknown>): Record<string, unknown> => {
  const out: Record<string, unknown> = {};
  for (const k of Object.keys(src)) {
    if (k.startsWith('present.')) out[k.replace('present.', 'past_key_values.')] = src[k];
  }
  return out;
};

async function manualLoop(
  model: (i: Record<string, unknown>) => Promise<Record<string, unknown>>,
  inputs: Record<string, unknown>,
  promptLen: number,
  usePositionIds: boolean,
): Promise<number[]> {
  const first = await model({ ...inputs });
  let past = toPast(first);
  let nextId = argmax(lastRow(first.logits as never));
  const generated: number[] = [];

  for (let step = 0; step < STEPS; step++) {
    generated.push(nextId);
    const total = promptLen + generated.length;
    // past_key_values must be passed as ONE nested object; spreading the
    // `past_key_values.N.key` names at the top level is silently ignored and the
    // model then runs with an empty cache (verified in phase0c trace).
    const feed: Record<string, unknown> = {
      input_ids: new Tensor('int64', BigInt64Array.from([BigInt(nextId)]), [1, 1]),
      attention_mask: new Tensor('int64', BigInt64Array.from({ length: total }, () => 1n), [1, total]),
      past_key_values: past,
    };
    if (usePositionIds) {
      feed.position_ids = new Tensor('int64', BigInt64Array.from([BigInt(total - 1)]), [1, 1]);
    }
    const out = await model(feed);
    past = toPast(out);
    nextId = argmax(lastRow(out.logits as never));
  }
  return generated;
}

async function run(): Promise<void> {
  probe.dtype = DTYPE;
  log('dtype:', DTYPE);

  const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
  const model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
    dtype: DTYPE,
    device: (navigator as unknown as { gpu?: unknown }).gpu ? 'webgpu' : 'wasm',
  });
  log('model ready');

  const prompt = tokenizer.apply_chat_template(
    [{ role: 'user', content: 'Name three colours.' }],
    { tokenize: false, add_generation_prompt: true },
  ) as string;
  const inputs = await tokenizer(prompt);
  const promptLen = inputs.input_ids.dims[1] as number;

  // 1 — reference
  const refOut = await model.generate({
    ...inputs,
    max_new_tokens: STEPS,
    do_sample: false,
  });
  const refIds = Array.from((refOut as { data: BigInt64Array }).data).map(Number).slice(promptLen);
  probe.generateRef = tokenizer.decode(refIds, { skip_special_tokens: true });
  log('\n1) generate() greedy:', JSON.stringify(probe.generateRef));

  // 2 — manual with position_ids
  const withPos = await manualLoop(model as never, inputs as never, promptLen, true);
  probe.manualWithPositions = tokenizer.decode(withPos, { skip_special_tokens: true });
  log('2) manual +position_ids:', JSON.stringify(probe.manualWithPositions));

  // 3 — manual without position_ids
  const withoutPos = await manualLoop(model as never, inputs as never, promptLen, false);
  probe.manualWithoutPositions = tokenizer.decode(withoutPos, { skip_special_tokens: true });
  log('3) manual -position_ids:', JSON.stringify(probe.manualWithoutPositions));

  probe.matches =
    probe.generateRef.trim() === probe.manualWithPositions.trim() ||
    probe.generateRef.trim() === probe.manualWithoutPositions.trim();
  log('\nmanual loop matches reference:', String(probe.matches));
  probe.done = true;
}

run()
  .then(() => log('\nPROBE COMPLETE'))
  .catch((e: unknown) => {
    probe.failed = e instanceof Error ? e.message : String(e);
    probe.done = true;
    log('\nFAILED:', probe.failed);
    console.error(e);
  });
