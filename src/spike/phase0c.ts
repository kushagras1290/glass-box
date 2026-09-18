/**
 * Phase 0c — instrument the reference implementation.
 *
 * Wraps model.forward and records the exact tensor names, dtypes and dims that
 * transformers.js feeds on the prefill step and on the first two cached steps.
 * We then replicate those feeds byte-for-byte in our own loop (spec §6.4).
 */
import { AutoTokenizer, AutoModelForCausalLM, env } from '@huggingface/transformers';

interface FeedRecord {
  call: number;
  keys: Array<{ name: string; type: string; dims: number[]; sample?: string }>;
}
interface Trace {
  done: boolean;
  failed: string | null;
  feeds: FeedRecord[];
  output: string | null;
}

const trace: Trace = { done: false, failed: null, feeds: [], output: null };
(window as unknown as { __TRACE__: Trace }).__TRACE__ = trace;

const el = document.getElementById('out') as HTMLPreElement;
const log = (...p: unknown[]): void => {
  const line = p.map(String).join(' ');
  el.textContent += '\n' + line;
  console.log(line);
};

env.allowLocalModels = false;
const MODEL_ID = 'HuggingFaceTB/SmolLM2-360M-Instruct';

function describe(value: unknown): { type: string; dims: number[]; sample?: string } {
  const t = value as { type?: string; dims?: number[]; data?: ArrayLike<unknown> };
  if (t && Array.isArray(t.dims)) {
    const sample =
      t.data && t.data.length <= 40
        ? Array.from(t.data as ArrayLike<unknown>).map(String).join(',')
        : undefined;
    return { type: t.type ?? 'tensor', dims: t.dims, sample };
  }
  return { type: typeof value, dims: [] };
}

async function run(): Promise<void> {
  const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
  const model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
    dtype: 'q4',
    device: (navigator as unknown as { gpu?: unknown }).gpu ? 'webgpu' : 'wasm',
  });
  log('model ready — instrumenting forward()');

  const target = model as unknown as {
    forward: (inputs: Record<string, unknown>) => Promise<unknown>;
  };
  const originalForward = target.forward.bind(target);
  let call = 0;

  target.forward = async (inputs: Record<string, unknown>) => {
    if (call < 3) {
      const keys = Object.keys(inputs)
        .filter((k) => !k.startsWith('past_key_values.') || k.endsWith('.0.key'))
        .map((name) => ({ name, ...describe(inputs[name]) }));
      const pkvCount = Object.keys(inputs).filter((k) => k.startsWith('past_key_values.')).length;
      keys.push({ name: `…past_key_values.* total`, type: 'count', dims: [pkvCount] });
      trace.feeds.push({ call, keys });
      log(`\n── forward call ${call} ──`);
      for (const k of keys) {
        log(`  ${k.name}  [${k.dims.join(',')}] ${k.type}${k.sample ? ' = ' + k.sample : ''}`);
      }
    }
    call++;
    return originalForward(inputs);
  };

  const prompt = tokenizer.apply_chat_template(
    [{ role: 'user', content: 'Name three colours.' }],
    { tokenize: false, add_generation_prompt: true },
  ) as string;
  const inputs = await tokenizer(prompt);

  const out = await model.generate({ ...inputs, max_new_tokens: 8, do_sample: false });
  const promptLen = inputs.input_ids.dims[1] as number;
  const ids = Array.from((out as { data: BigInt64Array }).data).map(Number).slice(promptLen);
  trace.output = tokenizer.decode(ids, { skip_special_tokens: true });
  log('\nreference output:', JSON.stringify(trace.output));
  trace.done = true;
}

run()
  .then(() => log('\nTRACE COMPLETE'))
  .catch((e: unknown) => {
    trace.failed = e instanceof Error ? e.message : String(e);
    trace.done = true;
    log('\nFAILED:', trace.failed);
    console.error(e);
  });
