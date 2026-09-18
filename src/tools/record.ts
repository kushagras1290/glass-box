/**
 * Recorder — captures the REAL model data that the narrative chapters display.
 *
 * The story chapters (I–IV) show recorded distributions rather than live ones so a
 * visitor sees the idea before paying the model download. Those numbers must come
 * from this exact model, not be invented; this script is the provenance. Re-run it
 * (open /record.html) whenever the model or dtype changes, and paste the JSON into
 * src/data/recorded.json.
 */
import { AutoTokenizer, AutoModelForCausalLM, env } from '@huggingface/transformers';
import { config } from '../config';
import { topKFromLogits } from '../worker/sampler';
import { visibleText } from '../worker/protocol';

env.allowLocalModels = false;

const TOP_K = 24;
const SHATTER_TEXT = 'The capital of France is';
const PROMPTS = {
  certain: 'The capital of France is',
  open: 'The rain began just after midnight, and',
} as const;

interface RecordedCandidate { id: number; text: string; logit: number }

const out = document.getElementById('out') as HTMLPreElement;

async function record(): Promise<void> {
  const tokenizer = await AutoTokenizer.from_pretrained(config.modelId);
  const model = await AutoModelForCausalLM.from_pretrained(config.modelId, {
    dtype: config.dtype,
    device: (navigator as unknown as { gpu?: unknown }).gpu ? 'webgpu' : 'wasm',
  });

  const encodedShatter = await tokenizer(SHATTER_TEXT);
  const shatterIds = Array.from(encodedShatter.input_ids.data as BigInt64Array).map(Number);

  const distributions: Record<string, { prompt: string; candidates: RecordedCandidate[] }> = {};
  for (const [key, prompt] of Object.entries(PROMPTS)) {
    const inputs = await tokenizer(prompt);
    const result = await model({ ...inputs });
    const logits = result.logits as unknown as { dims: number[]; data: Float32Array };
    const seq = logits.dims[1];
    const vocab = logits.dims[2];
    const row = logits.data.subarray((seq - 1) * vocab, seq * vocab) as Float32Array;
    distributions[key] = {
      prompt,
      candidates: topKFromLogits(row, TOP_K).map((c) => ({
        id: c.id,
        text: visibleText(tokenizer.decode([c.id])),
        logit: Math.round(c.logit * 10_000) / 10_000,
      })),
    };
  }

  const payload = {
    model: config.modelId,
    dtype: config.dtype,
    topK: TOP_K,
    recordedAt: new Date().toISOString().slice(0, 10),
    shatter: {
      text: SHATTER_TEXT,
      tokens: shatterIds.map((id) => ({ id, text: visibleText(tokenizer.decode([id])) })),
    },
    distributions,
  };

  (window as unknown as { __RECORD__: unknown }).__RECORD__ = payload;
  out.textContent = JSON.stringify(payload, null, 2);
}

record().catch((err: unknown) => {
  (window as unknown as { __RECORD__: unknown }).__RECORD__ = { error: String(err) };
  out.textContent = `FAILED: ${String(err)}`;
});
