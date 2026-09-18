# Phase 0 results

The gate from `TECHNICAL_SPEC.md` §4, run on 2026-09-18 in Chromium with WebGPU.
Everything below was measured, not estimated.

## Environment

| | |
|---|---|
| Library | `@huggingface/transformers` **4.3.0** (the spec assumed v3) |
| Model | `HuggingFaceTB/SmolLM2-360M-Instruct`, `q4`, file `onnx/model_q4.onnx` |
| Device | WebGPU |
| Vocabulary | 49,152 tokens |
| Layers | 32 (64 KV-cache tensors) |

## Assertions

| ID | Assertion | Result |
|---|---|---|
| A1 | Model loads in-browser on WebGPU | ✅ |
| A2 | Forward pass returns `logits` with dims `[1, seq, vocab]` | ✅ `[1, 34, 49152]` |
| A3 | `logits.data` is a readable `Float32Array` | ✅ |
| A4 | Output carries a reusable KV cache | ✅ — see finding 1 |
| A5 | Feeding the cache back yields coherent continuation | ✅ after fix — see finding 2 |
| A6 | `apply_chat_template` exists | ✅ |

## Findings that contradicted the spec

**1. There is no `past_key_values` output.** The ONNX export emits the cache as flat
`present.<layer>.key` / `present.<layer>.value` tensors. The spec's §4.2 spike code looks for
`out.past_key_values`, finds nothing, and reports A4 as failed. The capability exists; the name differs.

**2. The cache must be passed as ONE nested object.** Remapping `present.*` to `past_key_values.*`
and spreading those names at the top level of the next call is *silently ignored*: the model then
runs with an empty cache while being fed a single token, and produces fluent-looking garbage:

```
manual loop (wrong):  "1.  'cause to  self  self  self  self  self"
generate() reference: "1. Red\n2. Blue\n3. Green"
```

Instrumenting `model.forward` during a `generate()` call showed the library passes
`past_key_values` as a single object plus explicit `position_ids`. Matching that exactly made the
manual loop byte-identical to the reference. This is the "fluent but wrong" failure mode the spec
warned about in §6.4 — it looked like it worked.

**3. Instruct models answer; they don't continue.** Wrapping `"The capital of France is"` in the chat
template produced an assistant self-description instead of a completion, which destroys the
one-spike demonstration. The product therefore defaults to raw completion, with chat as an option.

## Measurements

| Metric | Value |
|---|---|
| First load (cold download) | 141.1 s |
| Warm load (browser cache) | 3.5 s |
| Prefill, 34 tokens | ~2.4–3.1 s |
| Decode step, p50 | 89 ms (≈ 11 tok/s raw) |
| Decode step, p95 | 1236 ms (first-step shader compilation) |
| End-to-end in the app | ~6.7 tok/s |

**Against the spec's targets:** S3 asked for ≥ 12 tok/s. The raw decode step alone runs at about
11 tok/s, and the full app path (top-k, decode, postMessage, React) runs at about 6.7 tok/s. That is
**below target**. It's usable for the product, since watching the distribution form is the point,
but it's not met. The obvious next levers are a smaller top-k payload per step, batching React
updates per frame, and trying `q4f16`.

## Real distribution recorded for the story chapters

`"The capital of France is"` → `·Paris` 88.9% at T = 1, renormalised over the top 24.
Full data and provenance: `src/data/recorded.json`, produced by `src/tools/record.ts`.
