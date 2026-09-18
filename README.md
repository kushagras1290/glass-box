# Glass Box

**A language model that runs entirely in your browser and shows the probability distribution behind
every word it writes. You can fork it: pick a token it didn't choose, and watch the future rewrite.**

No API key, no backend, no server. A real transformer (SmolLM2-360M) runs on your GPU through WebGPU,
so your prompt never leaves the tab.

## What you do

1. **Scroll the story.** Four chapters built from *real recorded outputs* of this model explain what
   it actually does. In chapter IV your scroll position **is** the temperature dial: the same logits
   get re-softmaxed live as you scroll, so the field visibly collapses to one certain token at T = 0
   and blooms into a flat ring of possibilities at T = 1.6.
2. **Load the model**, which happens only when you click. The browser caches it after that.
3. **Generate.** Every token arrives with the top-24 distribution it was sampled from. Confident
   tokens and precarious ones are coloured differently in the timeline.
4. **Fork.** Click any token, pick an alternative, and generation resumes from that point with your
   choice. The abandoned branch is kept.

## Two decisions that make it work

**It doesn't use `generate()`.** A fork *is* a KV-cache rewind, and the visualisation needs the
logits at every step. The library's `generate()` hides both, so Glass Box runs its own decode loop
over raw forward passes, and owns the sampler. That's also why the temperature you see is exactly
the temperature that sampled.

**The KV cache contract was found by instrumenting the library, not by trusting docs.** The ONNX export emits the cache as `present.*` tensors
but only accepts it back as a single nested `past_key_values` object. Passing it any other way is
silently ignored, and the model then produces fluent-looking garbage. The full account is in
[`docs/phase0-results.md`](docs/phase0-results.md).

## Run it

```bash
npm install
npm run dev        # http://localhost:5173
npm test           # 33 unit tests: sampler, branch tree, story director
npm run typecheck
npm run build
```

It needs a WebGPU-capable browser for good speed; without WebGPU it falls back to WebAssembly on
the CPU and says so. The first load downloads the q4 weights, and later loads come from browser
cache.

## Architecture

```
main thread                                   worker
─────────────────────────────────────────     ─────────────────────────────────
scroll surface (the only scroller)            InferenceEngine
 └ sticky stage                                ├ own decode loop, KV threading
    ├ field canvas  ← director (pure)          ├ fork = replay prefix + force token
    ├ chapter layers ← CSS vars from scroll    ├ sampler (pure, unit-tested)
    ├ instrument    ← useInference ◀──────────▶│ runs serialised, never concurrent
    └ constant frame                           └ posts top-k only, never full logits
branch tree (single source of truth)
```

- `src/worker/`: engine, sampler, and the typed message protocol that both sides import
- `src/state/`: the branch tree (forks append and never mutate) plus the React binding
- `src/story/`: the chapter score, the scroll engine, and the director that maps story state to the field
- `src/render/`: the field canvas and the token timeline (DOM, so it stays selectable and readable by screen readers)

The cinematic layer follows a decoupled-scroll architecture: the document never scrolls, a fixed
surface does, and every visual state is derived from scroll position rather than played as a
timeline. Reverse scroll, refresh mid-page and deep links (`#temperature`, `#instrument`)
all reconstruct the exact frame.

## Accessibility

- Every token is a real `<button>`, announced with its probability ("·Paris, 96 percent").
- Tokens you forked are announced as "chosen by you".
- The alternatives are a listbox, and output is mirrored to an `aria-live` region.
- The chapter numerals double as a keyboard table of contents. Hidden chapters are `inert`.
- With `prefers-reduced-motion`, nothing animates but the whole story still reads.

## Honest limits

- **Throughput is below the spec's 12 tok/s target**: about 6.7 tok/s end to end on the test
  machine. See the Phase 0 results.
- The model is 360M parameters, and it will say false things with confidence. The *Confidently
  wrong* prompt is there to show exactly that.
- Story percentages are renormalised over the top 24 candidates, not the full 49,152-token
  vocabulary. The page says so.
- Real-device mobile touch scrolling has not been tested; only emulated mobile has.

## Credits

Built by Kushagra Singh. Model: [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
by Hugging Face. Inference: [transformers.js](https://github.com/huggingface/transformers.js).
