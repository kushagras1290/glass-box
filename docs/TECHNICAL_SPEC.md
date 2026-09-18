# Glass Box — Technical Specification

> **Read [`phase0-results.md`](phase0-results.md) before implementing from this document.**
> This spec was written before the Phase 0 gate ran, and the gate proved parts of it wrong:
>
> - **§4.2 spike code threads the KV cache incorrectly.** The ONNX export emits `present.*`
>   tensors, not `past_key_values`, and the cache must be passed back as **one nested
>   `past_key_values` object** plus explicit `position_ids`. Following §4.2 literally produces
>   fluent-looking garbage. The working implementation is `src/worker/engine.ts`.
> - **Fork steps must be absolute** (spec §6.5 doesn't say). Numbering a fork's tokens from 0
>   makes the tree discard the forced token as stale. Covered by a regression test.
> - The library is `@huggingface/transformers` **v4.3.0**, not v3.
> - Default prompt mode is **raw completion**, not the chat template (§6.3), because an instruct
>   template turns completion prompts into questions.
>
> The rest stands as the design record.

**Status:** Implemented · v1.0 — see the notice above for where the implementation diverged
**Author:** Kushagra Singh
**Audience:** the implementer (you), and any engineer picking the repo up cold
**Scope:** complete build spec, Phase 0 through deploy

---

## 0. How to read this document

Sections 1–2 define *what* is being built. Section 3 defines the architecture. **Section 4 is the
gate** — nothing else starts until the spike passes. Sections 5–12 are the build. Sections 13–17 are
process.

Anything marked **⚠ VERIFY** is an API shape taken from documentation rather than from code I have
executed. Every one of them is resolved by the Phase 0 spike. Do not build on an unverified call.

---

## 1. Product definition

### 1.1 Thesis

A real transformer runs entirely in the visitor's browser. Instead of hiding it behind a chat bubble,
the interface renders the model's next-token probability distribution at every step, and lets the
visitor **fork the generation** by choosing a token the model didn't pick.

The product argument: *an LLM is not an oracle producing text, it is a probability distribution being
repeatedly collapsed — and you can put your hand on the collapse.*

### 1.2 The one interaction that matters

1. Visitor types a prompt.
2. Tokens stream out. Each token carries the top-k distribution it was sampled from.
3. Visitor clicks any generated token.
4. The alternatives fan out, ranked by probability.
5. Visitor picks one. **The timeline forks at that point and regenerates forward.**

Everything else in this document exists to serve that loop. If a feature does not make that loop
clearer or faster, it does not ship in v1.

### 1.3 Success criteria

| # | Criterion | Measure |
|---|---|---|
| S1 | Works with no key, no backend, no sign-up | Static hosting only; zero server-side compute |
| S2 | Time to first token on a warm cache | ≤ 2.0s on a mid-range laptop with WebGPU |
| S3 | Generation throughput | ≥ 12 tok/s with WebGPU on discrete/integrated GPU |
| S4 | Render never blocks | Main thread ≥ 55fps while generating; inference off-thread |
| S5 | Fork latency | ≤ 1.5s from click to first re-generated token |
| S6 | Honest degradation | No WebGPU → WASM path with a visible, explained quality/speed notice |
| S7 | Accessible | Full keyboard path; reduced-motion path; screen-reader-readable transcript |

### 1.4 Non-goals (v1)

- Multi-turn chat history
- Model switching in the UI
- Server-side anything
- Mobile-first experience (mobile gets a reduced, working, honest version — see §9.4)
- Attention-head visualisation (stretch, see §17)
- Fine-tuning, RAG, tools, agents

---

## 2. Experience specification

### 2.1 Chapters

The page is a scroll-driven cinematic sequence built on the `cinematic-web` skill, using the
decoupled scroll surface and composed chapter score from that skill. The subject is abstract, so
procedural/canvas rendering is the *correct* choice here rather than the janky one.

| Chapter | Dwell | State | What the visitor sees |
|---|---|---|---|
| C0 · Cold | 1.0 vh | idle | Title. One sentence. A single dormant point of light. |
| C1 · Load | variable | loading | Model weights streaming in; progress rendered as the point accreting mass. Honest byte counter. |
| C2 · Prompt | 1.4 vh | ready | Input. Two or three suggested prompts as one-click chips. |
| C3 · Shatter | 0.8 vh | tokenizing | The typed string breaks into token cells. Token ids visible. This is the first "oh" moment. |
| C4 · Field | 8.0 vh | generating | **The main stage.** Probability cloud + token timeline. Temperature control. This is the product. |
| C5 · Fork | — | forked | Not a scroll chapter; a state of C4. Timeline branches visibly. |
| C6 · Exhale | 1.6 vh | done | Output as plain readable text, with a "read the path" summary and a share link. |

Chapter lengths are deliberately unequal (skill §3.2). C4 is the centrepiece and gets ~10× the dwell
of the quick beats.

### 2.2 The probability cloud

The core visual. At each decode step the model yields a distribution over the whole vocabulary; we
render only the top-k (k ≤ 40).

- Each candidate token is a cell positioned on a radial field; **radius is inversely proportional to
  probability** — likely tokens sit near the centre, unlikely ones drift out.
- Cell area maps to probability mass. Label = the decoded token string, with whitespace made visible
  (`·` for space, `⏎` for newline) because whitespace tokens are otherwise baffling.
- The sampled token ignites and travels from the cloud into the timeline.
- Temperature changes visibly redistribute mass **before** the next sample: T→0 collapses the field
  to one dominant cell; T→1.5 spreads it wide. The control must feel physical.

### 2.3 The timeline

A horizontal (desktop) / vertical (mobile) run of committed tokens.

- Each token cell shows its own sampled probability as a thin fill bar — a 0.12 token looks visibly
  precarious next to a 0.98 token. This single detail teaches more than a paragraph of copy.
- Hover/focus → its stored top-k fans out.
- Click → fork affordance.
- Forked history is retained as a visible tree: prior branches ghost out rather than being destroyed.

### 2.4 Copy rules

Plain, declarative, no hype. The model is small and will say wrong things — lean into that. A line
like *"360M parameters. It will be confidently wrong. That's the point — watch where the certainty
actually is."* is worth more than any claim of intelligence.

---

## 3. System architecture

### 3.1 Threads

```
┌────────────────────── main thread ───────────────────────┐
│  React shell                                             │
│   ├── scroll surface + chapter score (cinematic-web)     │
│   ├── BranchStore (tree state, single source of truth)   │
│   └── Renderers                                          │
│        ├── CloudCanvas   (2D canvas, rAF)                │
│        └── TimelineDOM   (React, cheap)                  │
└──────────────┬───────────────────────────────────────────┘
               │  structured postMessage (see §7)
┌──────────────▼──────────── worker ───────────────────────┐
│  InferenceEngine                                         │
│   ├── tokenizer  (AutoTokenizer)                         │
│   ├── model      (AutoModelForCausalLM, webgpu|wasm)     │
│   ├── decode loop (own loop — NOT generate())            │
│   ├── KV cache management + fork rewind                  │
│   └── sampler (temperature / top-k / top-p)              │
└──────────────────────────────────────────────────────────┘
```

**Hard rule:** no model object, tensor, or `Float32Array` of vocab size ever crosses to the main
thread. The worker posts only the already-reduced top-k (≤ 40 entries). Vocabularies are ~49k–152k
floats per step; shipping those per token would murder the frame budget.

### 3.2 Why we write our own decode loop

`generate()` is a closed box: it owns the loop, the KV cache and the sampler. We need all three.

1. **Logits per step** — the entire product is the distribution; we must read it every step.
2. **Fork rewind** — forking means re-running from step *i* with a substituted token. That requires
   either truncating the KV cache or re-prefilling. `generate()` exposes neither.
3. **Sampling ownership** — the temperature control must act on logits we hold, so the visual and
   the actual sample are guaranteed to be the same computation, not two implementations that drift.

The cost is that we hand-thread `past_key_values`. That is exactly what Phase 0 de-risks.

### 3.3 Module layout

```
glass-box/
├── index.html
├── public/_headers                 # COOP/COEP (see §13.3)
├── src/
│   ├── main.tsx
│   ├── config.ts                   # validated at startup (§10.4)
│   ├── errors.ts                   # exception hierarchy (§10.1)
│   ├── log.ts                      # structured logger (§10.2)
│   ├── worker/
│   │   ├── inference.worker.ts     # worker entry, message router
│   │   ├── engine.ts               # load, prefill, decode, fork
│   │   ├── sampler.ts              # pure fns: softmax, topK, topP, sample
│   │   └── protocol.ts             # shared message types (imported by both sides)
│   ├── state/
│   │   ├── branchStore.ts          # the tree (§5)
│   │   └── useInference.ts         # React hook wrapping the worker client
│   ├── render/
│   │   ├── CloudCanvas.tsx
│   │   ├── cloudScene.ts           # rAF loop, layout, easing
│   │   └── Timeline.tsx
│   └── ui/  …chapters, controls, shell
└── tests/
```

`protocol.ts` and `sampler.ts` are pure and shared/testable — the sampler must be unit-tested in
Node without a browser.

---

## 4. Phase 0 — the spike (GATE)

**Do not write a line of UI until this prints real numbers.** Budget: one evening.

### 4.1 What it must prove

| ID | Assertion | Fail ⇒ |
|---|---|---|
| A1 | Model loads in-browser on WebGPU | drop to WASM, revise S2/S3 |
| A2 | `await model(inputs)` returns a `logits` tensor with dims `[1, seq, vocab]` | project dead in this form — stop |
| A3 | `logits.data` is a readable `Float32Array` | as A2 |
| A4 | Output contains reusable `past_key_values` | fall back to §6.6 Plan B |
| A5 | Feeding `past_key_values` back produces coherent continuation | as A4 |
| A6 | `apply_chat_template` exists and produces a sane instruct prompt | hand-write the template string |

### 4.2 Spike code

Single file. Serve it (`npx serve .`) — ES module imports will not run from `file://`.

```html
<!doctype html>
<html lang="en">
<head><meta charset="utf-8" /><title>Glass Box · Phase 0</title></head>
<body style="background:#06080b;color:#dfe7ea;font:13px/1.6 ui-monospace,monospace;padding:24px">
<pre id="out">booting…</pre>
<script type="module">
import {
  AutoTokenizer, AutoModelForCausalLM, env,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';

const el = document.getElementById('out');
const log = (...a) => { el.textContent += '\n' + a.join(' '); console.log(...a); };

env.allowLocalModels = false;                     // always fetch from the Hub
const MODEL_ID = 'HuggingFaceTB/SmolLM2-360M-Instruct';   // ⚠ VERIFY onnx weights exist

function topK(arr, k) {
  // arr: Float32Array of logits. Returns [{id, logit}] descending, without sorting 150k items.
  const best = [];
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (best.length < k) { best.push({ id: i, logit: v }); best.sort((a, b) => a.logit - b.logit); }
    else if (v > best[0].logit) { best[0] = { id: i, logit: v }; best.sort((a, b) => a.logit - b.logit); }
  }
  return best.reverse();
}
function softmax(items, T = 1) {
  const m = Math.max(...items.map(i => i.logit));
  const ex = items.map(i => Math.exp((i.logit - m) / T));
  const s = ex.reduce((a, b) => a + b, 0);
  return items.map((it, i) => ({ ...it, p: ex[i] / s }));
}

try {
  log('webgpu:', !!navigator.gpu);
  const device = navigator.gpu ? 'webgpu' : 'wasm';

  const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
  log('A6 apply_chat_template:', typeof tokenizer.apply_chat_template);

  const model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
    dtype: 'q4',
    device,
    progress_callback: (p) => { if (p.status === 'progress' && p.progress) log(`  ${p.file} ${p.progress.toFixed(0)}%`); },
  });
  log('A1 OK — model loaded on', device);

  const prompt = tokenizer.apply_chat_template(
    [{ role: 'user', content: 'Name three colours.' }],
    { tokenize: false, add_generation_prompt: true },
  );
  const inputs = await tokenizer(prompt);
  log('prompt tokens:', inputs.input_ids.dims.join('x'));

  const t0 = performance.now();
  const out = await model({ ...inputs });
  log('prefill ms:', (performance.now() - t0).toFixed(0));
  log('output keys:', Object.keys(out).join(', '));

  // A2 / A3
  const logits = out.logits;
  log('A2 logits dims:', JSON.stringify(logits.dims));
  const [, seq, vocab] = logits.dims;
  const data = logits.data;
  log('A3 data type:', data.constructor.name, 'len:', data.length);

  const last = data.subarray((seq - 1) * vocab, seq * vocab);
  const cands = softmax(topK(last, 5));
  log('\nTOP-5 NEXT TOKENS:');
  for (const c of cands) {
    log(`  ${(c.p * 100).toFixed(1).padStart(5)}%  id=${String(c.id).padStart(6)}  ${JSON.stringify(tokenizer.decode([c.id]))}`);
  }

  // A4 / A5 — KV reuse
  const pkv = out.past_key_values;
  log('\nA4 past_key_values present:', !!pkv);
  if (pkv) {
    let generated = [];
    let past = pkv, nextId = cands[0].id;
    const promptLen = inputs.input_ids.dims[1];
    for (let step = 0; step < 12; step++) {
      generated.push(nextId);
      const step_ids   = new Tensor('int64', BigInt64Array.from([BigInt(nextId)]), [1, 1]);
      const step_mask  = new Tensor('int64',
        BigInt64Array.from({ length: promptLen + generated.length }, () => 1n),
        [1, promptLen + generated.length]);
      const o = await model({ input_ids: step_ids, attention_mask: step_mask, past_key_values: past });
      past = o.past_key_values;
      const v = o.logits.dims[2];
      const lastStep = o.logits.data.subarray((o.logits.dims[1] - 1) * v, o.logits.dims[1] * v);
      nextId = topK(lastStep, 1)[0].id;
    }
    log('A5 continuation:', JSON.stringify(tokenizer.decode(generated)));
  }
  log('\nSPIKE COMPLETE');
} catch (err) {
  log('\nFAILED:', err?.message ?? String(err));
  console.error(err);
}
</script>
</body>
</html>
```

`Tensor` must be imported alongside the rest (`import { Tensor } from '…/transformers@3'`) — add it
when you run this; it is called in the A4 block above.

### 4.3 Reading the result

- **A5 prints coherent text** → build the full engine (§6). Green light.
- **A4/A5 fail** → adopt Plan B (§6.6): stream with `generate()`, compute distributions lazily on
  demand. Product survives; the cloud becomes on-hover rather than per-step live.
- **A2/A3 fail** → stop. The concept requires logits. Do not attempt a workaround; report it and
  pick a different project.

Record the actual numbers (load seconds, prefill ms, per-step ms, model MB) in
`docs/phase0-results.md`. Every estimate later in this document should be replaced by them.

---

## 5. Data model

The branch tree is the single source of truth. Everything rendered derives from it.

```ts
/** A candidate token from one decode step's top-k. */
export interface Candidate {
  readonly tokenId: number;
  readonly text: string;      // decoded, whitespace already made visible
  readonly p: number;         // post-temperature probability, 0..1
  readonly logit: number;     // raw, kept for display/debug
}

/** One committed token in the tree. */
export interface BranchNode {
  readonly id: string;              // nanoid
  readonly parentId: string | null;
  readonly childIds: string[];      // >1 ⇒ a fork happened here
  readonly step: number;            // 0-based decode index
  readonly tokenId: number;
  readonly text: string;
  readonly p: number;               // probability of THIS token when sampled
  readonly candidates: Candidate[]; // the top-k it was drawn from
  readonly chosenByUser: boolean;   // true if forked here by hand
  readonly temperature: number;     // T in force at this step
}

export interface BranchTree {
  readonly nodes: Record<string, BranchNode>;
  readonly rootIds: string[];
  readonly activePath: string[];    // root → tip, the currently displayed run
  readonly promptTokenIds: number[];
}
```

Invariants (assert in dev):

1. `activePath[0]` ∈ `rootIds`.
2. For consecutive `activePath` entries, `nodes[b].parentId === a`.
3. `nodes[n].step === activePath.indexOf(n)` for every node on the active path.
4. Every `candidates` array is sorted descending by `p` and has length ≤ `config.topK`.

**Forking** never mutates or deletes: it appends a new child to an existing node and re-points
`activePath`. Old branches stay in `nodes` and render as ghosts. This makes the tree the visitor's
history of interrogation, which is the point of the product.

---

## 6. Inference engine

### 6.1 Model selection

| Model | Params | Why |
|---|---|---|
| **`HuggingFaceTB/SmolLM2-360M-Instruct`** | 360M | **Default.** Smallest download → best first-visit experience. Vocab ~49k keeps top-k cheap. |
| `onnx-community/Qwen2.5-0.5B-Instruct` | 500M | Optional "bigger" toggle post-v1. Vocab ~152k — verify top-k cost. |

⚠ VERIFY in Phase 0: ONNX q4 weights exist for the chosen id, and the real download size. Do not
quote a size in the UI you have not measured.

Precision `q4` — the quality loss is irrelevant here because *being visibly imperfect is the
product*. Do not ship fp32; the download cost is not justifiable.

### 6.2 Load sequence

1. Capability probe: `navigator.gpu` → `'webgpu'`, else `'wasm'` with a user-visible notice.
2. `AutoTokenizer.from_pretrained` then `AutoModelForCausalLM.from_pretrained` with
   `progress_callback` forwarding byte progress to the UI (chapter C1 renders it).
3. **Timeout**: wrap loading in an abortable race (`config.modelLoadTimeoutMs`, default 120_000).
   On timeout throw `ModelLoadError`. Never leave the visitor on an unbounded spinner.
4. Post `READY` with resolved `{ device, modelId, vocabSize, contextLength }`.

Weights are cached by the library in browser storage, so repeat visits skip the download. Say so in
the UI on first load — it converts an apparent cost into an apparent feature.

### 6.3 Prefill

```ts
const prompt = tokenizer.apply_chat_template(
  [{ role: 'user', content: userText }],
  { tokenize: false, add_generation_prompt: true },
);
const inputs = await tokenizer(prompt);
const out = await model({ ...inputs });
```

Keep `promptTokenIds` for the shatter chapter (C3) and for fork replay.

### 6.4 Decode loop

Pseudocode of the real loop; it runs in the worker and yields once per token.

```
past      ← prefill.past_key_values
logits    ← last row of prefill.logits
generated ← []

loop until stop:
    cands   ← topK(logits, config.topK)
    scaled  ← softmax(cands, temperature)          // temperature read fresh each step
    chosen  ← forcedToken ?? sample(scaled, topP)  // forcedToken set only on a fork's first step
    emit TOKEN { step, chosen, candidates: scaled }

    if chosen.tokenId ∈ eosTokenIds:      emit DONE('eos');    break
    if generated.length ≥ maxNewTokens:   emit DONE('length'); break
    if cancelled:                         emit DONE('cancelled'); break

    generated.push(chosen)
    out    ← model({ input_ids: [[chosen]], attention_mask: ones(promptLen+generated.length), past_key_values: past })
    past   ← out.past_key_values
    logits ← last row of out.logits
```

Notes that matter:

- **Temperature is read per step, not captured at start.** The visitor must be able to turn the dial
  mid-generation and see the field change. Store the T in force on each node.
- `attention_mask` must be full length (prompt + generated so far), while `input_ids` is only the
  single new token. Getting this wrong produces fluent-but-subtly-wrong output — the worst failure
  mode because it looks like it works.
- Yield to the event loop between steps (`await Promise.resolve()` / a 0ms task) so `CANCEL` messages
  are processed promptly.

### 6.5 Forking

To fork at node *i* with candidate *c*:

1. Truncate the conceptual sequence to `promptTokenIds + path[0..i-1]`, then append `c.tokenId`.
2. **v1 strategy — re-prefill.** Run one batched forward over that whole sequence. A single prefill
   of a few hundred tokens is fast and, critically, *obviously correct*.
3. Continue the normal decode loop from the returned `past_key_values`.
4. Append the new node as a child of `nodes[path[i-1]]`, set `chosenByUser: true`, re-point
   `activePath`.

Optimisation for later (only if measurement says S5 fails): cache a `past_key_values` snapshot every
N steps and resume from the nearest one. Do not build this first — KV snapshot retention is a memory
liability and premature here.

### 6.6 Plan B — if `past_key_values` cannot be threaded (A4/A5 fail)

Architecture degrades, product survives:

1. Generate the run with `generate()` + a `TextStreamer` for live text.
2. When the visitor **hovers or focuses** a token, run a single forward pass over the prefix up to
   that token and compute its top-k on demand. Cache per node.
3. Forking = re-prompt with the forced prefix and generate again.

Cost: the cloud is no longer live-per-step, it is inspect-on-demand. Honestly, this is a defensible
product in its own right — it just tells a slightly quieter story. Label the chapter copy accordingly
rather than implying live introspection you are not doing.

### 6.7 Sampler (pure, unit-tested)

```ts
export function softmaxOver(cands: RawCandidate[], temperature: number): Candidate[];
export function applyTopP(cands: Candidate[], topP: number): Candidate[];   // renormalises
export function sampleFrom(cands: Candidate[], rng: () => number): Candidate;
export function topKFromLogits(logits: Float32Array, k: number): RawCandidate[];
```

Requirements:

- `topKFromLogits` must **not** sort the full vocabulary. Use a bounded min-heap / insertion scan
  (O(V·log k)). Sorting 152k floats per token will cost you the frame budget.
- Softmax must subtract the max logit before `exp` — without it, fp32 overflows and you get `NaN`
  probabilities that render as an empty cloud and look like a rendering bug for an hour.
- `temperature === 0` is a special case: return argmax with p=1.0. Do not divide by zero.
- Inject `rng` so tests are deterministic and so a `?seed=` URL param can make a share link
  reproducible.

---

## 7. Worker protocol

Shared, exhaustive, discriminated unions in `worker/protocol.ts`. Both sides import this file; no
stringly-typed messages anywhere.

```ts
export type MainToWorker =
  | { type: 'LOAD';     modelId: string; device: 'webgpu' | 'wasm' | 'auto' }
  | { type: 'GENERATE'; runId: string; prompt: string; params: GenParams }
  | { type: 'FORK';     runId: string; prefixTokenIds: number[]; forcedTokenId: number; params: GenParams }
  | { type: 'SET_TEMPERATURE'; temperature: number }   // applies to the in-flight run
  | { type: 'CANCEL';   runId: string }
  | { type: 'DISPOSE' };

export type WorkerToMain =
  | { type: 'LOAD_PROGRESS'; file: string; loaded: number; total: number }
  | { type: 'READY';   modelId: string; device: 'webgpu' | 'wasm'; vocabSize: number; contextLength: number }
  | { type: 'PROMPT_TOKENS'; runId: string; tokenIds: number[]; texts: string[] }
  | { type: 'TOKEN';   runId: string; step: number; chosen: Candidate; candidates: Candidate[]; msSinceLast: number }
  | { type: 'DONE';    runId: string; reason: 'eos' | 'length' | 'cancelled'; totalMs: number; tokensPerSecond: number }
  | { type: 'ERROR';   runId: string | null; code: ErrorCode; message: string; recoverable: boolean };

export interface GenParams {
  readonly temperature: number;   // 0..2
  readonly topK: number;          // ≤ 40, bounds the payload
  readonly topP: number;          // 0..1
  readonly maxNewTokens: number;  // default 160
  readonly seed?: number;
}
```

Rules:

- `runId` on every run-scoped message. A late `TOKEN` from a cancelled run must be droppable by id —
  otherwise a fast forker gets two runs interleaving into one timeline.
- Worker never posts a payload larger than `topK` entries.
- Worker never throws across the boundary; it catches and posts `ERROR` with a code from §10.1.

---

## 8. Rendering

### 8.1 Split

| Layer | Tech | Why |
|---|---|---|
| Probability cloud | **Canvas 2D** + rAF | ≤ 40 animated cells. 2D is more than sufficient and avoids a shader pipeline you'd have to debug. |
| Timeline | **React DOM** | Text, focusable, selectable, screen-reader-legible. Do not draw text you want readable into canvas. |
| Ambient field | Canvas 2D | Reuse the particle-field approach from the portfolio. |

Deliberate anti-pattern to avoid: do **not** render the token timeline in canvas. It is text; it must
be selectable, copyable and reachable by assistive tech.

### 8.2 Cloud layout

```
for each candidate c (rank r, probability p):
    targetRadius ← R_min + (1 - p^0.45) * (R_max - R_min)
    targetAngle  ← goldenAngle * r + slowDrift(t)
    targetSize   ← S_min + sqrt(p) * (S_max - S_min)
```

- `p^0.45` compresses the dynamic range so a 0.97 token doesn't sit alone at the centre with 39 cells
  smeared at the rim.
- Golden-angle placement avoids the spokes you get from even angular division.
- Cells **ease toward** targets (`1 - exp(-dt/τ)`, τ ≈ 140ms) rather than snapping, so a temperature
  change reads as the field breathing.
- Frame-rate independence is mandatory — use elapsed ms, never per-frame constants (skill §3.11).

### 8.3 Frame budget

Per frame, with generation running:

| Work | Budget |
|---|---|
| Cloud draw (≤40 cells + trails) | ≤ 4ms |
| Ambient field | ≤ 2ms |
| React commit (timeline append) | ≤ 3ms |
| Slack | ≥ 7ms |

Enforcement: cap DPR at 1.5. `contain: layout paint style` on the timeline. Append tokens to the
timeline via a keyed list — never re-map the full array into new objects each token. Pause all rAF on
`document.hidden`.

### 8.4 Reduced motion

`prefers-reduced-motion: reduce` ⇒ cells jump straight to layout positions, no drift, no trails, no
ignite animation. The distribution is still fully visible and the product still teaches — it just
stops moving. Test this path; it is not a second-class experience.

---

## 9. UI composition

### 9.1 Controls

- **Temperature**: the only always-visible control. Slider 0–2, default 0.8, live during generation.
- **Top-p** and **max tokens**: behind a "parameters" disclosure. Not front and centre in v1.
- **Stop**: visible whenever a run is in flight.
- **Reset**: clears the tree, keeps the loaded model.

### 9.2 Prompt chips

Three fixed prompts chosen to *demonstrate distribution shape*, not to show off the model:

1. A forced-continuation with an obvious answer (`"The capital of France is"`) → one spike, near 1.0.
   Shows confidence.
2. An open creative line (`"Write the first sentence of a story about rain"`) → a wide flat field.
   Shows genuine uncertainty.
3. A factual question the 360M model will get wrong → confident *and* wrong. This is the most
   valuable chip in the product; it teaches that confidence ≠ correctness.

### 9.3 Share links

Encode `{prompt, seed, temperature, forkPath}` into the URL (`replaceState`, per the cinematic-web
routing pattern). A shared link reproduces the exact run including the visitor's forks. This is the
single highest-leverage growth feature and it costs almost nothing.

### 9.4 Mobile

Honest degradation, decided by capability not width:

- No WebGPU, or `deviceMemory < 4` → do **not** auto-download weights. Show a static, pre-recorded
  walkthrough of a real session plus an explicit "run it anyway" button.
- With WebGPU → vertical timeline, cloud reduced to top-12, same logic.
- Never silently start a multi-hundred-MB download on a phone.

---

## 10. Errors, logging, config

### 10.1 Exception hierarchy

```ts
export type ErrorCode =
  | 'UNSUPPORTED_DEVICE' | 'MODEL_LOAD' | 'MODEL_TIMEOUT' | 'OUT_OF_MEMORY'
  | 'TOKENIZE' | 'INFERENCE' | 'PROTOCOL' | 'CONFIG';

export class GlassBoxError extends Error {
  constructor(message: string, readonly code: ErrorCode, readonly recoverable: boolean,
              readonly cause?: unknown) { super(message); this.name = new.target.name; }
}
export class UnsupportedDeviceError extends GlassBoxError { /* code 'UNSUPPORTED_DEVICE' */ }
export class ModelLoadError        extends GlassBoxError { /* 'MODEL_LOAD',  recoverable: true  */ }
export class ModelTimeoutError     extends GlassBoxError { /* 'MODEL_TIMEOUT', recoverable: true */ }
export class OutOfMemoryError      extends GlassBoxError { /* 'OUT_OF_MEMORY', recoverable: false */ }
export class InferenceError        extends GlassBoxError { /* 'INFERENCE', recoverable: true */ }
export class ConfigError           extends GlassBoxError { /* 'CONFIG', recoverable: false */ }
```

No bare `throw new Error` anywhere in `src/`. Every catch site decides using `code` + `recoverable`,
never by string matching.

### 10.2 Structured logging

```ts
type Level = 'debug' | 'info' | 'warn' | 'error';
export function log(level: Level, event: string, fields: Record<string, unknown> = {}): void {
  const line = { ts: new Date().toISOString(), level, event, ...fields };
  if (import.meta.env.DEV) console[level === 'debug' ? 'log' : level](line);
  else if (level === 'error' || level === 'warn') console[level](JSON.stringify(line));
}
```

Log events, not prose: `model.load.start`, `model.load.done`, `decode.step`, `fork.begin`,
`worker.error`. `decode.step` at `debug` only — it fires 160 times a run.

### 10.3 User-facing failure states

Every failure gets a designed state, never a blank screen:

| Condition | UI |
|---|---|
| No WebGPU | Banner: what will happen (slower, WASM), with a proceed button |
| No WebGPU **and** no WASM/SIMD | Pre-recorded walkthrough + explanation. Do not pretend. |
| Load failed / timed out | Retry with byte progress preserved |
| OOM mid-run | Stop, keep the tree, suggest fewer max tokens, offer reload |

### 10.4 Config validated at startup

```ts
export interface AppConfig {
  readonly modelId: string;
  readonly topK: number;            // 1..40
  readonly maxNewTokens: number;    // 1..512
  readonly defaultTemperature: number;
  readonly modelLoadTimeoutMs: number;
  readonly dprCap: number;
}
export function loadConfig(env: ImportMetaEnv): AppConfig;  // throws ConfigError on any invalid field
```

Call it once in `main.tsx` before render. Crash loudly on a bad value rather than degrading quietly.

---

## 11. Performance

### 11.1 Instrumentation (build it in from day one)

Collect and expose in a dev-only HUD: model load ms, prefill ms, per-step ms (p50/p95), tokens/sec,
frame time p95, `performance.memory` where available. You cannot tune S2–S5 without these, and
retrofitting them is miserable.

### 11.2 Known hazards

| Hazard | Mitigation |
|---|---|
| Top-k over a 152k vocab each step | Bounded heap; prefer the 49k-vocab model |
| Posting full logits to main thread | Forbidden by §3.1 — top-k only |
| React re-render per token | Keyed append, memoised nodes, no full-array remap |
| KV cache growth | Cap `maxNewTokens`; release `past` on run end |
| Canvas at DPR 3 on a retina laptop | Cap DPR 1.5 |
| Ghost branches accumulating | Cap retained branches (e.g. 12), prune oldest |

---

## 12. Accessibility

Non-negotiable, and a genuine differentiator in this category:

- **Transcript region**: an `aria-live="polite"` element carrying the committed text so a screen
  reader follows generation without the canvas.
- **Timeline tokens are real `<button>`s** — tabbable, Enter/Space opens alternatives, arrow keys move
  between tokens, Escape closes the fan-out.
- **Alternatives** are a proper listbox with `aria-selected`; each option labelled
  `"'Paris', 97 percent"` — the probability must be *spoken*, not only drawn.
- Canvas is `aria-hidden="true"` throughout; it is decoration over an accessible data structure.
- Contrast ≥ 4.5:1 for all token text. Probability must never be encoded by colour alone — always
  colour **and** the fill bar **and** a number.
- Full keyboard path from prompt → generate → fork → share, verified manually.

---

## 13. Build & deploy

### 13.1 Stack

Vite + React + TypeScript (strict). `@huggingface/transformers` v3.x — pin the exact minor and record
it in `docs/phase0-results.md`.

### 13.2 Worker

```ts
new Worker(new URL('./worker/inference.worker.ts', import.meta.url), { type: 'module' });
```

Vite handles module-worker bundling natively. Do not inline the worker.

### 13.3 Headers

WebGPU needs no special headers. **Multi-threaded WASM fallback needs cross-origin isolation.**
`public/_headers` (Cloudflare Pages):

```
/*
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
```

⚠ Enabling COEP can block cross-origin resources — including CDN model fetches — unless they serve
CORS/CORP headers. Verify Hub asset fetching still works with isolation on; if it does not, ship
single-threaded WASM and drop the headers. Test this explicitly, it is a classic late-stage ambush.

### 13.4 Hosting

Cloudflare Pages, static. No functions, no env secrets, nothing to rotate. Set a long cache on
hashed assets. Weights come from the HF CDN and are cached client-side.

---

## 14. Testing

| Level | Target | Tool |
|---|---|---|
| Unit | `sampler.ts` — softmax stability, top-k correctness, T=0 argmax, top-p renormalisation, seeded determinism | Vitest (Node) |
| Unit | `branchStore` — fork appends without mutation, invariants 1–4 hold | Vitest |
| Contract | Every `MainToWorker` handled; unknown type → `PROTOCOL` error | Vitest w/ mocked worker |
| Integration | Load → generate → fork → cancel against the real model | Playwright, generous timeout, tagged `@slow` |
| Visual | Chapter frames at C0/C3/C4/C6, desktop + mobile, light of reduced-motion | Playwright screenshots |
| A11y | Keyboard-only run; axe pass on each chapter | Playwright + axe |

Explicit numeric tests worth writing because they catch real bugs:

- softmax of `[1000, 1001]` must not produce `NaN`.
- top-k of a 152k array returns exactly k, correctly ordered.
- `temperature: 0` twice in a row yields identical output.
- Same `seed` + same prompt ⇒ identical token ids.

---

## 15. Milestones

| M | Deliverable | Gate |
|---|---|---|
| **M0** | Phase 0 spike | A1–A6 recorded in `docs/phase0-results.md` |
| **M1** | Worker + engine + sampler, no UI. Generation logged to console. | 160 tokens, correct text, measured tok/s |
| **M2** | Timeline + probability numbers, plain unstyled DOM | Distribution visibly correct; a11y tree sane |
| **M3** | Fork works end to end | S5 met |
| **M4** | Cloud canvas + temperature control | S4 met at 60fps |
| **M5** | Cinematic chapters, copy, share links | Visual pass at every chapter frame |
| **M6** | Failure states, mobile, a11y, deploy | S1–S7 all verified |

M1–M2 are the real work. M4 is the one that photographs well — resist doing it first; a beautiful
cloud over a wrong distribution is worse than useless.

---

## 16. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| `past_key_values` not threadable | High | Phase 0 A4/A5 → Plan B §6.6 |
| Model download too large for casual visitors | High | 360M q4; measure; never auto-download on constrained devices |
| WebGPU absent on reviewer's machine | Medium | WASM path + pre-recorded walkthrough |
| `attention_mask` mistake → subtly wrong output | Medium | Assert length each step; compare 12 greedy tokens against a Python reference once |
| COEP breaks Hub fetches | Medium | §13.3 test; drop threads if needed |
| Scope creep (chat, models, RAG) | High | §1.4 is binding |

---

## 17. Stretch, after v1 ships

- **Attention heads.** Only if ONNX exports expose them; most do not. Investigate as a separate
  spike, never as a v1 promise.
- Embedding-space view of the top-k (2D projection of candidate token embeddings).
- A/B the same prompt at two temperatures side by side.
- Larger model toggle with an explicit download-cost dialogue.

---

## 18. Open questions

1. Exact ONNX q4 artefact + size for the chosen model — resolve in Phase 0.
2. Does `apply_chat_template` exist on the tokenizer for that model, and what does it emit?
3. Does COEP isolation break Hub asset fetching on Cloudflare Pages?
4. Real p95 per-step latency on a mid-range integrated GPU — decides whether §6.5 needs the KV
   snapshot optimisation.

Answer 1–3 in Phase 0. Answer 4 at M1.
