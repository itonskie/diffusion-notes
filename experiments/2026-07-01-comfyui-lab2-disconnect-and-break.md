---
type: experiment
tags:
  - foundations
  - comfyui
  - lab
  - txt2img
  - hands-on
date: 2026-07-01
updated: 2026-07-01
status: exploring
---

# Lab 2 — Disconnect and break the txt2img graph

Lab 1 built the 7-node graph and ran it. You know it works. This lab takes the same graph and asks a different question: **what does each wire actually do?** Instead of explaining it, we'll break it — one wire at a time — and let the error messages (or the broken images) teach you.

Successor to [[2026-07-01-comfyui-lab1-txt2img-from-scratch]]. Reuses the same workflow as the starting state.

**Expected time: ~15 minutes.**

## Hypothesis

The Lab 1 graph has 9 wires. Each one carries *something* that *something else* needs. If we pull one wire and queue the prompt, the graph should break — but the question is **how**. There are three possible failure modes:

1. **Validation error** — ComfyUI refuses to even start the run, pops up a red error before any node executes.
2. **Runtime crash** — the run starts but a node throws mid-execution.
3. **Wrong output** — the run completes cleanly but the image is garbage.

The bet: ComfyUI does aggressive **upfront validation**, so most disconnections will be type (1) — the graph engine knows every required input must be wired before it'll let you run. We'll find out which (if any) slip through to (2) or (3).

Then a stretch test at the end: instead of *unplugging* a wire, we *swap* one — wire the positive prompt into the negative slot and vice versa. The graph stays valid, all inputs are connected. So what happens?

## Setup

**1. Start from the working Lab 1 graph.**

Open ComfyUI at http://127.0.0.1:8188. Load your Lab 1 workflow:

- If it's still on the canvas from Lab 1, you're set.
- Otherwise: top menu → **Workflow → Open** → pick the saved JSON, OR drag any image from `~/Projects/ComfyUI/output/lab1_*.png` onto the canvas (ComfyUI embeds the workflow in PNG metadata — dropping the image restores the graph).

Confirm it still runs end-to-end before you start breaking things. Queue once. Should produce an image.

**2. How to unplug a wire.** Two ways:

- **Click the input dot** (the left-side dot where the wire arrives) **and drag away** into empty canvas → wire detaches.
- OR **click on the wire itself** to select it, then press **Delete** / **Backspace**.

**3. How to re-plug.** Click and drag from the output dot (right side of the source node) back to the input dot of the destination node. Wire snaps back if the types match.

**4. The test cycle for each wire (same every time):**

1. **Predict** — before you unplug, say out loud: which node fails, and what error message?
2. **Unplug** the wire.
3. **Queue Prompt.**
4. **Observe** — does the run start? If yes, does it finish? What does the error message say (or what does the broken image look like)?
5. **Re-plug** before moving to the next test.

We'll test these 6 wires (chosen to span the graph — model, condition, canvas, output, decoder). The remaining 3 wires (negative CLIP, negative CONDITIONING, VAE Decode → Save Image) are symmetric with ones we test, so we skip them.

| Test | Wire to unplug |
|---|---|
| A | **Load Checkpoint** MODEL → **KSampler** model |
| B | **Load Checkpoint** CLIP → positive **CLIP Text Encode** clip |
| C | Positive **CLIP Text Encode** CONDITIONING → **KSampler** positive |
| D | **Empty Latent Image** LATENT → **KSampler** latent_image |
| E | **KSampler** LATENT → **VAE Decode** samples |
| F | **Load Checkpoint** VAE → **VAE Decode** vae |

**5. Then the stretch test (G) — swap, don't unplug.**

Re-plug everything from F. Now:

- Disconnect Positive **CLIP Text Encode** CONDITIONING → **KSampler** positive
- Disconnect Negative **CLIP Text Encode** CONDITIONING → **KSampler** negative
- Re-wire them swapped:
  - Positive **CLIP Text Encode** CONDITIONING → **KSampler** **negative**
  - Negative **CLIP Text Encode** CONDITIONING → **KSampler** **positive**

The graph is fully wired. Every input has something. Queue Prompt. What comes out?

For a clean signal, set prompts so the swap shows up visibly:
- Positive prompt: `a photo of a clean modern kitchen, bright daylight, sharp focus`
- Negative prompt: `blurry, lowres, dark, distorted, ugly`

## Observe

Three places to watch (same as Lab 1, but the error messages matter more this time):

1. **The red error popup in the browser** — ComfyUI shows the validation error here. It usually names the node and the missing input. Read it word for word; that text is the lesson.
2. **The terminal logs** — `got prompt` will still appear, but you'll see a stack trace if a node crashed mid-run, or nothing extra if validation rejected the queue before execution.
3. **The output image** (test G only) — the swap is the only test where you should get a *finished* image. Compare it to the Lab 1 baseline.

## Findings

**Per-wire results.** Overwrite the placeholders in each row.

| Test | Wire | Your prediction | Actual error / outcome | Match? |
|---|---|---|---|---|
| A | MODEL → KSampler | _(your guess: what fails, what message?)_ | _(verbatim or paraphrased)_ | _(y/n)_ |
| B | CLIP → positive encoder | _(your guess)_ | _(actual)_ | _(y/n)_ |
| C | CONDITIONING+ → KSampler positive | _(your guess)_ | _(actual)_ | _(y/n)_ |
| D | Empty Latent → KSampler latent_image | _(your guess)_ | _(actual)_ | _(y/n)_ |
| E | KSampler → VAE Decode samples | _(your guess)_ | _(actual)_ | _(y/n)_ |
| F | VAE → VAE Decode | _(your guess)_ | _(actual)_ | _(y/n)_ |

**Test G — positive/negative swap:**

- Did the graph queue and run without error? _(y/n)_ — _(any caveats)_
- Describe the output image in one sentence: _(your description)_
- How does it compare to a Lab 1 image with the same positive prompt? _(one sentence)_

**Failure-mode tally** (fill after running all tests):

- Validation errors (refused to start): _(count out of 6)_
- Runtime crashes (started but threw mid-run): _(count out of 6)_
- Completed-but-wrong: _(count out of 6)_

**Anything surprising:**

- _(your bullet)_
- _(your bullet)_

**Anything that broke differently than you expected (and what you concluded):**

- _(your bullet)_

## What this lab teaches

- **ComfyUI validates the graph before it runs it.** The graph engine knows which inputs are required and refuses to dispatch nodes whose inputs aren't wired. You'll have seen this six times if all 6 disconnect tests failed at the validation stage. That upfront check is why ComfyUI feels "rigid" the first time you use it — and why broken graphs almost never produce silent garbage.
- **Every input in the minimum graph is required.** There are no optional inputs in the 7 nodes we used. Bigger graphs *do* have optionals (e.g. LoRA loaders take an optional `clip` passthrough), but the txt2img core is fully tight.
- **The error message is the lesson.** ComfyUI tells you exactly which node and which input is missing — "Required input is missing: model" on KSampler is the engine pointing at the exact wire you pulled. Future debugging is faster if you train yourself to read these literally.
- **Type checking happens at *wire time*, validity checking happens at *queue time*.** Try wiring an IMAGE output to a MODEL input — the UI ghosts the wire and refuses (type mismatch, caught instantly). Now disconnect a required input — the canvas accepts that just fine; the engine complains only when you press Queue. Different layers, different checks.
- **Graph validity ≠ graph correctness.** This is what test G shows. Swapping positive and negative still produces a *valid* graph — every input is wired with a matching type. But the image will lean toward the wrong prompt and away from the right one. ComfyUI doesn't know that the input *labeled* "positive" should carry the conditioning you *want*; the labels are conventions for humans. The engine sees two CONDITIONING tensors, full stop. This is the same reason confusing wirings in [[classifier-free-guidance]] math (which prompt gets the positive weight, which the negative) silently produce inverted results in research code too.
- **The dependency graph branches.** The Load Checkpoint node's CLIP output goes to *two* text encoders. Its VAE output goes to one decoder. Outputs can fan out; inputs cannot — each input takes exactly one wire. (This is a directed acyclic graph, like a build system, not a tree.)
- **Test G surfaces the deeper question: what *is* the difference between the positive and negative slots?** Inside KSampler, the answer is [[classifier-free-guidance]] — the sampler runs the denoiser *twice* per step, once with the positive conditioning and once with the negative, then pushes the latent toward positive and away from negative. Swap them and the push reverses. Lab 3 will turn the CFG knob and watch the strength of that push directly.

## See Also

- [[2026-07-01-comfyui-lab1-txt2img-from-scratch]] — the working graph this lab breaks
- [[classifier-free-guidance]] — what the positive/negative split actually does inside KSampler
- [[reference/comfyui]] — graph model, type system, why nodes have two names
- [[u-net]] — the MODEL output the KSampler needs to denoise
- [[latent-diffusion]] — why the latent canvas and the VAE decoder bookend the sampler
- [[foundations]] — Week 1 MOC
