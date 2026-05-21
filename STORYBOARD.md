# Animation storyboard

This is the shot-by-shot plan for the animation that visualizes
[`THEORY.md`](THEORY.md). It is the spec the Manim code implements.

The film tells the story in the order the viewer's understanding should build:

1. a network *trains* — its weights change;
2. those weights are a **point in weight space**, and the init is a **ball** (the prior);
3. training **drags the point away** from that ball;
4. drop the network — bring in the **loss landscape** and the **ε budget**;
5. replace the single point with a **posterior ball $Q$** grown to the budget;
6. read off **KL** = how far + how much $Q$ shifted and shrank from $P$;
7. **PAC-Bayes payoff**: low KL ⇒ tighter generalization bound ⇒ flat minima win.

## Layout & conventions

- **Background:** white. **Text/axes:** black / grey. Consistent with the
  existing scene.
- **Split screen:** LEFT panel = the "concrete" view (network, then loss curve).
  RIGHT panel = weight space in 3D (balls and a moving dot).
- **Color key (kept consistent across the whole film):**
  - Prior $P$ — **blue**, big and translucent.
  - Posterior $Q$ — **maroon** while generic; **green** for the *flat* well,
    **red** for the *sharp* well.
  - Budget $\varepsilon$ — **gold**.
  - Samples that break the budget — **red**.
  - KL penalty bar — **purple**.
- **Captions:** one short line at the top, swapped beat to beat (the existing
  `say()` helper).
- **Render budget:** 3D `Surface`s are expensive under camera motion (see the
  project memory note). Keep sphere `resolution` ~18–24, prefer `FadeIn` over
  `Create`, and avoid `move_camera`/ambient rotation unless parallax is
  essential. Validate every act with `manim -s` (single frame) before a full
  video render.

---

## ACT 1 — A network learns (LEFT panel)

**Goal:** ground the abstraction. "Weights" are the numbers on the edges, and
training changes them.

- **1a.** Fade in a small feed-forward net: ~3 input → 4 hidden → 2 output
  nodes (black circles), fully connected by edges. Edge **thickness ∝ |weight|**,
  edge **color = sign** (blue +, red −). Caption: *"A neural network: every edge
  carries a weight."*
- **1b.** "Train": animate the edge weights changing — thicknesses/colors
  shifting over ~2 s, as if descending a loss. A small `Loss ↓` ticker or a
  shrinking number in the corner. Caption: *"Training adjusts the weights to fit
  the data."*
- **1c.** Freeze. Caption: *"All those weights together are ONE point in a huge
  'weight space'."* This sets up the cut to the right panel.

*Implementation:* `nodes = VGroup(Dot...)`, `edges = VGroup(Line...)`, weights
stored in a numpy array; animate with `edge.animate.set_stroke(width=..., color=...)`.
2-D, added with `add_fixed_in_frame_mobjects` so the 3D camera does not distort it.

---

## ACT 2 — Weight space and the prior ball (RIGHT panel)

**Goal:** introduce the 3D weight space, the prior $P$ as a ball, and the dot.

- **2a.** Bring in a faint 3D axis box on the right (`ThreeDAxes`). Caption:
  *"Weight space — collapsed to 3D so we can see it."*
- **2b.** A connector animation: a copy of the whole network on the left
  collapses / flies into a **single dot** at the origin of the box. Caption:
  *"This dot = the current weights."*
- **2c.** Fade in the **prior ball $P$**: a big translucent **blue sphere**
  centered at the origin. Label it the "init distribution". A thin **inner
  shell** marks the ~60% mass region (for a 3-D Gaussian, ~1.72σ — the 0.6
  quantile of $\chi_3$) so "the 60% ball" the user mentioned is literally drawn.
  Caption: *"Before any
  data, the weights live in this broad 'prior' ball $P$ (the initialization)."*
  - The dot sits at the **center** of $P$ at init.

*Implementation:* reuse `make_ball(mu, sigma, color, opacity)` and
`ball_center()` from the current file. The dot is a small `Sphere` or `Dot3D`.
The 60% shell is a second `Sphere` at radius `1.72 * sigma_P * RAD_SCALE`,
stroke only.

---

## ACT 3 — Training drags the weight away from the prior (RIGHT panel)

**Goal:** show the *mean-shift* — the heart of the first KL term.

- **3a.** As Act 1's "training" replays (or in sync), the **dot travels** from
  the origin out toward a target point (the trained weights), tracing a faint
  path. Caption: *"Training pulls the weights away from where they started."*
- **3b.** A dashed segment from the prior center $\mu_P$ to the dot $\mu_Q$
  appears, labeled $\|\mu_Q-\mu_P\|$. Caption: *"How far it moved is the first
  part of the story."*
- **3c.** Fade the LEFT network out entirely. Caption: *"We're done with the
  network — from here it's all weight space."* (User's explicit beat: "then we
  remove the neural network on the left side.")

*Implementation:* `MoveAlongPath` or `dot.animate.move_to(ball_center(mu_q))`
with a `TracedPath`/faint `Line`. The dashed mean-shift line is a `DashedLine`
in 3D (or fixed-in-frame for legibility).

---

## ACT 4 — The loss landscape and the ε budget (LEFT panel returns)

**Goal:** bring back loss, now as the thing that constrains the posterior.

- **4a.** On the now-empty LEFT, fade in the **loss curve** $L(w)$ over a 1-D
  weight slice — the existing two-well landscape (a WIDE/flat well and a
  NARROW/sharp well, equal depth). Caption: *"Back to the loss — along one
  weight direction. Two valleys: one flat, one sharp, equally deep."*
- **4b.** Draw the **budget line** $z=\varepsilon$ (gold dashed) and shade the
  **feasible intervals** $\{L\le\varepsilon\}$ under each well — wide under the
  flat well, narrow under the sharp one. Caption: *"We'll accept any weights with
  loss ≤ ε. That feasible region is WIDE in the flat valley, NARROW in the sharp
  one."*
- **4c.** (Bridge) note that the dot from Act 3 sits in one of these valleys.

*Implementation:* this is exactly `make_loss_plot()` from the current file.

---

## ACT 5 — From a point to a posterior; grow $Q$ to the budget

**Goal:** the constrained-optimization picture — the visual core.

For **each well** (sharp first, then flat), run the `run_well` beat:

- **5a.** Replace the single dot with a **posterior ball $Q$** that starts *as
  wide as the prior* $P$, centered in the well. On the LEFT, draw the matching
  density **bell** on the loss curve, and scatter **sampled weights** on the
  curve. Samples above $\varepsilon$ turn **red**. Caption: *"Start with a $Q$
  as broad as the prior — but it spills over the budget (red)."*
- **5b.** **Shrink** $Q$ (ball radius and bell width together) until its samples
  fall back under $\varepsilon$ — i.e. $Q$ just fits inside the feasible region.
  The live $D_{KL}(Q\|P)$ readout updates. Caption (sharp): *"Shrink to fit →
  BIG shrink → large KL."* Caption (flat): *"Shrink to fit → tiny shrink → small
  KL."*
- **5c.** Keep the sharp result on screen at low opacity while the flat one runs,
  so the **size contrast** of the two final $Q$ balls is directly comparable.

*Implementation:* exactly the existing `run_well(...)` helper, `make_bell`,
`sample_marks`, `kl_total`. The dot from Act 3 morphs (`ReplacementTransform`)
into the first $Q$ ball.

---

## ACT 6 — Read off KL: shift + shrink

**Goal:** decompose KL into its two terms, visually.

- **6a.** Show the closed form with the two terms color-coded:
  - **teal** mean-shift $\frac{\|\mu_Q-\mu_P\|^2}{2\sigma_P^2}$ — *equal* for both
    wells (equidistant from $P$);
  - **gold** flatness/variance term — the entire difference.
  Caption: *"Both valleys are the same distance from $P$, so the move costs the
  same. The whole gap is FLATNESS."*

*Implementation:* a fixed-in-frame `MathTex` with `set_color_by_tex`, plus two
small numeric chips driven by `kl_mean_term` / `kl_scale_term`.

---

## ACT 7 — PAC-Bayes payoff

**Goal:** land the "why we care".

- **7a.** Fade in the bound
  $\mathbb{E}_Q[L_\text{test}] \le \mathbb{E}_Q[L_\text{train}] + \sqrt{(D_{KL}(Q\|P)+\ln\frac1\delta)/2n}$
  with the KL term highlighted purple. Caption: *"KL is the complexity penalty in
  the PAC-Bayes generalization bound."*
- **7b.** Fade in the **gauge**: two horizontal bars (sharp vs flat) = equal grey
  "empirical loss" segment + a purple "penalty ∝ KL" segment (long for sharp,
  short for flat). Caption: *"Same training loss → the flat, low-KL $Q$ gets the
  tighter guarantee → it generalizes."*
- **7c.** Hold on the final frame.

*Implementation:* `guarantee_gauge(...)` and `pac_bound`, faded in over the
final frame.

---

## Scene structure in code

- **`KLStory`** — the full film, Acts 1→7 (the single canonical scene).
- New helpers beyond the original file:
  - `make_network(...)` — nodes + weighted edges (Act 1).
  - `animate_training(...)` — weight changes + dot drift (Acts 1b, 3a).
  - `prior_shell(...)` — the 60% iso-density inner shell (Act 2c).
  - everything else (`make_loss_plot`, `make_ball`, `make_bell`, `sample_marks`,
    `kl_*`, `guarantee_gauge`) is **reused** from `kl_3d.py`.

## Render commands (pixi tasks)

```bash
pixi run render   # KLStory, 480p15
pixi run smoke    # single-frame smoke test of KLStory (use this first!)
```
</content>
