# KL in weight space (PAC-Bayes)

A [Manim](https://www.manim.community/) animation for the **machine-learning**
meaning of $\mathrm{KL}(Q,P)$: it is the **intrinsic information** a solution
carries beyond the prior — the PAC-Bayes complexity / generalization penalty —
and the **loss curvature (the Hessian)** is what sets it.

The scene shows **two synced panels**:

* **Left — loss landscape (a 1-D weight slice).** The training loss
  $\mathcal{L}(\theta)$, the trained minimiser $\theta_f$, and the **osculating
  parabola** at $\theta_f$ whose curvature *is* the Hessian
  $H=\nabla^2\mathcal{L}(\theta_f)$. The prior $P$ and posterior $Q$ are density
  **bells**. The budget $\varepsilon$ caps the **expected** loss,
  $|\mathcal{L}(Q)-\mathcal{L}(\theta_f)|\le\varepsilon$ with
  $\mathcal{L}(Q)=\mathbb{E}_{\theta\sim Q}[\mathcal{L}(\theta)]$, drawn as the
  **gap** between the $\mathcal{L}(\theta_f)$ and $\mathcal{L}(Q)$ levels.
* **Right — weight space as 3-D balls.** A broad translucent **prior** $P$, and a
  **posterior** $Q$ nested inside it that **rescales in every dimension** as the
  curvature or the budget changes. How much smaller $Q$ is than $P$ — its shrunk
  volume — is exactly the information $\mathrm{KL}(Q,P)$ measures.

## The idea

We minimise the penalised (Gibbs / variational) objective, equivalently a
constrained one:

$$\min_Q\ \mathbb{E}_{\theta\sim Q}[\mathcal{L}(\theta)] + \beta\,\mathrm{KL}(Q,P)
\quad\Longleftrightarrow\quad
\min_Q \mathrm{KL}(Q,P)\ \ \text{s.t.}\ \ |\mathcal{L}(Q)-\mathcal{L}(\theta_f)|\le\varepsilon .$$

For a locally-quadratic loss with prior $P=\mathcal{N}(0,\sigma^2 I_d)$ this has
the closed form

$$Q=\mathcal{N}\!\Big(\theta_f,\ \tfrac{\beta}{2}\big(H+\tfrac{\beta}{2}I_d\big)^{-1}\Big),
\qquad H=\nabla^2\mathcal{L}(\theta_f),$$

so the **posterior covariance is the inverse curvature**:

* **flat** minimum (small $H$) → wide $Q\approx P$ → **low** $\mathrm{KL}$ → little information → **tight** bound → generalizes;
* **sharp** minimum (large $H$) → narrow $Q$ → **high** $\mathrm{KL}$ → much information → **loose** bound.

**The PAC-Bayes payoff.** That KL *is* the complexity penalty in the bound

$$\mathbb{E}_Q[\mathcal{L}_\text{test}] \;\le\; \mathbb{E}_Q[\mathcal{L}_\text{train}] \;+\; \sqrt{\tfrac{\mathrm{KL}(Q,P) + \ln\frac1\delta}{2n}}.$$

Minimising KL therefore means finding the **flattest, least-informative**
posterior that still fits — the flat-minima ⇒ generalization story (Hochreiter &
Schmidhuber; Keskar et al.; Dziugaite & Roy 2017; Blundell et al. 2015).

## The animation (`KLStory`)

The `KLStory` scene plays the whole arc, from a network learning to the
PAC-Bayes payoff:

1. the **network** (left) and the **weight space** (right) appear together; a
   weight vector $\theta\sim P$ is sampled from the broad prior **ball $P$**;
2. the network **trains** (edge weights change, the loss ticks down) while that
   vector **drifts** from the prior centre to $\theta_f$ — the KL *mean-shift*;
3. the **loss landscape** with its minimiser $\theta_f$ and the **osculating
   parabola** — the Hessian $H$, shown live;
4. the **$\varepsilon$ budget** on the *expected* loss (the gap between
   $\mathcal{L}(\theta_f)$ and $\mathcal{L}(Q)$) and the **curvature-shaped
   posterior** $Q$ (a bell on the left, a ball on the right);
5. **morph the curvature** flat↔sharp at fixed $\varepsilon$: $H$, $Q$ and
   $\mathrm{KL}$ all move together;
6. **sweep $\varepsilon$** at fixed curvature: the $\mathcal{L}(Q)$ level moves
   and $Q$ resizes;
7. the **PAC-Bayes bound** — $\mathrm{KL}$ is the complexity term.

![](videos/KLStory.gif)

> The `.mp4` original (sharper) is next to the gif:
> [KLStory](videos/KLStory.mp4).

## Render

Needs [`pixi`](https://pixi.sh) and a system LaTeX with `standalone.cls`
(e.g. `texlive-latex-extra cm-super dvisvgm` on Debian).

```bash
pixi install
pixi run render        # KLStory, 480p15  → media/videos/kl_3d/480p15/
pixi run render-hq     # KLStory, 1080p60 → media/videos/kl_3d/1080p60/
pixi run smoke         # single-frame smoke test of KLStory (fast, reliable)
```

> On the NFS-backed pixi env, full video renders can intermittently die with
> `mmap`/`MemoryError`, and the per-animation cache can occasionally serve a
> stale segment. `pixi run smoke` (a single `manim -s` frame) exercises all of
> `construct()` and is the reliable way to validate a scene; for a clean full
> render, retry — adding `--disable_caching` avoids stale cached segments.

The previews in [`videos/`](videos/) are the committed 1080p60 outputs.

## License

[MIT](LICENSE).
