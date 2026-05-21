# KL in weight space (PAC-Bayes)

A [Manim](https://www.manim.community/) animation showing the **machine-learning**
meaning of $D_{KL}(Q\,\|\,P)$ — the PAC-Bayes complexity / generalization penalty —
rather than the information-theory "extra nats" reading.

The scene shows **two synced panels**:

* **Left — loss + ε (a 2-D projection).** Training loss $L(w)$ along a 1-D weight
  slice, with the budget line $z=\varepsilon$ and the feasible interval
  $\{w : L(w)\le\varepsilon\}$ — **wide** in the flat well, **narrow** in the
  sharp one. The posterior is drawn as a density **bell**; its sampled weights on
  the loss curve turn **red** when they spill above $\varepsilon$.
* **Right — weight space as 3-D balls.** Each Gaussian is a **ball** whose radius
  is its spread: a big translucent **prior** $P$, and a **posterior** $Q$ that
  starts as wide as $P$ and **shrinks to fit** the budget. $D_{KL}(Q\|P)$ is how
  much smaller (and shifted) $Q$ is than $P$.

The two move together: as the posterior ball shrinks on the right, the bell
narrows on the left and its samples come back under $\varepsilon$. Running this
for each well gives:

* **flat** minimum → tiny shrink → broad $Q\approx P$ → **low** $D_{KL}(Q\|P)$;
* **sharp** minimum → big shrink → peaked $Q\ll P$ → **high** $D_{KL}(Q\|P)$.

Because $P$ is equidistant from both wells, the **teal** mean-shift term
$\frac{\|\mu_Q-\mu_P\|^2}{2\sigma_P^2}$ is identical for both — the entire KL gap
lives in the **gold** flatness / variance term
$\frac{1}{2}[\frac{2\sigma_Q^2}{\sigma_P^2} - 2 + 2\ln\frac{\sigma_P^2}{\sigma_Q^2}]$.

**The PAC-Bayes payoff.** That KL *is* the complexity penalty in the bound

$$\mathbb{E}_Q[L_\text{test}] \;\le\; \underbrace{\mathbb{E}_Q[L_\text{train}]}_{\le\,\varepsilon} \;+\; \sqrt{\tfrac{D_{KL}(Q\|P) + \ln\frac1\delta}{2n}}.$$

Since both wells share the same empirical loss, the bound is decided by KL alone:
the flat (low-KL) posterior gets the **tighter test-loss guarantee** — it
generalizes. That is the flat-minima ⇒ generalization story (Hochreiter &
Schmidhuber; Keskar et al.; Dziugaite & Roy 2017; Blundell et al. 2015).

## The animation (`KLStory`)

The `KLStory` scene plays the whole arc, from a network learning to the
PAC-Bayes payoff:

1. a small network **trains** (its edge weights change);
2. those weights are **one point in weight space**, and the init is the broad
   prior **ball $P$** (with its ~60% iso-density shell);
3. training **drags the point away** from $P$ — the KL *mean-shift* term;
4. drop the network, bring in the **loss landscape** and the **ε budget**;
5. replace the point with a **posterior ball $Q$** grown to the budget
   (sharp well, then flat);
6. read off **KL = shift + shrink**;
7. **PAC-Bayes payoff**: low KL ⇒ tighter bound ⇒ flat minima win.

![](videos/KLStory.gif)

> The `.mp4` original (sharper, smaller) is next to the gif:
> [KLStory](videos/KLStory.mp4).

Two design docs drive it:

* [`THEORY.md`](THEORY.md) — the maths, bottom-up (PAC-Bayes, KL, the bound).
* [`STORYBOARD.md`](STORYBOARD.md) — the shot-by-shot animation plan.

## Render

Needs [`pixi`](https://pixi.sh) and a system LaTeX with `standalone.cls`
(e.g. `texlive-latex-extra cm-super dvisvgm` on Debian).

```bash
pixi install
pixi run render        # KLStory, 480p15 → media/videos/kl_3d/480p15/
pixi run render-hq     # KLStory, 1080p60
pixi run smoke         # single-frame smoke test of KLStory (fast, reliable)
```

> On the NFS-backed pixi env, full video renders can intermittently die with
> `mmap`/`MemoryError`. `pixi run smoke` (a single `manim -s` frame) exercises
> all of `construct()` and is the reliable way to validate a scene; retry a
> full render if it fails.

The previews in [`videos/`](videos/) are the committed 480p15 outputs.

## License

[MIT](LICENSE).
