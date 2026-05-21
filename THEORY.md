# How we measure the "intrinsic complexity" of a neural network

This document explains the idea the animation visualizes. The goal is to make
one quantity intuitive:

$$D_{KL}(Q \,\|\, P)$$

— the **KL divergence between the posterior $Q$ and the prior $P$ over the
network's weights**. In the PAC-Bayes framework this number *is* the complexity
of the learned solution, and it is the term that controls how well the network
generalizes.

The explanation is built bottom-up. Each section is one idea, in plain language
first and then with the formula.

---

## 0. The one-sentence version

> Training does not just find *one* good setting of the weights — it carves out
> a whole *region* of good settings. The bigger and more "ordinary" that region
> is (close to where we started, before seeing data), the **simpler** the
> solution, and the better it generalizes. KL measures exactly how far that
> region had to move and shrink away from the starting distribution.

---

## 1. From one weight vector to a *distribution* over weights

In ordinary training we pick a single weight vector $w^\*$ by minimizing the
training loss. PAC-Bayes refuses to commit to a single point. Instead it works
with a **distribution $Q$ over weights**. To make a prediction you *sample*
$w \sim Q$ and run that network. The quantity we care about is the loss
*averaged over the distribution*:

$$\mathbb{E}_{w \sim Q}[L(w)].$$

**Why bother?** Because a single point has no notion of "size" or "robustness".
A distribution does. Asking *how wide can the good region be* turns out to be
the same as asking *how complex is the solution* — and information theory gives
us the tool (KL divergence) to put a number on it. A point estimate has no such
handle.

Concretely, $Q$ is usually a **Gaussian** centered at the trained weights with a
learned per-parameter variance. A network with 10M parameters has a $Q$ with 10M
means and 10M variances (diagonal covariance). In the animation we collapse this
to a tiny, picturable weight space so $Q$ is a **ball** (a Gaussian blob) you can
see.

---

## 2. The prior $P$ and the posterior $Q$

- **Prior $P$** — a distribution over weights chosen *before seeing the data*.
  Typically a broad Gaussian centered at the random initialization. The crucial
  word is *before*: $P$ encodes "which weights looked plausible before training".
- **Posterior $Q$** — a distribution over weights chosen *after seeing the data*.
  This is what training produces. (Despite the name, $Q$ need not be the strict
  Bayesian posterior $P(w \mid \text{data})$; it is any data-dependent
  distribution we construct.)

**Data-independence of $P$ is what makes the math valid.** If $P$ peeks at the
training data, the bound in §4 breaks. Using the *initialization* distribution as
$P$ is legal precisely because init happens before any data is seen.

---

## 3. KL divergence: what it actually measures here

$$D_{KL}(Q \,\|\, P) = \mathbb{E}_{w \sim Q}\!\left[\log \frac{Q(w)}{P(w)}\right].$$

Three readings — pick whichever clicks:

**(a) Extra coding length (MDL view).** If you built an optimal compression code
for samples of $P$ and used it on samples of $Q$, you would waste
$D_{KL}(Q\|P)$ extra nats per sample. So KL is the number of bits of *new
information* the data forced into the weights, beyond the prior.

**(b) Surprise.** How surprised would someone who believed $P$ be to see weights
drawn from $Q$? More surprise ⇒ $Q$ moved further from the prior ⇒ you fit the
data harder ⇒ more complexity.

**(c) Closed form for Gaussians.** For $P=\mathcal{N}(\mu_P,\sigma_P^2 I)$ and
$Q=\mathcal{N}(\mu_Q,\sigma_Q^2 I)$ in $d$ dimensions, per parameter:

$$D_{KL} = \frac{1}{2}\left(
\underbrace{\frac{(\mu_Q-\mu_P)^2}{\sigma_P^2}}_{\textbf{mean-shift term}}
\;+\;
\underbrace{\frac{\sigma_Q^2}{\sigma_P^2} - 1 + \log\frac{\sigma_P^2}{\sigma_Q^2}}_{\textbf{variance / flatness term}}
\right).$$

The two pieces are the heart of the visualization:

- **Mean-shift** $\dfrac{(\mu_Q-\mu_P)^2}{\sigma_P^2}$: how far the weights moved
  from initialization, measured in units of the prior's width. This is the
  "the dot drifted away from the ball" part.
- **Variance / flatness** terms in $\sigma_Q$: how much the posterior had to
  *shrink* its uncertainty. A peaked (overconfident) $Q$ has small $\sigma_Q$,
  and through $\log(\sigma_P^2/\sigma_Q^2)$ this contributes a large KL.

> **KL is NOT a mean difference.** A common mistake is to think KL is just
> $\|\mu_Q-\mu_P\|$. It also contains the variance term — and that variance term
> is exactly what distinguishes a *sharp* minimum from a *flat* one.

---

## 4. The PAC-Bayes bound, term by term

McAllester's bound: with probability at least $1-\delta$ over the draw of $n$
training samples,

$$\underbrace{\mathbb{E}_{w\sim Q}[L_\text{test}(w)]}_{\text{what we care about}}
\;\le\;
\underbrace{\mathbb{E}_{w\sim Q}[L_\text{train}(w)]}_{\text{what we can measure}}
\;+\;
\underbrace{\sqrt{\frac{D_{KL}(Q\|P) + \log\frac{1}{\delta}}{2n}}}_{\text{complexity penalty}}.$$

Reading left to right:

- **Left:** the true generalization error, averaged over $Q$. Unknown, but
  bounded.
- **First term right:** average *training* loss under $Q$ — measurable by
  sampling weights from $Q$ and evaluating on the training set.
- **Square-root term:** the price paid for having used the data. It shrinks as
  $n$ grows and **grows with $D_{KL}(Q\|P)$**.
- $\delta$ is the confidence level; $\log(1/\delta)$ is tiny ($\delta=0.05 \Rightarrow \log 20 \approx 3$).

So $D_{KL}(Q\|P)$ is **the** complexity term. The entire PAC-Bayes story is:
*keep training loss low AND keep $Q$ close to $P$ in KL.*

This is **non-vacuous**: classical bounds (VC dimension, Rademacher) give
deep-net guarantees like "test error $\le 1.0$", which says nothing. Dziugaite &
Roy (2017) were the first to compute PAC-Bayes bounds for real trained deep nets
that came out *below 1*.

---

## 5. The constrained view = the penalized view (why "minimize KL under an ε
budget" is the same as Bayes-by-Backprop)

The intuition "find the posterior with minimal KL while keeping loss $\le
\varepsilon$" is the **constrained form**:

$$\min_Q \; D_{KL}(Q\|P) \quad\text{s.t.}\quad \mathbb{E}_{w\sim Q}[L_\text{train}(w)] \le \varepsilon.$$

By Lagrangian duality this is equivalent to the **penalized form**:

$$\min_Q \; \mathbb{E}_{w\sim Q}[L_\text{train}(w)] + \beta\, D_{KL}(Q\|P),$$

for some $\beta$ tied to $\varepsilon$. Written in negative-log-likelihood terms,
the penalized form is exactly the **ELBO** of variational inference. So:

- **PAC-Bayes** motivation: generalization guarantees.
- **Variational Bayes** motivation: approximate the posterior.
- **Same objective.** **Bayes-by-Backprop** (Blundell et al. 2015) is the
  practical algorithm: parameterize $Q$ as a diagonal Gaussian, sample weights
  with the reparameterization trick, and backprop through loss + closed-form KL.

The animation uses the **constrained picture** because it is the most visual:
grow $Q$ as large as possible *inside* the loss-budget region.

---

## 6. Why flat minima win (the punchline)

Consider two minima of the loss with the **same depth** (same training loss):

| | feasible region $\{L\le\varepsilon\}$ | widest $Q$ that fits | $\sigma_Q$ | KL | bound |
|---|---|---|---|---|---|
| **flat** well | wide | broad, close to $P$ | large | **low** | tight ✓ |
| **sharp** well | narrow | must shrink | small | **high** | loose ✗ |

A flat minimum lets $Q$ keep a large $\sigma_Q$ (you can wiggle the weights
without hurting the loss), which keeps the variance term small, which keeps KL
low, which tightens the bound. That is the **flat-minima ⇒ generalization**
story (Hochreiter & Schmidhuber; Keskar et al.; Entropy-SGD; Dziugaite & Roy).

If both wells are **equidistant** from the prior mean, the mean-shift term is
*identical* for both — so the entire KL gap lives in the flatness term. That is
the cleanest possible demonstration that the complexity difference is about
*shape*, not *location*.

---

## 7. Putting the original intuition back together (corrected)

The starting intuition was: *"the prior is the init distribution, the posterior
is what training gives me, and the KL between them measures how much the network
had to learn — its complexity."* That is right in spirit. The corrections:

1. The prior must **not** look at the training data. Init is fine because init
   is data-independent.
2. KL is **not** just a mean difference — it includes a variance term, and the
   variance term is where flatness (and most of the interesting behavior) lives.
3. Complexity is a property of the **$(P, Q)$ pair**, not of the architecture
   alone. Two identical architectures can have very different KL complexity.
4. "min KL s.t. loss $\le\varepsilon$" and "min loss $+\,\beta\,$KL"
   (ELBO / Bayes-by-Backprop) are the **same thing**, related by Lagrangian
   duality.

---

## 8. The variables, as used in the animation

| symbol | meaning | in the picture |
|---|---|---|
| $w$ | a weight vector | a **dot** in weight space; a value on the weight axis |
| $P$ | prior = init distribution, broad, data-independent | a big translucent **ball** at the origin |
| $Q$ | posterior = what training produces | a smaller **ball** that drifts and shrinks |
| $\mu_Q$ | center of $Q$ | where the trained dot sits |
| $\sigma_Q$ | spread of $Q$ | the radius of the $Q$ ball |
| $L(w)$ | training loss | the landscape / 1-D loss curve |
| $\varepsilon$ | loss budget | a horizontal line; defines the feasible region |
| $\{L \le \varepsilon\}$ | feasible region | the interval/volume $Q$ must stay inside |
| $D_{KL}(Q\|P)$ | complexity | how far + how much $Q$ shifted and shrank from $P$ |

---

## References

- Dziugaite & Roy (2017), *Computing Nonvacuous Generalization Bounds for Deep
  (Stochastic) Neural Networks with Many More Parameters than Training Data.*
- Blundell et al. (2015), *Weight Uncertainty in Neural Networks*
  (Bayes-by-Backprop).
- McAllester (1999), *PAC-Bayesian Model Averaging.*
- Pérez-Ortiz, Rivasplata et al., self-certified PAC-Bayes bounds.
- Hochreiter & Schmidhuber (1997), *Flat Minima*; Keskar et al. (2017).
</content>
</invoke>
