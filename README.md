# KL Divergence in 3D — Manim Slides

A short [Manim](https://www.manim.community/) presentation built with the
[`manim-slides`](https://www.manim.community/plugin/manim-slides/) plugin that
extends the classic 1D Gaussian-shift example of the Kullback–Leibler
divergence into **three dimensions**.

Two isotropic 3D Gaussians are rendered as semi-transparent spheres drawn at
their 60 % probability isosurface — the same convention used to depict atomic
orbitals in quantum mechanics. The animation then varies:

1. the **translation** of $\mu_q$ away from $\mu_p$,
2. the **scaling** of $\sigma_q$ while $\mu_q = \mu_p$,
3. the **combined** motion of both.

A live read-out of $D_{KL}(p\,\|\,q)$ updates as the q-orbital deforms.

The closed-form KL divergence between two isotropic Gaussians in $d$ dimensions is

$$
D_{KL}\bigl(\mathcal{N}(\mu_p,\sigma_p^2 I) \,\big\|\, \mathcal{N}(\mu_q,\sigma_q^2 I)\bigr)
= \tfrac{1}{2}\!\left[\,
d\,\frac{\sigma_p^2}{\sigma_q^2}
+ \frac{\|\mu_p-\mu_q\|^2}{\sigma_q^2}
- d
+ 2d\ln\!\frac{\sigma_q}{\sigma_p}
\right].
$$

## Live slides

Once GitHub Pages is enabled on this repo, the rendered Reveal.js slides are
served at:

    https://<your-user>.github.io/<this-repo>/

(See _Hosting on GitHub Pages_ below.)

## Local rendering

```bash
# 1. install deps (ffmpeg + LaTeX must be on PATH for full output)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. render the slides (low-quality preview is fine for browsing)
manim-slides render -ql kl_3d.py KLDivergence3D

# 3a. play interactively (PySide6 window with arrow keys)
manim-slides KLDivergence3D

# 3b. or export a self-contained Reveal.js HTML
manim-slides convert KLDivergence3D index.html
xdg-open index.html
```

Higher quality:

```bash
manim-slides render -qh kl_3d.py KLDivergence3D     # 1080p
manim-slides render -qk kl_3d.py KLDivergence3D     # 4K (slow)
```

## Hosting on GitHub Pages

This repo ships a GitHub Actions workflow
([`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)) that on every
push to `main`:

1. installs ffmpeg, Cairo/Pango, a full TeX Live, manim and manim-slides;
2. renders `KLDivergence3D` at preview quality (`-ql`);
3. converts the result to a self-contained Reveal.js HTML file;
4. publishes it to GitHub Pages.

To enable hosting on a fresh fork:

1. Push the repo to GitHub.
2. On GitHub, go to **Settings → Pages → Build and deployment → Source** and
   pick **GitHub Actions**.
3. Push to `main` (or run the workflow manually from the Actions tab). The
   first run takes ~6–10 min because of the TeX Live install + 3D render.
4. The site URL appears in the **Actions → deploy → deploy** job summary and
   under **Settings → Pages** once the run completes.

> **Tip:** if you want a faster CI loop, edit `-ql` to `-qm` in the workflow
> only after the slides look right at low quality, and consider caching the
> apt + pip layers.

## Project layout

```
.
├── kl_3d.py                  # the Manim/Manim-Slides scene
├── requirements.txt          # pinned Python deps
├── .github/workflows/deploy.yml   # render + deploy to Pages
├── .gitignore
└── README.md
```

## License

MIT — see [`LICENSE`](LICENSE).
