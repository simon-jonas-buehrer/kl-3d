"""
3D KL Divergence visualization with Manim Slides.

Two isotropic 3D Gaussians are rendered as semi-transparent "orbital" spheres
drawn at the 60% probability isosurface --- analogous to how atomic orbitals
are usually depicted. The animation shows how KL(p || q) changes under:

    1. translation of q's mean,
    2. scaling of q's standard deviation,
    3. combined translation + scaling.

A 1D recap is shown first so the link to the classical Gaussian-shift example
is explicit.

Install:
    pip install manim manim-slides

Render (preview):
    manim -pql kl_3d.py KLDivergence3D

Render as slides:
    manim-slides render kl_3d.py KLDivergence3D
    manim-slides KLDivergence3D
"""

from __future__ import annotations

import numpy as np
from manim import *

try:
    from manim_slides import ThreeDSlide
    BaseScene = ThreeDSlide
    HAS_SLIDES = True
except ImportError:
    BaseScene = ThreeDScene
    HAS_SLIDES = False


# For an isotropic 3D Gaussian, the sphere of radius r encloses probability
# mass P(chi^2_3 <= (r/sigma)^2). For 60% mass, (r/sigma)^2 ≈ 2.9462,
# so the "60% orbital" radius is sigma * sqrt(2.9462) ≈ 1.7164 * sigma.
ISO_FACTOR_60 = float(np.sqrt(2.9462))

P_COLOR = BLUE
Q_COLOR = RED


def kl_iso_gaussian(mu_p, sigma_p, mu_q, sigma_q, d: int = 3) -> float:
    """Closed-form KL( N(mu_p, sigma_p^2 I) || N(mu_q, sigma_q^2 I) ) in d dims."""
    mu_p = np.asarray(mu_p, dtype=float)
    mu_q = np.asarray(mu_q, dtype=float)
    diff2 = float(np.sum((mu_p - mu_q) ** 2))
    return 0.5 * (
        d * (sigma_p ** 2) / (sigma_q ** 2)
        + diff2 / (sigma_q ** 2)
        - d
        + 2.0 * d * np.log(sigma_q / sigma_p)
    )


def make_orbital(center, sigma, color, opacity: float = 0.30) -> Sphere:
    """Semi-transparent sphere drawn at the 60% probability isosurface."""
    s = Sphere(radius=sigma * ISO_FACTOR_60, resolution=(24, 48))
    s.set_color(color)
    s.set_opacity(opacity)
    s.move_to(np.array(center, dtype=float))
    return s


class KLDivergence3D(BaseScene):
    """KL divergence demo: 1D recap -> 3D orbitals -> translation -> scaling."""

    def construct(self) -> None:
        self.slide_1d_recap()
        self.slide_intro_3d()
        self.slide_translation_3d()
        self.slide_scaling_3d()
        self.slide_combined_3d()
        self.slide_outro()

    # ------------------------------------------------------------------
    # Slide helpers
    # ------------------------------------------------------------------
    def _next(self) -> None:
        """Slide boundary (no-op when manim-slides is not installed)."""
        if HAS_SLIDES:
            self.next_slide()
        else:
            self.wait(0.4)

    # ------------------------------------------------------------------
    # Slide 1: 1D recap
    # ------------------------------------------------------------------
    def slide_1d_recap(self) -> None:
        title = Text("KL Divergence: from 1D to 3D", weight=BOLD).scale(0.8).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[0, 0.55, 0.1],
            x_length=9,
            y_length=3.2,
            axis_config={"include_tip": False, "stroke_width": 2},
        ).shift(DOWN * 0.6)
        self.add_fixed_in_frame_mobjects(axes)

        def gauss(x: float, mu: float, sigma: float) -> float:
            return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

        mu_q = ValueTracker(0.0)
        sigma_q = ValueTracker(1.0)

        p_curve = axes.plot(lambda x: gauss(x, 0.0, 1.0), color=P_COLOR, x_range=[-4.8, 4.8])
        q_curve = always_redraw(
            lambda: axes.plot(
                lambda x: gauss(x, mu_q.get_value(), sigma_q.get_value()),
                color=Q_COLOR,
                x_range=[-4.8, 4.8],
            )
        )
        self.add_fixed_in_frame_mobjects(p_curve, q_curve)

        p_lbl = MathTex("p = \\mathcal{N}(0,\\,1)", color=P_COLOR).scale(0.7)
        q_lbl = MathTex("q = \\mathcal{N}(\\mu_q,\\,\\sigma_q^2)", color=Q_COLOR).scale(0.7)
        legend = VGroup(p_lbl, q_lbl).arrange(RIGHT, buff=1.0).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(legend)

        kl_lbl = MathTex("D_{KL}(p\\|q) =").scale(0.8)
        kl_val = DecimalNumber(0.0, num_decimal_places=3).scale(0.8)
        kl_val.add_updater(
            lambda m: m.set_value(
                kl_iso_gaussian(
                    [0.0, 0.0, 0.0], 1.0,
                    [mu_q.get_value(), 0.0, 0.0], sigma_q.get_value(),
                    d=1,
                )
            )
        )
        kl_group = VGroup(kl_lbl, kl_val).arrange(RIGHT, buff=0.2).to_corner(UR).shift(DOWN * 0.6 + LEFT * 0.2)
        self.add_fixed_in_frame_mobjects(kl_group)

        self.play(Create(p_curve), Write(p_lbl))
        self.play(Create(q_curve), Write(q_lbl), FadeIn(kl_group))
        self._next()

        # Translation: shift q's mean
        self.play(mu_q.animate.set_value(2.0), run_time=2)
        self.play(mu_q.animate.set_value(-2.0), run_time=3)
        self.play(mu_q.animate.set_value(0.0), run_time=2)
        self._next()

        # Scaling: change q's variance
        self.play(sigma_q.animate.set_value(2.0), run_time=2)
        self.play(sigma_q.animate.set_value(0.5), run_time=2)
        self.play(sigma_q.animate.set_value(1.0), run_time=1.5)
        self._next()

        kl_val.clear_updaters()
        self.play(
            FadeOut(axes), FadeOut(p_curve), FadeOut(q_curve),
            FadeOut(legend), FadeOut(kl_group), FadeOut(title),
        )

    # ------------------------------------------------------------------
    # Slide 2: introduce the 3D orbital view
    # ------------------------------------------------------------------
    def slide_intro_3d(self) -> None:
        title = Text("Same idea in 3D: probability \"orbitals\"", weight=BOLD).scale(0.7).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)
        axes_3d = ThreeDAxes(
            x_range=[-4, 4, 1], y_range=[-4, 4, 1], z_range=[-3, 3, 1],
            x_length=7, y_length=7, z_length=5,
        )

        p_sphere = make_orbital(ORIGIN, sigma=1.0, color=P_COLOR, opacity=0.30)
        p_dot = Dot3D(point=ORIGIN, color=P_COLOR, radius=0.06)

        caption = Tex(
            r"Each sphere is the 60\% probability isosurface of an isotropic Gaussian."
        ).scale(0.55).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(caption)

        self.play(Create(axes_3d))
        self.play(FadeIn(p_sphere), FadeIn(p_dot), Write(caption))
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(2)
        self.stop_ambient_camera_rotation()
        self._next()

        # Add q coincident with p (KL = 0)
        q_sphere = make_orbital(ORIGIN, sigma=1.0, color=Q_COLOR, opacity=0.28)
        q_dot = Dot3D(point=ORIGIN, color=Q_COLOR, radius=0.06)

        formula = MathTex(
            r"D_{KL}(p\|q) = \tfrac{1}{2}\!\left[\, "
            r"\tfrac{3\sigma_p^2}{\sigma_q^2} + \tfrac{\|\mu_p-\mu_q\|^2}{\sigma_q^2}"
            r" - 3 + 3\ln\!\tfrac{\sigma_q^2}{\sigma_p^2} \right]"
        ).scale(0.55).to_corner(UL).shift(DOWN * 0.6)
        self.add_fixed_in_frame_mobjects(formula)

        self.play(FadeIn(q_sphere), FadeIn(q_dot), Write(formula))
        self.wait(0.5)
        self._next()

        # Stash for next slides
        self._axes_3d = axes_3d
        self._p_sphere = p_sphere
        self._p_dot = p_dot
        self._q_sphere = q_sphere
        self._q_dot = q_dot
        self._title_3d = title
        self._caption_3d = caption
        self._formula = formula

    # ------------------------------------------------------------------
    # Slide 3: translation
    # ------------------------------------------------------------------
    def slide_translation_3d(self) -> None:
        sub = Text("Translation: move μ_q away from μ_p", weight=BOLD).scale(0.55).to_edge(UP).shift(DOWN * 0.5)
        self.add_fixed_in_frame_mobjects(sub)

        mu_q = np.array([0.0, 0.0, 0.0])

        kl_lbl = MathTex("D_{KL}(p\\|q) =").scale(0.7)
        kl_val = DecimalNumber(0.0, num_decimal_places=3).scale(0.7)
        kl_group = VGroup(kl_lbl, kl_val).arrange(RIGHT, buff=0.15).to_corner(UR).shift(DOWN * 0.5 + LEFT * 0.2)
        self.add_fixed_in_frame_mobjects(kl_group)

        # Trackers drive position
        tx = ValueTracker(0.0)
        ty = ValueTracker(0.0)
        tz = ValueTracker(0.0)
        sigma_q_val = 1.0

        def current_mu():
            return np.array([tx.get_value(), ty.get_value(), tz.get_value()])

        self._q_sphere.add_updater(lambda m: m.move_to(current_mu()))
        self._q_dot.add_updater(lambda m: m.move_to(current_mu()))
        kl_val.add_updater(
            lambda m: m.set_value(
                kl_iso_gaussian([0.0, 0.0, 0.0], 1.0, current_mu(), sigma_q_val, d=3)
            )
        )

        self.play(Write(sub), FadeIn(kl_group))
        self.begin_ambient_camera_rotation(rate=0.08)

        # Translate along x
        self.play(tx.animate.set_value(2.5), run_time=2.5)
        self.wait(0.3)
        # Diagonal translation
        self.play(tx.animate.set_value(1.8), ty.animate.set_value(1.8), run_time=2.5)
        self.wait(0.3)
        # Lift along z too
        self.play(tz.animate.set_value(1.5), run_time=2)
        self.wait(0.3)
        # Back to origin
        self.play(tx.animate.set_value(0.0), ty.animate.set_value(0.0), tz.animate.set_value(0.0), run_time=2.5)

        self.stop_ambient_camera_rotation()
        self._next()

        # Clean up updaters but keep spheres for next slide
        self._q_sphere.clear_updaters()
        self._q_dot.clear_updaters()
        kl_val.clear_updaters()
        self.play(FadeOut(sub), FadeOut(kl_group))
        self._sub_t = None
        self._kl_t = None

    # ------------------------------------------------------------------
    # Slide 4: scaling
    # ------------------------------------------------------------------
    def slide_scaling_3d(self) -> None:
        sub = Text("Scaling: change σ_q while μ_q = μ_p", weight=BOLD).scale(0.55).to_edge(UP).shift(DOWN * 0.5)
        self.add_fixed_in_frame_mobjects(sub)

        kl_lbl = MathTex("D_{KL}(p\\|q) =").scale(0.7)
        kl_val = DecimalNumber(0.0, num_decimal_places=3).scale(0.7)
        sigma_lbl = MathTex("\\sigma_q =").scale(0.7)
        sigma_val = DecimalNumber(1.0, num_decimal_places=2).scale(0.7)
        kl_group = VGroup(kl_lbl, kl_val).arrange(RIGHT, buff=0.15)
        sigma_group = VGroup(sigma_lbl, sigma_val).arrange(RIGHT, buff=0.15)
        readout = VGroup(sigma_group, kl_group).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_corner(UR).shift(DOWN * 0.5 + LEFT * 0.2)
        self.add_fixed_in_frame_mobjects(readout)

        sigma_q = ValueTracker(1.0)

        # Rebuild q sphere as a function of sigma_q so the radius updates smoothly.
        self.remove(self._q_sphere)
        q_sphere = always_redraw(
            lambda: make_orbital(ORIGIN, sigma=sigma_q.get_value(), color=Q_COLOR, opacity=0.28)
        )
        self.add(q_sphere)
        self._q_sphere = q_sphere

        kl_val.add_updater(
            lambda m: m.set_value(
                kl_iso_gaussian([0.0, 0.0, 0.0], 1.0, [0.0, 0.0, 0.0], sigma_q.get_value(), d=3)
            )
        )
        sigma_val.add_updater(lambda m: m.set_value(sigma_q.get_value()))

        self.play(Write(sub), FadeIn(readout))
        self.begin_ambient_camera_rotation(rate=0.08)

        self.play(sigma_q.animate.set_value(1.8), run_time=2.5)
        self.wait(0.3)
        self.play(sigma_q.animate.set_value(0.5), run_time=2.5)
        self.wait(0.3)
        self.play(sigma_q.animate.set_value(1.0), run_time=2)

        self.stop_ambient_camera_rotation()
        self._next()

        # Detach updaters
        q_sphere.clear_updaters()
        kl_val.clear_updaters()
        sigma_val.clear_updaters()
        self.play(FadeOut(sub), FadeOut(readout))

    # ------------------------------------------------------------------
    # Slide 5: combined translation + scaling
    # ------------------------------------------------------------------
    def slide_combined_3d(self) -> None:
        sub = Text("Translate and scale together", weight=BOLD).scale(0.55).to_edge(UP).shift(DOWN * 0.5)
        self.add_fixed_in_frame_mobjects(sub)

        kl_lbl = MathTex("D_{KL}(p\\|q) =").scale(0.7)
        kl_val = DecimalNumber(0.0, num_decimal_places=3).scale(0.7)
        kl_group = VGroup(kl_lbl, kl_val).arrange(RIGHT, buff=0.15).to_corner(UR).shift(DOWN * 0.5 + LEFT * 0.2)
        self.add_fixed_in_frame_mobjects(kl_group)

        tx = ValueTracker(0.0)
        ty = ValueTracker(0.0)
        tz = ValueTracker(0.0)
        sigma_q = ValueTracker(1.0)

        def mu_now():
            return np.array([tx.get_value(), ty.get_value(), tz.get_value()])

        # Rebuild q sphere reactive to both center and radius
        self.remove(self._q_sphere)
        q_sphere = always_redraw(
            lambda: make_orbital(mu_now(), sigma=sigma_q.get_value(), color=Q_COLOR, opacity=0.28)
        )
        self.add(q_sphere)
        self._q_sphere = q_sphere

        self._q_dot.add_updater(lambda m: m.move_to(mu_now()))
        kl_val.add_updater(
            lambda m: m.set_value(
                kl_iso_gaussian([0.0, 0.0, 0.0], 1.0, mu_now(), sigma_q.get_value(), d=3)
            )
        )

        self.play(Write(sub), FadeIn(kl_group))
        self.begin_ambient_camera_rotation(rate=0.08)

        self.play(tx.animate.set_value(2.0), sigma_q.animate.set_value(1.6), run_time=3)
        self.wait(0.3)
        self.play(ty.animate.set_value(-1.5), sigma_q.animate.set_value(0.7), run_time=3)
        self.wait(0.3)
        self.play(
            tx.animate.set_value(0.0), ty.animate.set_value(0.0),
            tz.animate.set_value(0.0), sigma_q.animate.set_value(1.0),
            run_time=3,
        )

        self.stop_ambient_camera_rotation()
        self._next()

        q_sphere.clear_updaters()
        self._q_dot.clear_updaters()
        kl_val.clear_updaters()
        self.play(
            FadeOut(sub), FadeOut(kl_group), FadeOut(q_sphere), FadeOut(self._q_dot),
        )

    # ------------------------------------------------------------------
    # Slide 6: outro
    # ------------------------------------------------------------------
    def slide_outro(self) -> None:
        takeaways = VGroup(
            Tex(r"Translation only $\Rightarrow$ KL grows as $\|\mu_p-\mu_q\|^2 / (2\sigma_q^2)$."),
            Tex(r"Scaling only $\Rightarrow$ KL is asymmetric in $\sigma_q$; minimum at $\sigma_q=\sigma_p$."),
            Tex(r"KL is not symmetric: $D_{KL}(p\|q) \ne D_{KL}(q\|p)$."),
        ).scale(0.65).arrange(DOWN, aligned_edge=LEFT, buff=0.45).to_edge(LEFT).shift(RIGHT * 0.5)
        self.add_fixed_in_frame_mobjects(takeaways)

        self.play(FadeOut(self._p_sphere), FadeOut(self._p_dot),
                  FadeOut(self._axes_3d), FadeOut(self._formula),
                  FadeOut(self._caption_3d), FadeOut(self._title_3d))
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)

        title = Text("Takeaways", weight=BOLD).scale(0.9).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        for line in takeaways:
            self.play(FadeIn(line, shift=RIGHT * 0.2))
        self._next()
