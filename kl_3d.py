"""
3D KL divergence visualization.

Three independent Manim scenes (rendered to separate mp4 files) showing how
KL( p || q ) between two isotropic 3D Gaussians changes under:

    KLTranslation -- moving mu_q away from mu_p (sigma fixed)
    KLScaling     -- changing sigma_q with mu_q = mu_p
    KLCombined    -- translation + scaling together

Each scene shows the orbitals (60% probability isosurfaces) plus a stacked
bar that breaks D_KL into its two non-negative contributions:

    mean term :  ||mu_p - mu_q||^2 / (2 sigma_q^2)
    scale term:  ( d sigma_p^2 / sigma_q^2  -  d  +  d ln(sigma_q^2 / sigma_p^2) ) / 2

Render:
    manim -ql kl_3d.py KLTranslation KLScaling KLCombined
"""

from __future__ import annotations

import numpy as np
from manim import *


config.background_color = WHITE

TEXT_COLOR = BLACK
AXIS_COLOR = GREY_C
P_COLOR = BLUE_D
Q_COLOR = RED_D
MEAN_TERM_COLOR = TEAL_D
SCALE_TERM_COLOR = GOLD_E

SIGMA_P = 1.0

# 60% probability isosurface for a 3D isotropic Gaussian:
#   P(chi^2_3 <= (r/sigma)^2) = 0.60  =>  (r/sigma)^2 ~= 2.9462
ISO_FACTOR_60 = float(np.sqrt(2.9462))

# Cap of the bar's value axis (so D_KL ~= 5 fills the bar).
KL_BAR_MAX = 5.0


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def kl_mean_term(mu_p, mu_q, sigma_q: float) -> float:
    diff2 = float(np.sum((np.asarray(mu_p, dtype=float) - np.asarray(mu_q, dtype=float)) ** 2))
    return 0.5 * diff2 / (sigma_q ** 2)


def kl_scale_term(sigma_p: float, sigma_q: float, d: int = 3) -> float:
    # = 0 iff sigma_q == sigma_p; otherwise strictly positive.
    return 0.5 * (d * (sigma_p ** 2) / (sigma_q ** 2) - d + 2.0 * d * np.log(sigma_q / sigma_p))


# ---------------------------------------------------------------------------
# 3D mobjects
# ---------------------------------------------------------------------------

def make_orbital(center, sigma: float, color, opacity: float = 0.45) -> Sphere:
    s = Sphere(radius=sigma * ISO_FACTOR_60, resolution=(24, 48))
    s.set_color(color)
    s.set_opacity(opacity)
    s.move_to(np.asarray(center, dtype=float))
    return s


# ---------------------------------------------------------------------------
# KL stacked bar (fixed-in-frame overlay)
#
# We mutate three persistent child rectangles in place (rather than rebuilding
# via always_redraw or .become()) so the mobject identities are preserved and
# the fixed-in-frame attribute survives across frames.
# ---------------------------------------------------------------------------

def make_kl_bar(get_mean, get_scale, *, max_val: float = KL_BAR_MAX,
                width: float = 3.0, height: float = 0.35) -> VGroup:
    bg = Rectangle(
        width=width, height=height,
        stroke_color=TEXT_COLOR, stroke_width=1.5, fill_opacity=0,
    )
    # Start the fills tiny but nonzero so stretch_to_fit_width has a sane base.
    mean_rect = Rectangle(
        width=1e-3, height=height - 0.08,
        fill_color=MEAN_TERM_COLOR, fill_opacity=0.9, stroke_width=0,
    )
    scale_rect = Rectangle(
        width=1e-3, height=height - 0.08,
        fill_color=SCALE_TERM_COLOR, fill_opacity=0.9, stroke_width=0,
    )

    group = VGroup(bg, mean_rect, scale_rect)

    def update(g: VGroup) -> None:
        _bg, mr, sr = g[0], g[1], g[2]
        bl = _bg.get_corner(DL)
        mean_v = max(0.0, get_mean())
        scale_v = max(0.0, get_scale())
        mean_w = max(1e-3, min(width, width * mean_v / max_val))
        scale_w = max(1e-3, min(width - mean_w, width * scale_v / max_val))

        mr.stretch_to_fit_width(mean_w)
        mr.stretch_to_fit_height(height - 0.08)
        mr.move_to(bl + np.array([mean_w / 2, height / 2, 0]))

        sr.stretch_to_fit_width(scale_w)
        sr.stretch_to_fit_height(height - 0.08)
        sr.move_to(bl + np.array([mean_w + scale_w / 2, height / 2, 0]))

    group.add_updater(update)
    return group


def make_bar_legend() -> VGroup:
    def row(color, latex: str) -> VGroup:
        sq = Square(side_length=0.16, fill_color=color, fill_opacity=0.9, stroke_width=0)
        tex = MathTex(latex, color=TEXT_COLOR).scale(0.5)
        return VGroup(sq, tex).arrange(RIGHT, buff=0.12)

    return VGroup(
        row(MEAN_TERM_COLOR, r"\tfrac{\|\mu_p-\mu_q\|^2}{2\,\sigma_q^2}"),
        row(SCALE_TERM_COLOR,
            r"\tfrac{1}{2}\!\left[\tfrac{3\sigma_p^2}{\sigma_q^2}-3+3\ln\tfrac{\sigma_q^2}{\sigma_p^2}\right]"),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)


# ---------------------------------------------------------------------------
# Scene scaffolding
# ---------------------------------------------------------------------------

class _KLSceneBase(ThreeDScene):
    """Shared setup: axes, p sphere, formula at the bottom, KL bar on the right."""

    def setup_3d(self):
        axes_3d = ThreeDAxes(
            x_range=[-4, 4, 1], y_range=[-4, 4, 1], z_range=[-3, 3, 1],
            x_length=6, y_length=6, z_length=4,
            axis_config={"include_tip": False, "stroke_color": AXIS_COLOR, "stroke_width": 1.5},
        )
        p_sphere = make_orbital(ORIGIN, sigma=SIGMA_P, color=P_COLOR, opacity=0.45)
        p_dot = Dot3D(point=ORIGIN, color=P_COLOR, radius=0.06)
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)
        self.add(axes_3d, p_sphere, p_dot)
        return axes_3d, p_sphere, p_dot

    def build_overlay(self, get_mean, get_scale):
        formula = MathTex(
            r"D_{KL}(p\|q) = \tfrac{1}{2}\!\left[\,"
            r"\tfrac{3\sigma_p^2}{\sigma_q^2}"
            r" + \tfrac{\|\mu_p-\mu_q\|^2}{\sigma_q^2}"
            r" - 3 + 3\ln\tfrac{\sigma_q^2}{\sigma_p^2}\right]",
            color=TEXT_COLOR,
        ).scale(0.55).to_edge(DOWN, buff=0.25)

        kl_lbl = MathTex(r"D_{KL}(p\|q) =", color=TEXT_COLOR).scale(0.7)
        kl_val = DecimalNumber(0.0, num_decimal_places=3, color=TEXT_COLOR).scale(0.7)
        kl_val.add_updater(lambda m: m.set_value(get_mean() + get_scale()))
        kl_group = VGroup(kl_lbl, kl_val).arrange(RIGHT, buff=0.15)
        kl_group.to_corner(UR).shift(DOWN * 0.15 + LEFT * 0.3)

        bar = make_kl_bar(get_mean, get_scale)
        bar.next_to(kl_group, DOWN, buff=0.4, aligned_edge=RIGHT)

        legend = make_bar_legend()
        legend.next_to(bar, DOWN, buff=0.2, aligned_edge=LEFT)

        self.add_fixed_in_frame_mobjects(formula, kl_group, bar, legend)
        return dict(formula=formula, kl_group=kl_group, kl_val=kl_val,
                    bar=bar, legend=legend)


# ---------------------------------------------------------------------------
# Scene 1: pure translation
# ---------------------------------------------------------------------------

class KLTranslation(_KLSceneBase):
    def construct(self) -> None:
        self.setup_3d()

        tx, ty, tz = ValueTracker(0.0), ValueTracker(0.0), ValueTracker(0.0)

        def mu_q() -> np.ndarray:
            return np.array([tx.get_value(), ty.get_value(), tz.get_value()])

        q_sphere = make_orbital(ORIGIN, sigma=SIGMA_P, color=Q_COLOR, opacity=0.40)
        q_sphere.add_updater(lambda m: m.move_to(mu_q()))
        q_dot = Dot3D(point=ORIGIN, color=Q_COLOR, radius=0.06)
        q_dot.add_updater(lambda m: m.move_to(mu_q()))
        self.add(q_sphere, q_dot)

        self.build_overlay(
            get_mean=lambda: kl_mean_term([0.0, 0.0, 0.0], mu_q(), SIGMA_P),
            get_scale=lambda: kl_scale_term(SIGMA_P, SIGMA_P),  # identically 0
        )

        self.begin_ambient_camera_rotation(rate=0.08)
        self.play(tx.animate.set_value(2.5), run_time=2.5)
        self.wait(0.4)
        self.play(tx.animate.set_value(1.8), ty.animate.set_value(1.8), run_time=2.5)
        self.wait(0.4)
        self.play(tz.animate.set_value(1.5), run_time=2.0)
        self.wait(0.4)
        self.play(
            tx.animate.set_value(0.0), ty.animate.set_value(0.0),
            tz.animate.set_value(0.0), run_time=2.5,
        )
        self.stop_ambient_camera_rotation()
        self.wait(0.8)


# ---------------------------------------------------------------------------
# Scene 2: pure scaling
# ---------------------------------------------------------------------------

class KLScaling(_KLSceneBase):
    def construct(self) -> None:
        self.setup_3d()

        sigma_q = ValueTracker(SIGMA_P)

        # always_redraw so the sphere mesh follows the radius. For a single
        # sphere this is cheap enough; the cleaner alternative (scale a
        # persistent mesh) needs careful bookkeeping of cumulative scale.
        q_sphere = always_redraw(
            lambda: make_orbital(ORIGIN, sigma=sigma_q.get_value(),
                                 color=Q_COLOR, opacity=0.40)
        )
        q_dot = Dot3D(point=ORIGIN, color=Q_COLOR, radius=0.06)
        self.add(q_sphere, q_dot)

        overlay = self.build_overlay(
            get_mean=lambda: 0.0,
            get_scale=lambda: kl_scale_term(SIGMA_P, sigma_q.get_value()),
        )

        sigma_row = VGroup(
            MathTex(r"\sigma_q =", color=TEXT_COLOR).scale(0.6),
            DecimalNumber(1.0, num_decimal_places=2, color=TEXT_COLOR).scale(0.6),
        ).arrange(RIGHT, buff=0.12)
        sigma_row[1].add_updater(lambda m: m.set_value(sigma_q.get_value()))
        sigma_row.next_to(overlay["legend"], DOWN, buff=0.2, aligned_edge=LEFT)
        self.add_fixed_in_frame_mobjects(sigma_row)

        self.begin_ambient_camera_rotation(rate=0.08)
        self.play(sigma_q.animate.set_value(1.8), run_time=2.5)
        self.wait(0.4)
        self.play(sigma_q.animate.set_value(0.5), run_time=2.5)
        self.wait(0.4)
        self.play(sigma_q.animate.set_value(1.0), run_time=2.0)
        self.stop_ambient_camera_rotation()
        self.wait(0.8)


# ---------------------------------------------------------------------------
# Scene 3: translation + scaling
# ---------------------------------------------------------------------------

class KLCombined(_KLSceneBase):
    def construct(self) -> None:
        self.setup_3d()

        tx, ty, tz = ValueTracker(0.0), ValueTracker(0.0), ValueTracker(0.0)
        sigma_q = ValueTracker(SIGMA_P)

        def mu_q() -> np.ndarray:
            return np.array([tx.get_value(), ty.get_value(), tz.get_value()])

        q_sphere = always_redraw(
            lambda: make_orbital(mu_q(), sigma=sigma_q.get_value(),
                                 color=Q_COLOR, opacity=0.40)
        )
        q_dot = Dot3D(point=ORIGIN, color=Q_COLOR, radius=0.06)
        q_dot.add_updater(lambda m: m.move_to(mu_q()))
        self.add(q_sphere, q_dot)

        self.build_overlay(
            get_mean=lambda: kl_mean_term([0.0, 0.0, 0.0], mu_q(), sigma_q.get_value()),
            get_scale=lambda: kl_scale_term(SIGMA_P, sigma_q.get_value()),
        )

        self.begin_ambient_camera_rotation(rate=0.08)
        self.play(tx.animate.set_value(2.0), sigma_q.animate.set_value(1.6), run_time=3.0)
        self.wait(0.3)
        self.play(ty.animate.set_value(-1.5), sigma_q.animate.set_value(0.7), run_time=3.0)
        self.wait(0.3)
        self.play(tz.animate.set_value(1.2), sigma_q.animate.set_value(1.3), run_time=3.0)
        self.wait(0.3)
        self.play(
            tx.animate.set_value(0.0), ty.animate.set_value(0.0),
            tz.animate.set_value(0.0), sigma_q.animate.set_value(1.0),
            run_time=3.0,
        )
        self.stop_ambient_camera_rotation()
        self.wait(0.8)
