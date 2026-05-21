r"""
PAC-Bayes KL in weight space:  KL(Q,P) = the INTRINSIC INFORMATION a solution
carries, and the loss CURVATURE (Hessian) is what sets it.

A single Manim scene, ``KLStory``, with two synced panels.

    LEFT  -- a 1-D weight slice of the training loss L(theta).  We mark the
             trained minimiser theta_f, fit the OSCULATING PARABOLA there (its
             curvature IS the Hessian H = nabla^2 L(theta_f)), draw the prior P
             and posterior Q as density bells, and show the two loss levels
             L(theta_f) and L(Q) = E_{theta~Q}[L(theta)] whose gap is the
             tolerance epsilon:   | L(Q) - L(theta_f) | <= epsilon.

    RIGHT -- weight space.  The prior P is a round ball; the posterior Q is a
             ball too -- it RESCALES UNIFORMLY IN ALL DIMENSIONS (we only slice
             one weight direction, so nothing privileges an axis).  The shrunken
             volume of Q vs P is the information KL(Q,P) measures.

The idea (what min KL "means"):

    * Training picks a DISTRIBUTION Q over weights; the prior P = N(0, sigma^2 I)
      is broad and data-independent.
    * Penalised objective:   min_Q  E_{theta~Q}[L(theta)] + beta * KL(Q,P),
      equivalently  min_Q KL(Q,P)  s.t.  | L(Q) - L(theta_f) | <= epsilon
      (keep the EXPECTED loss L(Q) within a tolerance epsilon of the minimum).
    * For a locally-quadratic loss this has the closed form
          Q = N( theta_f, (beta/2) (H + (beta/2) I_d)^{-1} ),   H = nabla^2 L(theta_f),
      so the posterior covariance is the INVERSE CURVATURE.
    * KL(Q,P) is the complexity term of the PAC-Bayes bound
          E_Q[L_test] <= E_Q[L_train] + sqrt( (KL(Q,P) + ln 1/delta) / 2n ).

    * FLAT minimum  (small H) -> wide Q ~ P -> LOW  KL -> little information -> tight bound
    * SHARP minimum (large H) -> narrow Q   -> HIGH KL -> much  information -> loose bound

So minimising KL = the FLATTEST, least-informative posterior that still fits.

The film:  ACT 1 train (a weight vector drifts from P to theta_f) ;  ACT 2 the
well + its Hessian ;  ACT 3 the budget epsilon (the L(Q) vs L(theta_f) gap) and
the curvature-shaped posterior Q ;  ACT 4a MORPH the curvature flat<->sharp at
fixed epsilon ;  ACT 4b SWEEP epsilon at fixed curvature ;  ACT 5 the payoff.

Render:  manim -ql kl_3d.py KLStory      (smoke test: manim -s -ql kl_3d.py KLStory)
"""

from __future__ import annotations

import numpy as np
from manim import *


config.background_color = WHITE

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

TEXT_COLOR = BLACK
AXIS_COLOR = GREY_C
CURVE_COLOR = GREY_D
P_COLOR = BLUE_D            # prior P
Q_COLOR = MAROON_C         # posterior Q
HESS_COLOR = TEAL_E         # the osculating parabola / Hessian
BUDGET_COLOR = GOLD_E       # epsilon tolerance / L(Q) level
KL_COLOR = PURPLE_D         # KL(Q,P) = information
SHIFT_COLOR = TEAL_D        # mean-shift segment

# ---------------------------------------------------------------------------
# Single Gaussian well over a 1-D weight slice; its WIDTH s controls curvature.
#   L(theta) = BASE - DEPTH * exp(-(theta-theta_f)^2 / 2 s^2)
#   H = L''(theta_f) = DEPTH / s^2     (small s -> sharp -> large H)
# ---------------------------------------------------------------------------

W_MIN, W_MAX = -4.5, 4.5
L_MAX = 4.5

BASE = 4.0
DEPTH = 3.5
C_WELL = 1.8               # theta_f: the trained minimiser
L_MIN = BASE - DEPTH       # L(theta_f), the loss at the minimiser (= 0.5)

S_FLAT = 1.6               # flat well  (gentle curvature)
S_MID = 1.0
S_SHARP = 0.55             # sharp well (steep curvature)

EPS_START = 1.2            # tolerance on the EXPECTED loss L(Q) over the minimum
SIGMA_P = 2.0              # broad prior  P = N(0, sigma^2 I)
MU_P = 0.0


def loss_1d(w: float, s: float = S_MID) -> float:
    return float(BASE - DEPTH * np.exp(-(w - C_WELL) ** 2 / (2.0 * s ** 2)))


def hessian(s: float) -> float:
    """H = L''(theta_f) for the Gaussian well of width s."""
    return DEPTH / s ** 2


def parabola(w: float, s: float) -> float:
    """Osculating parabola at theta_f: L(theta_f) + 1/2 H (theta-theta_f)^2."""
    return float(L_MIN + 0.5 * hessian(s) * (w - C_WELL) ** 2)


def sigma_q(s: float, eps: float) -> float:
    """1-sigma spread of Q: widest Q whose EXPECTED loss meets the tolerance,
        L(Q) - L(theta_f) <= eps,  with L(Q) = E_{theta~Q}[L(theta)].
    The excess is DEPTH*(1 - s/sqrt(s^2+sigma^2)); solving = eps gives
    sigma = s*sqrt((DEPTH/(DEPTH-eps))^2 - 1) -- the constrained twin of the
    curvature posterior Q = N(theta_f, (beta/2)(H+(beta/2)I)^-1)."""
    denom = DEPTH - eps
    if denom <= 0:
        return SIGMA_P
    r = DEPTH / denom
    return float(np.clip(s * np.sqrt(r * r - 1.0), 0.18, SIGMA_P))


def kl_info(sigma: float) -> float:
    """KL(Q,P) in nats = information Q carries beyond P (mean shift + scale)."""
    mean = 0.5 * (C_WELL - MU_P) ** 2 / SIGMA_P ** 2
    scale = 0.5 * (sigma ** 2 / SIGMA_P ** 2 - 1.0 + np.log(SIGMA_P ** 2 / sigma ** 2))
    return float(mean + scale)


# ---------------------------------------------------------------------------
# LEFT panel helpers (2-D loss plot)
# ---------------------------------------------------------------------------

def make_axes() -> tuple[Axes, VGroup]:
    plot = Axes(
        x_range=[W_MIN, W_MAX, 1.5], y_range=[0, L_MAX, 1.0],
        x_length=5.4, y_length=3.3,
        axis_config={"include_tip": False, "color": AXIS_COLOR, "stroke_width": 1.6},
    )
    x_lab = plot.get_x_axis_label(MathTex(r"\theta", color=TEXT_COLOR).scale(0.7),
                                  edge=RIGHT, direction=RIGHT, buff=0.15)
    y_lab = plot.get_y_axis_label(MathTex(r"\mathcal{L}(\theta)", color=TEXT_COLOR)
                                  .scale(0.7), edge=UP, direction=UP, buff=0.15)
    return plot, VGroup(plot, x_lab, y_lab)


def loss_curve(plot: Axes, s: float) -> ParametricFunction:
    return plot.plot(lambda w: float(np.clip(loss_1d(w, s), 0, L_MAX)),
                     x_range=[W_MIN, W_MAX, 0.05], color=CURVE_COLOR, stroke_width=3)


def parab_curve(plot: Axes, s: float) -> ParametricFunction:
    hw = min(C_WELL - W_MIN, W_MAX - C_WELL,
             float(np.sqrt(2.0 * (L_MAX - L_MIN) / hessian(s))))
    return plot.plot(lambda w: parabola(w, s),
                     x_range=[C_WELL - hw, C_WELL + hw, 0.03],
                     color=HESS_COLOR, stroke_width=2.5).set_stroke(opacity=0.9)


def make_bell(plot: Axes, mu: float, sigma: float, color, *,
              h_scale: float = 1.7, opacity: float = 0.35) -> VGroup:
    """Density bell on the loss plot; peak height ~ 1/sigma (narrow => tall)."""
    H = min(h_scale / sigma, L_MAX - 0.2)
    lo = max(W_MIN, mu - 3.4 * sigma)
    hi = min(W_MAX, mu + 3.4 * sigma)
    f = lambda w: H * float(np.exp(-(w - mu) ** 2 / (2.0 * sigma ** 2)))
    curve = plot.plot(f, x_range=[lo, hi, 0.04], color=color, stroke_width=2.5)
    area = plot.get_area(curve, x_range=[lo, hi], color=color, opacity=opacity)
    return VGroup(area, curve)


# ---------------------------------------------------------------------------
# RIGHT panel helpers (weight space: round balls; Q rescales in ALL dimensions)
# ---------------------------------------------------------------------------

BALL_ORIGIN = np.array([4.3, 0.0, 0.0])
POS_SCALE = 0.68
RAD_SCALE = 0.82


def ball_center(mu: float) -> np.ndarray:
    return BALL_ORIGIN + np.array([mu * POS_SCALE, 0.0, 0.0])


def make_ball(mu: float, sigma: float, color, opacity: float) -> Surface:
    """An isotropic ball of radius sigma (a 1-D slice privileges no axis, so Q
    scales the SAME in every weight-space dimension)."""
    s = Sphere(center=ball_center(mu), radius=sigma * RAD_SCALE, resolution=(24, 24))
    s.set_style(fill_opacity=opacity, stroke_width=0.25,
                stroke_color=color, stroke_opacity=0.4)
    s.set_fill(color, opacity=opacity)
    return s


# ---------------------------------------------------------------------------
# ACT 1 helpers: a small feed-forward net whose weights train.
# ---------------------------------------------------------------------------

NET_LAYERS = (3, 4, 2)
NET_NODE_COLOR = GREY_E
EDGE_COLOR = "#2b4a8b"


def _net_node_positions(layers=NET_LAYERS, width=2.6, height=2.6):
    xs = np.linspace(-width / 2, width / 2, len(layers))
    positions = []
    for li, n in enumerate(layers):
        ys = np.linspace(height / 2, -height / 2, n) if n > 1 else np.array([0.0])
        positions.append([np.array([xs[li], y, 0.0]) for y in ys])
    return positions


def _edge_width(w: float) -> float:
    return float(np.clip(abs(w) * 5.0, 0.5, 5.5))


def _edge_opacity(w: float) -> float:
    return float(np.clip(0.28 + abs(w) * 0.55, 0.28, 1.0))


def make_network(center=ORIGIN, layers=NET_LAYERS, seed=7):
    rng = np.random.default_rng(seed)
    pos = _net_node_positions(layers)
    nodes = VGroup()
    for layer in pos:
        for p in layer:
            nodes.add(Dot(point=p + center, radius=0.075, color=NET_NODE_COLOR,
                          fill_opacity=1.0).set_stroke(WHITE, width=1.2))
    edges = VGroup()
    weights = []
    for li in range(len(layers) - 1):
        for a in pos[li]:
            for b in pos[li + 1]:
                w = float(rng.normal(0.0, 0.6))
                e = Line(a + center, b + center, color=EDGE_COLOR,
                         stroke_width=_edge_width(w))
                e.set_stroke(opacity=_edge_opacity(w))
                edges.add(e)
                weights.append([e, w])
    return VGroup(edges, nodes), nodes, edges, weights


def retrain_edges(weights, rng, scale: float = 0.9):
    anims = []
    for pair in weights:
        e, _ = pair
        w_new = float(rng.normal(0.0, scale))
        pair[1] = w_new
        anims.append(e.animate.set_stroke(width=_edge_width(w_new),
                                          opacity=_edge_opacity(w_new)))
    return anims


# ===========================================================================
# KLStory
# ===========================================================================

class KLStory(ThreeDScene):
    def construct(self) -> None:
        self.set_camera_orientation(phi=66 * DEGREES, theta=-90 * DEGREES, zoom=0.6)

        title_L = Text("A neural network", color=TEXT_COLOR, weight=BOLD)\
            .scale(0.36).to_corner(UL, buff=0.3)
        title_R = Text("Weight space", color=TEXT_COLOR, weight=BOLD)\
            .scale(0.36).to_corner(UR, buff=0.3)
        self.add_fixed_in_frame_mobjects(title_L, title_R)

        # =================================================================
        # ACT 1 -- sample init weights from P, train, drift to theta_f
        # =================================================================
        ref_axes = ThreeDAxes(
            x_range=[-3, 3, 1.5], y_range=[-2, 2, 1], z_range=[-2, 2, 1],
            x_length=6.0, y_length=4.0, z_length=4.0,
            axis_config={"include_tip": False, "stroke_color": GREY_B,
                         "stroke_width": 1.2},
        ).move_to(BALL_ORIGIN)
        prior_ball = make_ball(MU_P, SIGMA_P, P_COLOR, opacity=0.13)
        p_lab = Text("prior P", color=P_COLOR, weight=BOLD).scale(0.3)
        p_lab.to_corner(UR, buff=0.3).shift(DOWN * 0.55)
        self.add_fixed_in_frame_mobjects(p_lab)

        net, nodes, edges, weights = make_network(center=LEFT * 3.5 + UP * 0.15)

        # the network (left) and the weight space (right) appear together
        self.add_fixed_in_frame_mobjects(edges, nodes)
        self.bring_to_front(nodes)
        self.play(FadeIn(title_L), FadeIn(title_R), FadeIn(ref_axes),
                  FadeIn(prior_ball), FadeIn(p_lab),
                  FadeIn(edges),
                  LaggedStartMap(FadeIn, nodes, lag_ratio=0.06, scale=0.5),
                  run_time=1.3)
        self.wait(0.2)

        init_pt = ball_center(MU_P) + np.array([0.0, 0.62, 0.5])
        dot = Dot3D(point=init_pt, radius=0.085, color=Q_COLOR)
        vec = Line(ball_center(MU_P), init_pt, color=Q_COLOR, stroke_width=3.5)
        w_lab = MathTex(r"\theta \sim P", color=Q_COLOR).scale(0.5)
        w_lab.to_corner(UR, buff=0.3).shift(DOWN * 1.55)
        self.add_fixed_in_frame_mobjects(w_lab)
        loss_tag = MathTex(r"\mathcal{L} = 1.80", color=TEXT_COLOR).scale(0.5)
        loss_tag.next_to(net, DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(loss_tag)
        self.play(GrowFromPoint(vec, ball_center(MU_P)),
                  FadeIn(dot, scale=0.5), FadeIn(w_lab), FadeIn(loss_tag),
                  run_time=1.1)

        start = dot.get_center()
        target = ball_center(C_WELL)
        path = VMobject()
        path.set_points_smoothly([
            start,
            start * 0.6 + target * 0.4 + np.array([0.0, -1.0, 0.7]),
            start * 0.2 + target * 0.8 + np.array([0.0, 0.6, -0.4]),
            target,
        ])
        learn_path = VGroup()
        self.play(FadeOut(vec), run_time=0.3)
        rng = np.random.default_rng(3)
        steps = [1.10, 0.62, 0.31]
        prev_f = 0.0
        for i, lv in enumerate(steps):
            f = (i + 1) / len(steps)
            seg = VMobject()
            seg.pointwise_become_partial(path, prev_f, f)
            seg_dashed = DashedVMobject(seg, num_dashes=11)\
                .set_stroke(color=Q_COLOR, width=3, opacity=0.9)
            new_tag = MathTex(rf"\mathcal{{L}} = {lv:.2f}", color=TEXT_COLOR)\
                .scale(0.5).move_to(loss_tag)
            self.add_fixed_in_frame_mobjects(new_tag)
            self.play(*retrain_edges(weights, rng, scale=0.6 + 0.5 * f),
                      MoveAlongPath(dot, seg), Create(seg_dashed),
                      ReplacementTransform(loss_tag, new_tag), run_time=1.4)
            loss_tag = new_tag
            learn_path.add(seg_dashed)
            prev_f = f

        shift_line = DashedLine(ball_center(MU_P), target,
                                color=SHIFT_COLOR, stroke_width=3)
        shift_lab = MathTex(r"\|\mu_Q-\mu_P\|", color=SHIFT_COLOR).scale(0.5)
        shift_lab.to_corner(UR, buff=0.3).shift(DOWN * 1.15)
        self.add_fixed_in_frame_mobjects(shift_lab)
        self.play(Create(shift_line), FadeIn(shift_lab), run_time=0.9)
        self.wait(0.4)

        new_title_L = Text("Loss landscape  (1-D slice)", color=TEXT_COLOR,
                           weight=BOLD).scale(0.36).to_corner(UL, buff=0.3)
        self.add_fixed_in_frame_mobjects(new_title_L)
        self.play(FadeOut(edges), FadeOut(nodes), FadeOut(loss_tag),
                  FadeOut(shift_line), FadeOut(shift_lab), FadeOut(learn_path),
                  FadeOut(w_lab),
                  ReplacementTransform(title_L, new_title_L), run_time=1.0)

        # =================================================================
        # ACT 2 -- the well + its curvature (Hessian)
        # =================================================================
        s_t = ValueTracker(S_FLAT)               # well width -> curvature
        eps_t = ValueTracker(EPS_START)          # tolerance on L(Q)-L(theta_f)

        plot, axes_grp = make_axes()
        axes_grp.shift(LEFT * 3.4 + UP * 0.15)
        self.add_fixed_in_frame_mobjects(axes_grp)

        curve = always_redraw(lambda: loss_curve(plot, s_t.get_value()))
        self.add_fixed_in_frame_mobjects(curve)
        self.play(FadeIn(axes_grp), Create(curve), run_time=1.1)

        thf_dot = Dot(plot.c2p(C_WELL, L_MIN), radius=0.055, color=TEXT_COLOR)
        thf_lab = MathTex(r"\theta_f", color=TEXT_COLOR).scale(0.5)\
            .next_to(thf_dot, DOWN, buff=0.08)
        self.add_fixed_in_frame_mobjects(thf_dot, thf_lab)
        self.play(FadeIn(thf_dot), FadeIn(thf_lab), run_time=0.6)

        # the osculating parabola: its curvature is the Hessian H
        parab = always_redraw(lambda: parab_curve(plot, s_t.get_value()))
        self.add_fixed_in_frame_mobjects(parab)
        hess_lbl = MathTex(r"H=\nabla^2 \mathcal{L}(\theta_f)=", color=HESS_COLOR)\
            .scale(0.44).move_to(np.array([-5.5, 2.98, 0.0]))
        hess_num = always_redraw(lambda: DecimalNumber(
            hessian(s_t.get_value()), num_decimal_places=2, color=HESS_COLOR)
            .scale(0.44).next_to(hess_lbl, RIGHT, buff=0.08))
        self.add_fixed_in_frame_mobjects(hess_lbl, hess_num)
        self.play(Create(parab), FadeIn(hess_lbl), FadeIn(hess_num), run_time=1.1)
        self.wait(0.4)

        # =================================================================
        # ACT 3 -- the budget epsilon (gap between L(theta_f) and L(Q)) and Q
        # =================================================================
        # the two loss LEVELS and the epsilon gap between them
        lf_level = DashedLine(plot.c2p(W_MIN, L_MIN), plot.c2p(W_MAX, L_MIN),
                              color=GREY_B, stroke_width=1.6).set_opacity(0.7)
        lf_lab = MathTex(r"\mathcal{L}(\theta_f)", color=TEXT_COLOR).scale(0.4)\
            .next_to(plot.c2p(W_MAX, L_MIN), UR, buff=0.04)
        lq_level = always_redraw(lambda: DashedLine(
            plot.c2p(W_MIN, L_MIN + eps_t.get_value()),
            plot.c2p(W_MAX, L_MIN + eps_t.get_value()),
            color=BUDGET_COLOR, stroke_width=2.6))
        lq_lab = always_redraw(lambda: MathTex(
            r"\mathcal{L}(Q)=\mathbb{E}_{\theta\sim Q}[\mathcal{L}(\theta)]",
            color=BUDGET_COLOR).scale(0.4)
            .next_to(plot.c2p(W_MAX, L_MIN + eps_t.get_value()), UL, buff=0.05))
        theta_b = -3.6
        eps_arrow = always_redraw(lambda: DoubleArrow(
            plot.c2p(theta_b, L_MIN),
            plot.c2p(theta_b, L_MIN + eps_t.get_value()),
            color=BUDGET_COLOR, stroke_width=3, buff=0, tip_length=0.13,
            max_tip_length_to_length_ratio=0.4))
        eps_lab = always_redraw(lambda: MathTex(r"\varepsilon", color=BUDGET_COLOR)
                                .scale(0.55).next_to(eps_arrow, LEFT, buff=0.08))
        self.add_fixed_in_frame_mobjects(lf_level, lf_lab, lq_level, lq_lab,
                                         eps_arrow, eps_lab)
        self.play(FadeIn(lf_level), FadeIn(lf_lab), run_time=0.5)
        self.play(Create(lq_level), FadeIn(lq_lab),
                  GrowFromPoint(eps_arrow, plot.c2p(theta_b, L_MIN)),
                  FadeIn(eps_lab), run_time=1.0)
        self.wait(0.4)

        # prior + posterior bells
        p_bell = make_bell(plot, MU_P, SIGMA_P, P_COLOR, opacity=0.20)
        p_bell_lab = MathTex("P", color=P_COLOR).scale(0.55)\
            .move_to(plot.c2p(MU_P, 0) + np.array([-0.55, 0.5, 0.0]))
        q_bell = always_redraw(lambda: make_bell(
            plot, C_WELL, sigma_q(s_t.get_value(), eps_t.get_value()),
            Q_COLOR, opacity=0.4))
        q_bell_lab = MathTex("Q", color=Q_COLOR).scale(0.55)\
            .move_to(plot.c2p(C_WELL, 0) + np.array([0.55, 0.7, 0.0]))
        self.add_fixed_in_frame_mobjects(p_bell, p_bell_lab, q_bell, q_bell_lab)

        # RIGHT: prior round ball; posterior Q a ball that rescales in ALL dims
        q_ball = make_ball(C_WELL, sigma_q(s_t.get_value(), eps_t.get_value()),
                           Q_COLOR, 0.5)
        q_ball_lab = Text("posterior Q", color=Q_COLOR, weight=BOLD).scale(0.3)
        q_ball_lab.to_corner(UR, buff=0.3).shift(DOWN * 0.95)

        # objective + closed-form posterior (whiteboard notation)
        obj_pen = MathTex(
            r"\min_Q\ \mathbb{E}_{\theta\sim Q}[\mathcal{L}(\theta)] + \beta\, \mathrm{KL}(Q,P)",
            substrings_to_isolate=[r"\mathrm{KL}(Q,P)"], color=TEXT_COLOR).scale(0.46)
        obj_pen.set_color_by_tex(r"\mathrm{KL}(Q,P)", KL_COLOR)
        obj_con = MathTex(
            r"\Longleftrightarrow\ \min_Q \mathrm{KL}(Q,P)\ \ \text{s.t.}\ \ "
            r"\big|\mathcal{L}(Q)-\mathcal{L}(\theta_f)\big|\le\varepsilon",
            substrings_to_isolate=[r"\mathrm{KL}(Q,P)", r"\varepsilon"],
            color=TEXT_COLOR).scale(0.4)
        obj_con.set_color_by_tex(r"\mathrm{KL}(Q,P)", KL_COLOR)
        obj_con.set_color_by_tex(r"\varepsilon", BUDGET_COLOR)
        obj_sol = MathTex(
            r"P=\mathcal{N}(0,\sigma^2 I_d),\quad "
            r"Q=\mathcal{N}\!\Big(\theta_f,\ \tfrac{\beta}{2}\big(H+\tfrac{\beta}{2}I_d\big)^{-1}\Big)",
            substrings_to_isolate=[r"H"], color=TEXT_COLOR).scale(0.36)
        obj_sol.set_color_by_tex(r"H", HESS_COLOR)
        obj = VGroup(obj_pen, obj_con, obj_sol).arrange(DOWN, buff=0.12)\
            .move_to(np.array([-0.1, 3.05, 0.0]))
        self.add_fixed_in_frame_mobjects(obj)

        kl_lbl = MathTex(r"\mathrm{KL}(Q,P)=", color=KL_COLOR).scale(0.5)\
            .move_to(np.array([2.0, 2.25, 0.0]))
        kl_num = always_redraw(lambda: DecimalNumber(
            kl_info(sigma_q(s_t.get_value(), eps_t.get_value())),
            num_decimal_places=2, color=KL_COLOR)
            .scale(0.5).next_to(kl_lbl, RIGHT, buff=0.08))
        kl_unit = MathTex(r"\text{nats}", color=KL_COLOR).scale(0.4)\
            .next_to(kl_num, RIGHT, buff=0.12)
        self.add_fixed_in_frame_mobjects(kl_lbl, kl_num, kl_unit, q_ball_lab)

        self.play(ReplacementTransform(dot, q_ball), FadeIn(q_ball_lab),
                  FadeIn(p_bell), FadeIn(p_bell_lab),
                  FadeIn(q_bell), FadeIn(q_bell_lab),
                  FadeIn(obj), FadeIn(kl_lbl), FadeIn(kl_num), FadeIn(kl_unit),
                  run_time=1.5)
        self.wait(0.8)

        def caption(text: str) -> Text:
            return Text(text, color=TEXT_COLOR).scale(0.29)\
                .move_to(np.array([0.0, -2.55, 0.0]))

        cap = caption("epsilon = how far above the minimum the expected loss L(Q) may sit.")
        self.add_fixed_in_frame_mobjects(cap)
        self.play(FadeIn(cap), run_time=0.7)
        self.wait(0.6)

        # =================================================================
        # ACT 4a -- MORPH curvature flat <-> sharp at FIXED epsilon.
        #           Q rescales in ALL dims on the right; KL tracks H.
        # =================================================================
        cap2 = caption("Same epsilon, SHARPER well: bigger H, Q pinned tight, more information.")
        self.add_fixed_in_frame_mobjects(cap2)
        b_sharp = make_ball(C_WELL, sigma_q(S_SHARP, eps_t.get_value()), Q_COLOR, 0.5)
        self.play(s_t.animate.set_value(S_SHARP), Transform(q_ball, b_sharp),
                  ReplacementTransform(cap, cap2), run_time=2.4)
        self.wait(1.1)

        cap3 = caption("Same epsilon, FLATTER well: smaller H, Q stays broad, little information.")
        self.add_fixed_in_frame_mobjects(cap3)
        b_flat = make_ball(C_WELL, sigma_q(S_FLAT, eps_t.get_value()), Q_COLOR, 0.5)
        self.play(s_t.animate.set_value(S_FLAT), Transform(q_ball, b_flat),
                  ReplacementTransform(cap2, cap3), run_time=2.4)
        self.wait(1.1)

        b_mid = make_ball(C_WELL, sigma_q(S_MID, eps_t.get_value()), Q_COLOR, 0.5)
        self.play(s_t.animate.set_value(S_MID), Transform(q_ball, b_mid),
                  run_time=1.4)
        self.wait(0.4)

        # =================================================================
        # ACT 4b -- SWEEP epsilon at FIXED curvature.  The L(Q) level moves;
        #           a looser budget lets Q widen (lower KL), tighter shrinks it.
        # =================================================================
        cap4 = caption("Fix the well, loosen epsilon: L(Q) rises, Q can widen, KL falls.")
        self.add_fixed_in_frame_mobjects(cap4)
        b_loose = make_ball(C_WELL, sigma_q(S_MID, 1.8), Q_COLOR, 0.5)
        self.play(eps_t.animate.set_value(1.8), Transform(q_ball, b_loose),
                  ReplacementTransform(cap3, cap4), run_time=2.2)
        self.wait(1.0)

        cap5 = caption("Tighten epsilon: L(Q) drops toward the minimum, Q is forced narrow, KL rises.")
        self.add_fixed_in_frame_mobjects(cap5)
        b_tight = make_ball(C_WELL, sigma_q(S_MID, 0.6), Q_COLOR, 0.5)
        self.play(eps_t.animate.set_value(0.6), Transform(q_ball, b_tight),
                  ReplacementTransform(cap4, cap5), run_time=2.2)
        self.wait(1.0)

        b_back = make_ball(C_WELL, sigma_q(S_MID, EPS_START), Q_COLOR, 0.5)
        self.play(eps_t.animate.set_value(EPS_START), Transform(q_ball, b_back),
                  run_time=1.4)
        self.wait(0.4)

        # =================================================================
        # ACT 5 -- the payoff: min KL = least information = best generalisation
        # =================================================================
        bound = MathTex(
            r"\mathbb{E}_Q[\mathcal{L}_{\text{test}}]\le \mathbb{E}_Q[\mathcal{L}_{\text{train}}]+"
            r"\sqrt{\tfrac{\mathrm{KL}(Q,P)+\ln\frac{1}{\delta}}{2n}}",
            substrings_to_isolate=[r"\mathrm{KL}(Q,P)"], color=TEXT_COLOR).scale(0.5)
        bound.set_color_by_tex(r"\mathrm{KL}(Q,P)", KL_COLOR)
        bound.move_to(np.array([0.0, -3.15, 0.0]))
        punch = caption("min KL  =  least information beyond P  =  flattest fit  =  tightest bound")
        self.add_fixed_in_frame_mobjects(bound, punch)
        self.play(ReplacementTransform(cap5, punch), FadeIn(bound), run_time=1.2)
        self.wait(1.6)
