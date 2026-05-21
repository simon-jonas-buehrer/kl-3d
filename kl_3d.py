"""
PAC-Bayes KL in weight space (the ML view of KL, not the information-theory one).

A single Manim scene, ``KLStory`` (the full ACT 1->7 film described in
STORYBOARD.md), with two synced panels:

    LEFT  -- a 2-D PROJECTION: training loss L(w) along a 1-D weight slice, the
             budget line z = eps, the feasible interval {w : L(w) <= eps}, and
             the PAC-Bayes test-loss bound. The posterior is drawn here as a
             density "bell" sitting in a well; its samples on the loss curve go
             RED when they spill above the eps budget.

    RIGHT -- 3-D BALLS in weight space: each Gaussian distribution is a ball
             whose radius is its spread. A big translucent prior P, and a
             posterior Q that starts as wide as P and then SHRINKS to fit the
             budget. KL(Q || P) is read off the ball: how much smaller (and
             shifted) Q is than P.

The two move together: as the posterior ball shrinks on the right, the bell
narrows on the left and its samples come back under the eps line.

Story:
    * Training picks a DISTRIBUTION Q over weights (Bayes-by-Backprop).
    * A fixed, data-independent PRIOR P (the init distribution) is broad.
    * We want the widest Q with E_{w~Q}[L(w)] <= eps  ->  minimal KL(Q||P).
    * KL(Q||P) is the complexity term of the PAC-Bayes bound
          E_Q[L_test] <= E_Q[L_train] + sqrt( (KL(Q||P) + ln 1/delta) / 2n ).

    * FLAT well  -> tiny shrink  -> Q ~ P -> low  KL -> tight bound -> generalizes
    * SHARP well -> big shrink   -> Q << P -> high KL -> loose bound

Both wells have equal training loss, and (P being equidistant) the KL mean-shift
term is equal too -- the whole gap is the flatness / variance term.

The film first shows a small network training (its weights become one point in
weight space, drifting away from the prior ball) before the loss/eps/KL climax.

Render:
    manim -ql kl_3d.py KLStory      (smoke test: manim -s -ql kl_3d.py KLStory)
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
Q_COLOR = MAROON_C         # posterior Q (the ball / bell that moves)
Q_FLAT_COLOR = GREEN_D
Q_SHARP_COLOR = RED_D
BUDGET_COLOR = GOLD_E       # eps line
MEAN_TERM_COLOR = TEAL_D
SCALE_TERM_COLOR = GOLD_E
OVER_COLOR = RED_E          # samples above budget

# ---------------------------------------------------------------------------
# Loss landscape over a 1-D weight slice
#   L(w) = base - depth*exp(-(w-c_flat)^2/2s_flat^2) - depth*exp(-(w-c_sharp)^2/2s_sharp^2)
# Equal depth => both minima reach the same training loss (fair comparison);
# only the basin WIDTH differs.
# ---------------------------------------------------------------------------

W_MIN, W_MAX = -4.5, 4.5
L_MAX = 4.5

BASE = 4.0
DEPTH = 4.0
C_FLAT = -2.2
S_FLAT = 1.6
C_SHARP = 2.2
S_SHARP = 0.55

EPS = 2.0                  # loss budget
SIGMA_P = 2.0              # broad, data-independent prior
MU_P = 0.0                 # equidistant from both wells
D_DIM = 2                  # KL math uses a 2-D weight space


def loss_1d(w: float) -> float:
    return float(
        BASE
        - DEPTH * np.exp(-(w - C_FLAT) ** 2 / (2.0 * S_FLAT ** 2))
        - DEPTH * np.exp(-(w - C_SHARP) ** 2 / (2.0 * S_SHARP ** 2))
    )


def feasible_halfwidth(s: float) -> float:
    """Half-width of {L <= eps} around an isolated well of width s."""
    return s * float(np.sqrt(2.0 * np.log(DEPTH / (BASE - EPS))))


HW_FLAT = feasible_halfwidth(S_FLAT)     # ~1.88
HW_SHARP = feasible_halfwidth(S_SHARP)   # ~0.65
SIGMA_Q_FLAT = min(HW_FLAT, SIGMA_P)
SIGMA_Q_SHARP = min(HW_SHARP, SIGMA_P)


def kl_mean_term(mu_q: float, mu_p: float = MU_P, sigma_p: float = SIGMA_P) -> float:
    return 0.5 * (mu_q - mu_p) ** 2 / sigma_p ** 2 * D_DIM / 1.0 * (1.0 / D_DIM) \
        if False else 0.5 * (mu_q - mu_p) ** 2 / sigma_p ** 2


def kl_scale_term(sigma_q: float, sigma_p: float = SIGMA_P, d: int = D_DIM) -> float:
    return 0.5 * (d * sigma_q ** 2 / sigma_p ** 2 - d
                  + d * np.log(sigma_p ** 2 / sigma_q ** 2))


def kl_total(mu_q: float, sigma_q: float) -> float:
    return kl_mean_term(mu_q) + kl_scale_term(sigma_q)


# ---------------------------------------------------------------------------
# LEFT panel: the 2-D loss plot and the posterior "bell"
# ---------------------------------------------------------------------------

def make_loss_plot() -> tuple[Axes, VGroup]:
    plot = Axes(
        x_range=[W_MIN, W_MAX, 1.5], y_range=[0, L_MAX, 1.0],
        x_length=5.4, y_length=3.3,
        axis_config={"include_tip": False, "color": AXIS_COLOR, "stroke_width": 1.6},
    )
    x_lab = plot.get_x_axis_label(MathTex("w", color=TEXT_COLOR).scale(0.7),
                                  edge=RIGHT, direction=RIGHT, buff=0.15)
    y_lab = plot.get_y_axis_label(MathTex("L(w)", color=TEXT_COLOR).scale(0.7),
                                  edge=UP, direction=UP, buff=0.15)

    curve = plot.plot(lambda w: float(np.clip(loss_1d(w), 0, L_MAX)),
                      x_range=[W_MIN, W_MAX, 0.05], color=CURVE_COLOR, stroke_width=3)

    # The eps line and the feasible bands are now DYNAMIC (built in construct,
    # driven by a ValueTracker), so the static plot is just axes + curve.
    group = VGroup(plot, x_lab, y_lab, curve)
    return plot, group


def well_halfwidth(s: float, eps: float) -> float:
    """Half-width of {L <= eps} around an isolated well of width s, at level eps."""
    arg = DEPTH / max(BASE - eps, 1e-3)
    return 0.0 if arg <= 1.0 else s * float(np.sqrt(2.0 * np.log(arg)))


def fit_sigma(s: float, eps: float) -> float:
    """Widest Q (1-sigma) that still fits the well's feasible region at level eps."""
    return float(np.clip(well_halfwidth(s, eps), 0.18, SIGMA_P))


def feas_segment(plot: Axes, center: float, hw: float, color) -> Line:
    """A thick segment on the w-axis marking a well's feasible interval."""
    hw = max(hw, 1e-3)
    a = plot.c2p(max(W_MIN, center - hw), 0)
    b = plot.c2p(min(W_MAX, center + hw), 0)
    return Line(a, b, color=color, stroke_width=7).set_opacity(0.6)


# ---------------------------------------------------------------------------
# Closed-form Gaussian posterior (PAC-Bayes / Laplace) and the certified gap.
#
#   min_Q  KL(Q||P)   trading off training loss  ->  for Gaussian P,Q and a
#   locally-quadratic loss with curvature a = L''(w*) = DEPTH/S^2, the optimal
#   posterior std obeys precision-addition:   1/sigma_Q^2 = 1/sigma_P^2 + a/beta.
#   Flat well (small a) -> wide Q -> small KL.  Sharp well -> narrow Q -> big KL.
#
#   The PAC-Bayes bound then caps the EXPECTED test-train gap:
#       E_Q[L_test - L_train] <= sqrt( (KL + ln(n/delta)) / (2(n-1)) )  ~  sqrt(KL).
#   We require that gap <= eps.  (Constants here are schematic -- small-sample --
#   so the gap visibly tracks sqrt(KL); the SHAPE is the message, not the scale.)
# ---------------------------------------------------------------------------

A_FLAT = DEPTH / S_FLAT ** 2            # gentle curvature
A_SHARP = DEPTH / S_SHARP ** 2          # steep curvature
BETA = 6.0                              # KL trade-off weight (penalised form)


def sigma_closed_form(a: float, beta: float = BETA, sigma_p: float = SIGMA_P) -> float:
    return float(np.sqrt(1.0 / (1.0 / sigma_p ** 2 + a / beta)))


GAP_DIV = 2.0                           # schematic 2(n-1); keeps gap ~ sqrt(KL)


def cert_gap(kl: float) -> float:
    """Certified PAC-Bayes test-train gap (schematic constants)."""
    return float(np.sqrt(max(kl, 0.0) / GAP_DIV))


def make_bell(plot: Axes, mu: float, sigma: float, color, *,
              h_scale: float = 1.7, opacity: float = 0.35) -> VGroup:
    """Posterior/prior density drawn as a filled bell on the loss plot.
    Peak height ~ 1/sigma (narrow => tall), so shrinking lifts the spike."""
    H = min(h_scale / sigma, L_MAX - 0.2)
    lo = max(W_MIN, mu - 3.4 * sigma)        # clamp so the bell never spills
    hi = min(W_MAX, mu + 3.4 * sigma)        # off the plot axes
    f = lambda w: H * float(np.exp(-(w - mu) ** 2 / (2.0 * sigma ** 2)))
    curve = plot.plot(f, x_range=[lo, hi, 0.04], color=color, stroke_width=2.5)
    area = plot.get_area(curve, x_range=[lo, hi], color=color, opacity=opacity)
    return VGroup(area, curve)


def sample_marks(plot: Axes, mu: float, sigma: float, n: int, seed: int) -> VGroup:
    """Sampled weights placed on the loss curve; red if they break the budget."""
    rng = np.random.default_rng(seed)
    ws = rng.normal(mu, sigma, size=n)
    dots = VGroup()
    for w in ws:
        w = float(np.clip(w, W_MIN, W_MAX))
        L = loss_1d(w)
        col = OVER_COLOR if L > EPS + 1e-3 else Q_FLAT_COLOR
        dots.add(Dot(plot.c2p(w, np.clip(L, 0, L_MAX)), radius=0.04, color=col))
    return dots


# ---------------------------------------------------------------------------
# RIGHT panel: 3-D balls in weight space
# ---------------------------------------------------------------------------

BALL_ORIGIN = np.array([4.3, 0.0, 0.0])   # world centre of the weight-space box
POS_SCALE = 0.78                          # weight-units -> world units (position)
RAD_SCALE = 0.60                          # sigma -> world radius


def ball_center(mu: float) -> np.ndarray:
    return BALL_ORIGIN + np.array([mu * POS_SCALE, 0.0, 0.0])


def make_ball(mu: float, sigma: float, color, opacity: float) -> Surface:
    """A clean translucent ball: soft fill, very faint mesh (no busy wireframe)."""
    s = Sphere(center=ball_center(mu), radius=sigma * RAD_SCALE,
               resolution=(24, 24))
    s.set_style(fill_opacity=opacity, stroke_width=0.25,
                stroke_color=color, stroke_opacity=0.35)
    s.set_fill(color, opacity=opacity)
    return s


# ---------------------------------------------------------------------------
# ACT 1-3 helpers: a small neural net whose weights train, then collapse to a
# single point that drifts away from the prior ball in weight space.
# ---------------------------------------------------------------------------

NET_LAYERS = (3, 4, 2)          # nodes per layer
NET_NODE_COLOR = GREY_E
EDGE_COLOR = "#2b4a8b"          # one calm blue; |weight| is shown by THICKNESS


def _net_node_positions(layers=NET_LAYERS, width=2.6, height=2.6):
    """Frame-space positions for each node, laid out left-to-right by layer."""
    xs = np.linspace(-width / 2, width / 2, len(layers))
    positions = []
    for li, n in enumerate(layers):
        ys = np.linspace(height / 2, -height / 2, n) if n > 1 else np.array([0.0])
        positions.append([np.array([xs[li], y, 0.0]) for y in ys])
    return positions


def make_network(center=ORIGIN, layers=NET_LAYERS, seed=7):
    """Return (group, nodes, edges, weights) for a small feed-forward net.

    Edge stroke width AND opacity encode |weight| (thick+solid = strong); the
    colour is constant. ``weights`` is a list of (edge, value) so the caller can
    re-style them during 'training'.
    """
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
    group = VGroup(edges, nodes)   # edges first so nodes draw on top
    return group, nodes, edges, weights


def _edge_width(w: float) -> float:
    return float(np.clip(abs(w) * 5.0, 0.5, 5.5))


def _edge_opacity(w: float) -> float:
    return float(np.clip(0.28 + abs(w) * 0.55, 0.28, 1.0))


def retrain_edges(weights, rng, scale: float = 0.9):
    """Animations that nudge every edge weight (a step of 'training')."""
    anims = []
    for pair in weights:
        e, _ = pair
        w_new = float(rng.normal(0.0, scale))
        pair[1] = w_new
        anims.append(e.animate.set_stroke(width=_edge_width(w_new),
                                          opacity=_edge_opacity(w_new)))
    return anims


# ---------------------------------------------------------------------------
# Overlay: PAC-Bayes bound + test-loss gauge
# ---------------------------------------------------------------------------

def guarantee_gauge(kl_flat: float, kl_sharp: float) -> VGroup:
    TRAIN_W = 0.5
    PEN_SCALE = 0.6
    BAR_W = TRAIN_W + PEN_SCALE * kl_sharp + 0.1
    H = 0.24
    TRAIN_COLOR = GREY_C
    PEN_COLOR = PURPLE_D

    def bar(title: str, title_color, kl: float) -> VGroup:
        head = Text(title, color=title_color, weight=BOLD).scale(0.26)
        bg = Rectangle(width=BAR_W, height=H, stroke_color=TEXT_COLOR,
                       stroke_width=1.2, fill_opacity=0)
        bl = bg.get_corner(DL)
        train = Rectangle(width=TRAIN_W, height=H - 0.06, fill_color=TRAIN_COLOR,
                          fill_opacity=0.95, stroke_width=0)
        train.move_to(bl + np.array([TRAIN_W / 2, H / 2, 0]))
        pen_w = max(1e-3, PEN_SCALE * kl)
        pen = Rectangle(width=pen_w, height=H - 0.06, fill_color=PEN_COLOR,
                        fill_opacity=0.95, stroke_width=0)
        pen.move_to(bl + np.array([TRAIN_W + pen_w / 2, H / 2, 0]))
        return VGroup(head, VGroup(bg, train, pen)).arrange(
            DOWN, aligned_edge=LEFT, buff=0.05)

    legend = VGroup(
        VGroup(Square(side_length=0.11, fill_color=TRAIN_COLOR, fill_opacity=0.95,
                      stroke_width=0),
               Text("empirical loss (equal)", color=TEXT_COLOR).scale(0.24)
               ).arrange(RIGHT, buff=0.09),
        VGroup(Square(side_length=0.11, fill_color=PEN_COLOR, fill_opacity=0.95,
                      stroke_width=0),
               Text("penalty ∝ KL(Q‖P)", color=TEXT_COLOR).scale(0.24)
               ).arrange(RIGHT, buff=0.09),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)

    return VGroup(
        Text("PAC-Bayes test-loss bound", color=TEXT_COLOR, weight=BOLD).scale(0.28),
        bar("sharp Q  →  loose", Q_SHARP_COLOR, kl_sharp),
        bar("flat Q  →  tight", Q_FLAT_COLOR, kl_flat),
        legend,
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.14)


# ===========================================================================
# KLStory: the full film described in STORYBOARD.md.
#
#   ACT 1-3  (SYNCED)  the prior ball P on the right; we sample ONE weight
#            vector from it; that vector becomes the network's init weights on
#            the left.  Training then reshapes the edges (left) WHILE the same
#            vector drifts away from the prior centre (right) -- the mean shift.
#   ACT 4    the loss landscape + the epsilon budget
#   ACT 5    point -> posterior ball Q, grown to the budget (sharp then flat)
#   ACT 6    read off KL = shift + shrink
#   ACT 7    PAC-Bayes payoff
#
# Captions live at the BOTTOM (never colliding with the corner titles); the two
# panels move in lockstep inside shared self.play(...) calls.
#
# Render:  manim -ql kl_3d.py KLStory      (smoke test: manim -s -ql ...)
# ===========================================================================

MU_DRIFT = C_SHARP        # where the trained weights land (the first well we probe)


def prior_cloud(n: int, seed: int, color, *, spread: float = 0.62,
                radius: float = 0.06, opacity: float = 0.45) -> VGroup:
    """A little cloud of sample dots inside the prior ball P (3-D world)."""
    rng = np.random.default_rng(seed)
    g = VGroup()
    rmax = 1.7 * SIGMA_P * RAD_SCALE
    for _ in range(n):
        v = rng.normal(0.0, SIGMA_P * RAD_SCALE * spread, size=3)
        if np.linalg.norm(v) > rmax:
            v = v / np.linalg.norm(v) * rmax
        g.add(Dot3D(point=ball_center(MU_P) + v, radius=radius, color=color)
              .set_opacity(opacity))
    return g


class KLStory(ThreeDScene):
    def construct(self) -> None:
        self.set_camera_orientation(phi=66 * DEGREES, theta=-90 * DEGREES, zoom=0.6)

        title_L = Text("A neural network", color=TEXT_COLOR, weight=BOLD)\
            .scale(0.36).to_corner(UL, buff=0.3)
        title_R = Text("Weight space", color=TEXT_COLOR, weight=BOLD)\
            .scale(0.36).to_corner(UR, buff=0.3)
        self.add_fixed_in_frame_mobjects(title_L, title_R)

        # =================================================================
        # ACT 1-3 (SYNCED) -- sample init weights from P, then drift away
        # =================================================================

        # RIGHT: the 3-D weight space + the broad prior ball P
        ref_axes = ThreeDAxes(
            x_range=[-3, 3, 1.5], y_range=[-2, 2, 1], z_range=[-2, 2, 1],
            x_length=4.6, y_length=3.1, z_length=3.1,
            axis_config={"include_tip": False, "stroke_color": GREY_B,
                         "stroke_width": 1.2},
        ).move_to(BALL_ORIGIN)
        prior_ball = make_ball(MU_P, SIGMA_P, P_COLOR, opacity=0.14)
        p_lab = Text("prior P", color=P_COLOR, weight=BOLD).scale(0.3)
        p_lab.to_corner(UR, buff=0.3).shift(DOWN * 0.55)
        self.add_fixed_in_frame_mobjects(p_lab)

        # LEFT: the network -- nodes AND edges together (weights already inited)
        net, nodes, edges, weights = make_network(center=LEFT * 3.5 + UP * 0.15)

        self.play(FadeIn(title_L), FadeIn(title_R), FadeIn(ref_axes),
                  FadeIn(prior_ball), FadeIn(p_lab), run_time=1.2)

        # the whole network appears at once -- it already carries its init weights
        self.add_fixed_in_frame_mobjects(edges, nodes)
        self.bring_to_front(nodes)
        self.play(FadeIn(edges),
                  LaggedStartMap(FadeIn, nodes, lag_ratio=0.06, scale=0.5),
                  run_time=1.0)
        self.wait(0.3)

        # this network IS one weight vector -- a single point sampled from the
        # prior, drawn as a vector from the prior centre out to that point.
        init_pt = ball_center(MU_P) + np.array([0.0, 0.62, 0.5])
        dot = Dot3D(point=init_pt, radius=0.085, color=Q_COLOR)
        vec = Line(ball_center(MU_P), init_pt, color=Q_COLOR, stroke_width=3.5)
        w_lab = MathTex(r"w \sim P", color=Q_COLOR).scale(0.5)
        w_lab.to_corner(UR, buff=0.3).shift(DOWN * 1.55)
        self.add_fixed_in_frame_mobjects(w_lab)
        loss_tag = MathTex(r"L = 1.80", color=TEXT_COLOR).scale(0.5)
        loss_tag.next_to(net, DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(loss_tag)
        self.play(GrowFromPoint(vec, ball_center(MU_P)),
                  FadeIn(dot, scale=0.5), FadeIn(w_lab),
                  FadeIn(loss_tag), run_time=1.2)
        self.play(Indicate(dot, color=Q_COLOR, scale_factor=1.6), run_time=0.8)
        self.wait(0.5)

        # SYNCED TRAINING: edges reshape (left) while the weight vector drifts
        # away from where it started (right), tracing its path in weight space.
        start = dot.get_center()
        target = ball_center(MU_DRIFT)
        trail = TracedPath(dot.get_center, stroke_color=Q_COLOR,
                           stroke_width=3, stroke_opacity=0.6)
        self.add(trail)
        self.play(FadeOut(vec), run_time=0.4)   # vector shown; the path takes over
        rng = np.random.default_rng(3)
        steps = [1.10, 0.62, 0.31]
        for i, lv in enumerate(steps):
            new_tag = MathTex(rf"L = {lv:.2f}", color=TEXT_COLOR).scale(0.5)
            new_tag.move_to(loss_tag)
            self.add_fixed_in_frame_mobjects(new_tag)
            frac = (i + 1) / len(steps)
            via = start * (1 - frac) + target * frac
            self.play(*retrain_edges(weights, rng, scale=0.6 + 0.5 * frac),
                      dot.animate.move_to(via),
                      ReplacementTransform(loss_tag, new_tag), run_time=1.5)
            loss_tag = new_tag

        # the mean-shift segment + label
        shift_line = DashedLine(ball_center(MU_P), target,
                                color=MEAN_TERM_COLOR, stroke_width=3)
        shift_lab = MathTex(r"\|\mu_Q-\mu_P\|", color=MEAN_TERM_COLOR).scale(0.55)
        shift_lab.to_corner(UR, buff=0.3).shift(DOWN * 1.15)
        self.add_fixed_in_frame_mobjects(shift_lab)
        self.play(Create(shift_line), FadeIn(shift_lab), run_time=1.0)
        self.wait(0.6)

        # drop the network; relabel the left panel for the loss view
        new_title_L = Text("Loss + ε  (1-D weight slice)", color=TEXT_COLOR,
                           weight=BOLD).scale(0.36).to_corner(UL, buff=0.3)
        self.add_fixed_in_frame_mobjects(new_title_L)
        self.play(FadeOut(edges), FadeOut(nodes), FadeOut(loss_tag),
                  FadeOut(shift_line), FadeOut(shift_lab), FadeOut(trail),
                  FadeOut(w_lab),
                  ReplacementTransform(title_L, new_title_L), run_time=1.0)

        # =================================================================
        # ACT 4 -- loss landscape (LEFT) appears
        # =================================================================
        plot, plot_group = make_loss_plot()
        plot_group.shift(LEFT * 3.4 + UP * 0.15)
        self.add_fixed_in_frame_mobjects(plot_group)
        self.play(FadeIn(plot_group), run_time=1.1)
        self.wait(0.4)

        # =================================================================
        # ACT 5 -- the CLOSED-FORM Gaussian posterior in each well.
        #          Curvature a = L''(w*) sets sigma_Q via precision-addition;
        #          flat -> wide Q -> small KL, sharp -> narrow Q -> big KL.
        # =================================================================
        sig_flat = sigma_closed_form(A_FLAT)
        sig_sharp = sigma_closed_form(A_SHARP)
        kl_flat_v = kl_total(C_FLAT, sig_flat)
        kl_sharp_v = kl_total(C_SHARP, sig_sharp)

        # LEFT: the two closed-form bells on the loss curve
        flat_bell = make_bell(plot, C_FLAT, sig_flat, Q_FLAT_COLOR, opacity=0.35)
        sharp_bell = make_bell(plot, C_SHARP, sig_sharp, Q_SHARP_COLOR, opacity=0.35)
        self.add_fixed_in_frame_mobjects(flat_bell, sharp_bell)

        # the closed-form rule, fixed top-centre
        cf = MathTex(r"\frac{1}{\sigma_Q^2}=\frac{1}{\sigma_P^2}+"
                     r"\frac{a}{\beta}\quad(a=L''(w^*))",
                     color=TEXT_COLOR).scale(0.5).move_to(np.array([-0.4, 2.9, 0.0]))
        self.add_fixed_in_frame_mobjects(cf)

        # RIGHT: the two Q balls (closed-form widths) beside the prior P
        sharp_ball = make_ball(C_SHARP, sig_sharp, Q_SHARP_COLOR, 0.5)
        flat_ball = make_ball(C_FLAT, sig_flat, Q_FLAT_COLOR, 0.5)

        # static KL readouts (Q is fixed now -- KL no longer depends on eps)
        sharp_kl = MathTex(rf"\mathrm{{KL}}_{{\text{{sharp}}}}={kl_sharp_v:.2f}",
                           color=Q_SHARP_COLOR).scale(0.5)
        flat_kl = MathTex(rf"\mathrm{{KL}}_{{\text{{flat}}}}={kl_flat_v:.2f}",
                          color=Q_FLAT_COLOR).scale(0.5)
        VGroup(sharp_kl, flat_kl).arrange(DOWN, aligned_edge=LEFT, buff=0.18)\
            .move_to(np.array([2.75, 2.45, 0.0]))
        self.add_fixed_in_frame_mobjects(sharp_kl, flat_kl)

        self.play(ReplacementTransform(dot, sharp_ball), FadeIn(flat_ball),
                  FadeIn(flat_bell), FadeIn(sharp_bell), FadeIn(cf),
                  FadeIn(sharp_kl), FadeIn(flat_kl), run_time=1.5)
        self.wait(1.2)

        # =================================================================
        # ACT 6 -- the certified test-train gap, and the budget eps ON IT.
        #          gap = sqrt(KL / 2(n-1)) ~ sqrt(KL); we require gap <= eps.
        # =================================================================
        gap_flat = cert_gap(kl_flat_v)
        gap_sharp = cert_gap(kl_sharp_v)
        eps_t = ValueTracker(0.80)        # the gap budget (animated)

        bound = MathTex(
            r"\mathbb{E}_Q\!\big[L_{\text{test}}-L_{\text{train}}\big]\le"
            r"\sqrt{\tfrac{D_{KL}(Q\|P)}{2(n-1)}}\le\varepsilon",
            substrings_to_isolate=[r"D_{KL}(Q\|P)", r"\varepsilon"],
            color=TEXT_COLOR).scale(0.5)
        bound.set_color_by_tex(r"D_{KL}(Q\|P)", PURPLE_D)
        bound.set_color_by_tex(r"\varepsilon", BUDGET_COLOR)
        bound.move_to(np.array([0.0, -2.05, 0.0]))
        self.add_fixed_in_frame_mobjects(bound)

        # a horizontal gauge for the gap, with two bars and the eps threshold
        GAPMAX = 1.15
        gap_axis = NumberLine(x_range=[0, GAPMAX, 0.25], length=5.0,
                              color=GREY_C, stroke_width=2, include_ticks=False)
        gap_axis.move_to(np.array([0.75, -3.08, 0.0]))

        def bar(value, color, y_off):
            return Line(gap_axis.n2p(0), gap_axis.n2p(value), color=color,
                        stroke_width=9).set_opacity(0.85).shift(UP * y_off)
        sharp_bar = bar(gap_sharp, Q_SHARP_COLOR, 0.17)
        flat_bar = bar(gap_flat, Q_FLAT_COLOR, -0.17)
        # row labels in a fixed LEFT column (at each bar's start)
        sharp_tag = Text("sharp Q", color=Q_SHARP_COLOR).scale(0.26)\
            .next_to(sharp_bar.get_start(), LEFT, buff=0.2)
        flat_tag = Text("flat Q", color=Q_FLAT_COLOR).scale(0.26)\
            .next_to(flat_bar.get_start(), LEFT, buff=0.2)

        eps_rule = always_redraw(lambda: DashedLine(
            gap_axis.n2p(eps_t.get_value()) + UP * 0.36,
            gap_axis.n2p(eps_t.get_value()) + DOWN * 0.36,
            color=BUDGET_COLOR, stroke_width=3))
        eps_tag = always_redraw(lambda: VGroup(
            MathTex(r"\varepsilon=", color=BUDGET_COLOR).scale(0.45),
            DecimalNumber(eps_t.get_value(), num_decimal_places=2,
                          color=BUDGET_COLOR).scale(0.45)
        ).arrange(RIGHT, buff=0.07).next_to(
            gap_axis.n2p(eps_t.get_value()) + UP * 0.36, UP, buff=0.06))

        # pass/fail ticks in a fixed RIGHT column, updating live as eps sweeps
        right_x = gap_axis.n2p(GAPMAX)[0] + 0.5

        def verdict(gap_val, y_ref):
            ok = gap_val <= eps_t.get_value()
            return Text("✓" if ok else "✗", color=GREEN_E if ok else RED_E,
                        weight=BOLD).scale(0.42)\
                .move_to(np.array([right_x, y_ref, 0.0]))
        sharp_mark = always_redraw(lambda: verdict(gap_sharp,
                                                   sharp_bar.get_center()[1]))
        flat_mark = always_redraw(lambda: verdict(gap_flat,
                                                  flat_bar.get_center()[1]))

        gauge = (gap_axis, sharp_bar, flat_bar, sharp_tag, flat_tag,
                 eps_rule, eps_tag, sharp_mark, flat_mark)
        for m in gauge:
            self.add_fixed_in_frame_mobjects(m)

        self.play(FadeIn(bound), run_time=1.0)
        self.play(FadeIn(gap_axis),
                  GrowFromPoint(flat_bar, gap_axis.n2p(0)),
                  GrowFromPoint(sharp_bar, gap_axis.n2p(0)),
                  FadeIn(flat_tag), FadeIn(sharp_tag),
                  FadeIn(eps_rule), FadeIn(eps_tag),
                  FadeIn(sharp_mark), FadeIn(flat_mark), run_time=1.4)
        self.wait(0.8)

        # sweep the budget: high (both pass) -> mid (only flat) -> low (both fail)
        self.play(eps_t.animate.set_value(1.10), run_time=1.8)   # both fit
        self.wait(0.7)
        self.play(eps_t.animate.set_value(0.50), run_time=2.0)   # neither fits
        self.wait(0.7)
        self.play(eps_t.animate.set_value(0.80), run_time=1.8)   # only flat fits
        self.wait(2.0)
