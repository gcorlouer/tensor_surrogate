#!/usr/bin/env python3
"""Numerical checks of the section on balanced dynamics without alignment (ttns_surrogate.tex).

Every claim of the section is compared with direct integrations of the gradient flow of the cores
(A, B, C) of the four-leaf model, or of the seven-core tree for the rank-one channels of deeper trees.

    python3 sec_nonaligned.py
"""
import sys

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm

rng = np.random.default_rng(7)
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   ({detail})" if detail else ""), flush=True)


def section(title):
    print(f"\n== {title}", flush=True)


def orth(n, k):
    return np.linalg.qr(rng.standard_normal((n, n)))[0][:, :k]


def unit(x):
    return x / np.linalg.norm(x)


def psd_power(G, p, rtol=1e-10):
    lam, Q = np.linalg.eigh((G + G.T) / 2)
    lam = np.where(lam > rtol * max(lam.max(), 1e-300), lam, 0.0)
    return (Q * lam ** p) @ Q.T


# ---------------------------------------------------------------------------
# Gradient flow of the three cores, M = A C B^T, with R = grad_M L given as a function of M.
# ---------------------------------------------------------------------------
class Flow:
    def __init__(self, shapes, grad_M):
        self.shapes, self.grad_M = shapes, grad_M
        self.sizes = [int(np.prod(s)) for s in shapes]

    def unpack(self, th):
        parts = np.split(th[:sum(self.sizes)], np.cumsum(self.sizes)[:-1])
        return [p.reshape(s) for p, s in zip(parts, self.shapes)]

    def pack(self, A, B, C):
        return np.concatenate([A.ravel(), B.ravel(), C.ravel()])

    def velocities(self, A, B, C):
        R = self.grad_M(A @ C @ B.T)
        return -R @ B @ C.T, -R.T @ A @ C, -A.T @ R @ B

    def rhs(self, _, th):
        A, B, C = self.unpack(th)
        dA, dB, dC = self.velocities(A, B, C)
        out = self.pack(dA, dB, dC)
        if len(th) > sum(self.sizes):          # extra state: tau = int ||C||_F dt (rank-one channels)
            out = np.append(out, np.linalg.norm(C))
        return out

    def run(self, A, B, C, T, with_tau=False, rtol=1e-11, atol=1e-13, events=None):
        th0 = self.pack(A, B, C)
        if with_tau:
            th0 = np.append(th0, 0.0)
        return solve_ivp(self.rhs, (0, T), th0, method="DOP853", rtol=rtol, atol=atol, dense_output=True,
                         events=events)


def balanced(U, V, WL, WR, s):
    S = np.diag(s)
    return U @ S @ WL.T, V @ S @ WR.T, WL @ S @ WR.T


def Kop(M, X):
    L1, R1 = psd_power(M @ M.T, 1 / 3), psd_power(M.T @ M, 1 / 3)
    return L1 @ L1 @ X + L1 @ X @ R1 + X @ R1 @ R1


def svd_r(M, r):
    U, s, Vt = np.linalg.svd(M)
    return U[:, :r], s[:r], Vt[:r].T


# Four-leaf model of Section 3 and a rank-3 teacher.
d12, d34, DL, DR = 6, 4, 2, 3
sig_star = np.array([2.0, 1.4, 0.8])
Us, Vs = orth(d12, 3), orth(d34, 3)
Mstar = Us @ np.diag(sig_star) @ Vs.T
shapes = [(d12, DL), (d34, DR), (DL, DR)]
white = Flow(shapes, lambda M: M - Mstar)

# ---------------------------------------------------------------------------
section("1. Exact spectral equations on a non-aligned balanced trajectory")
A0, B0, C0 = balanced(orth(d12, 2), orth(d34, 2), orth(DL, 2), orth(DR, 2), np.array([0.7, 0.35]))
sol = white.run(A0, B0, C0, 4.0)
errs = dict(F=0.0, mu=0.0, OmU=0.0, OmV=0.0, outU=0.0, outV=0.0)
h = 1e-6
for t in np.linspace(0.2, 4.0, 12):
    A, B, C = white.unpack(sol.sol(t))
    dA, dB, dC = white.velocities(A, B, C)
    M = A @ C @ B.T
    Mdot = dA @ C @ B.T + A @ dC @ B.T + A @ C @ dB.T
    U, mu, V = svd_r(M, 2)
    Th = U.T @ Mstar @ V
    kap = mu[:, None] ** (4 / 3) + mu[:, None] ** (2 / 3) * mu[None, :] ** (2 / 3) + mu[None, :] ** (4 / 3)
    F = U.T @ Mdot @ V
    errs["F"] = max(errs["F"], np.abs(F - kap * (Th - np.diag(mu))).max())
    # derivative of the SVD along Mdot, by central differences of the SVD of M +- h Mdot
    Up, mup, Vp = svd_r(M + h * Mdot, 2)
    Um, mum, Vm = svd_r(M - h * Mdot, 2)
    sgn = lambda Xn, X0: Xn * np.sign(np.sum(Xn * X0, axis=0))
    Up, Um, Vp, Vm = sgn(Up, U), sgn(Um, U), sgn(Vp, V), sgn(Vm, V)
    Udot, Vdot, mudot = (Up - Um) / (2 * h), (Vp - Vm) / (2 * h), (mup - mum) / (2 * h)
    errs["mu"] = max(errs["mu"], np.abs(mudot - 3 * mu ** (4 / 3) * (np.diag(Th) - mu)).max())
    OmU, OmV = U.T @ Udot, V.T @ Vdot
    pred_U = kap[0, 1] * (mu[1] * Th[0, 1] + mu[0] * Th[1, 0]) / (mu[1] ** 2 - mu[0] ** 2)
    pred_V = kap[0, 1] * (mu[0] * Th[0, 1] + mu[1] * Th[1, 0]) / (mu[1] ** 2 - mu[0] ** 2)
    errs["OmU"] = max(errs["OmU"], abs(OmU[0, 1] - pred_U))
    errs["OmV"] = max(errs["OmV"], abs(OmV[0, 1] - pred_V))
    PU, PV = np.eye(d12) - U @ U.T, np.eye(d34) - V @ V.T
    errs["outU"] = max(errs["outU"], np.abs(PU @ Udot - PU @ Mstar @ V @ np.diag(mu ** (1 / 3))).max())
    errs["outV"] = max(errs["outV"], np.abs(PV @ Vdot - PV @ Mstar.T @ U @ np.diag(mu ** (1 / 3))).max())
print("  max errors:", {k: f"{v:.1e}" for k, v in errs.items()})
check("U^T Mdot V = kappa o (Theta - Sigma)", errs["F"] < 1e-10)
check("sigma_i' = 3 sigma_i^{4/3} (Theta_ii - sigma_i)", errs["mu"] < 1e-6)
check("in-span rotations (Omega_U)_12, (Omega_V)_12", errs["OmU"] < 1e-6 and errs["OmV"] < 1e-6)
check("out-of-span motion (I-UU^T) U' = (I-UU^T) M* V Sigma^{1/3}, and similarly for V",
      errs["outU"] < 1e-6 and errs["outV"] < 1e-6)

# non-whitened version: U^T Mdot V = -kappa o (U^T R V)
S12 = (lambda G: G @ G.T / d12 + 0.3 * np.eye(d12))(rng.standard_normal((d12, d12)))
S34 = (lambda G: G @ G.T / d34 + 0.3 * np.eye(d34))(rng.standard_normal((d34, d34)))
nonwhite = Flow(shapes, lambda M: S12 @ (M - Mstar) @ S34)
A, B, C = A0, B0, C0
dA, dB, dC = nonwhite.velocities(A, B, C)
M = A @ C @ B.T
U, mu, V = svd_r(M, 2)
kap = mu[:, None] ** (4 / 3) + mu[:, None] ** (2 / 3) * mu[None, :] ** (2 / 3) + mu[None, :] ** (4 / 3)
F = U.T @ (dA @ C @ B.T + A @ dC @ B.T + A @ C @ dB.T) @ V
check("general input law: U^T Mdot V = -kappa o (U^T R_Delta V)",
      np.abs(F + kap * (U.T @ S12 @ (M - Mstar) @ S34 @ V)).max() < 1e-12)

# ---------------------------------------------------------------------------
section("2. Invariant subspaces (whitened)")
n2, m2 = 8, 7
Us2, Vs2 = orth(n2, 2), orth(m2, 2)
Mst2 = Us2 @ np.diag([1.5, 0.7]) @ Vs2.T
flow2 = Flow([(n2, 2), (m2, 2), (2, 2)], lambda M: M - Mst2)
U0, V0 = orth(n2, 2), orth(m2, 2)
sol2 = flow2.run(*balanced(U0, V0, orth(2, 2), orth(2, 2), np.array([0.4, 0.2])), 20.0)
QL = np.linalg.qr(np.hstack([U0, Us2]))[0]
QR = np.linalg.qr(np.hstack([V0, Vs2]))[0]
leak = 0.0
for t in np.linspace(0, 20, 50):
    A, B, C = flow2.unpack(sol2.sol(t))
    M = A @ C @ B.T
    leak = max(leak, np.abs(M - QL @ QL.T @ M).max(), np.abs(M - M @ QR @ QR.T).max())
check("range M(t) in range M(0) + range M*, and similarly for the row spaces", leak < 1e-10, f"leak {leak:.1e}")

# ---------------------------------------------------------------------------
section("3. Rank-one channel: exact solution by quadrature")
# eigen-decomposition of the dilation L = [[0, M*], [M*^T, 0]]
def dilation_coefficients(Mt, u0, v0, factors=None):
    if factors is None:
        Ut, st, Vtt = np.linalg.svd(Mt, full_matrices=False)
        keep = st > 1e-12 * st.max()
        Ut, st, Vt = Ut[:, keep], st[keep], Vtt[keep].T
    else:
        Ut, st, Vt = factors
    p, q = Ut.T @ u0, Vt.T @ v0
    alpha, beta = (p + q) / 2, (p - q) / 2
    gamma2 = (np.sum(u0 ** 2) - np.sum(p ** 2) + np.sum(v0 ** 2) - np.sum(q ** 2)) / 2
    return st, alpha, beta, gamma2, Ut, Vt


def Zfun(tau, st, alpha, beta, gamma2):
    return np.sum(alpha ** 2 * np.exp(2 * st * tau) + beta ** 2 * np.exp(-2 * st * tau)) + gamma2


def rank_one_prediction(Mt, u0, v0, sigma0, tau_max, N=3):
    """Integrate (J, t) in tau: J' = Z^{N/2}, t' = Z^{-(N-2)/2} (1/sigma0 + N J)^{(N-2)/N}."""
    st, al, be, g2, _, _ = dilation_coefficients(Mt, u0, v0)

    def rhs(tau, y):
        Z = Zfun(tau, st, al, be, g2)
        return [Z ** (N / 2), Z ** (-(N - 2) / 2) * (1 / sigma0 + N * y[0]) ** ((N - 2) / N)]

    return solve_ivp(rhs, (0, tau_max), [0.0, 0.0], method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True)


def sigma_of_tau(tau, J, Mt, u0, v0, sigma0):
    st, al, be, g2, _, _ = dilation_coefficients(Mt, u0, v0)
    return Zfun(tau, st, al, be, g2) ** 1.5 / (1 / sigma0 + 3 * J)


u0, v0 = unit(rng.standard_normal(d12)), unit(rng.standard_normal(d34))
wl, wr = unit(rng.standard_normal(DL)), unit(rng.standard_normal(DR))
s0 = 0.08
A, B, C = s0 * np.outer(u0, wl), s0 * np.outer(v0, wr), s0 * np.outer(wl, wr)
T3 = 40.0
sol3 = white.run(A, B, C, T3, with_tau=True)
pred = rank_one_prediction(Mstar, u0, v0, s0 ** 3, sol3.sol(T3)[-1] + 1.0)
Ld = np.block([[np.zeros((d12, d12)), Mstar], [Mstar.T, np.zeros((d34, d34))]])
y0 = np.concatenate([u0, v0]) / np.sqrt(2)
e_dir, e_sig, e_t = 0.0, 0.0, 0.0
for t in np.linspace(0, T3, 81):
    th = sol3.sol(t)
    A_, B_, C_ = white.unpack(th)
    tau = th[-1]
    M = A_ @ C_ @ B_.T
    Uu, sv, Vv = svd_r(M, 1)
    z = expm(tau * Ld) @ y0
    yhat = z / np.linalg.norm(z)
    u_pred, v_pred = np.sqrt(2) * yhat[:d12], np.sqrt(2) * yhat[d12:]
    e_dir = max(e_dir, min(np.abs(Uu[:, 0] - u_pred).max(), np.abs(Uu[:, 0] + u_pred).max()),
                abs(abs(Uu[:, 0] @ u_pred) - 1), abs(abs(Vv[:, 0] @ v_pred) - 1))
    J, tp = pred.sol(tau)
    e_sig = max(e_sig, abs(sv[0] - sigma_of_tau(tau, J, Mstar, u0, v0, s0 ** 3)))
    e_t = max(e_t, abs(tp - t))
print(f"  max |direction error| {e_dir:.1e}, |sigma(t) error| {e_sig:.1e}, |t(tau) error| {e_t:.1e}")
check("rank-one channel: (u, v) = normalized exp(tau L) y(0), with tau = int s dt", e_dir < 1e-8)
check("rank-one channel: sigma(tau) = Z^{3/2} / (1/sigma0 + 3 int Z^{3/2})", e_sig < 1e-8)
check("rank-one channel: t(tau) = int Z^{-1/2} (1/sigma0 + 3 int Z^{3/2})^{1/3}", e_t < 1e-6)
A_, B_, C_ = white.unpack(white.run(A, B, C, 80.0).sol(80.0))
check("rank-one channel converges to the best rank-one approximation sigma*_1 u*_1 v*_1^T",
      np.abs(A_ @ C_ @ B_.T - sig_star[0] * np.outer(Us[:, 0], Vs[:, 0])).max() < 1e-6)

# ---------------------------------------------------------------------------
section("4. Escape time of a rank-one channel: t_esc = I / s0 + O(1)")
st, al, be, g2, _, _ = dilation_coefficients(Mstar, u0, v0)
I = quad(lambda tau: Zfun(tau, st, al, be, g2) ** -0.5, 0, np.inf, limit=400)[0]
eta = 0.5


def t_escape_quadrature(sig0, Mt, u, v, level):
    """Time at which sigma(t) first reaches `level`, from the exact quadrature."""
    pr = rank_one_prediction(Mt, u, v, sig0, 60.0)
    taus = np.linspace(0, 60, 600001)
    J = pr.sol(taus)
    sig = Zfun_vec(taus, Mt, u, v) ** 1.5 / (1 / sig0 + 3 * J[0])
    i = np.argmax(sig >= level)
    return J[1, i]


def Zfun_vec(taus, Mt, u, v):
    st_, al_, be_, g2_, _, _ = dilation_coefficients(Mt, u, v)
    return (al_[None, :] ** 2 * np.exp(2 * st_[None, :] * taus[:, None])
            + be_[None, :] ** 2 * np.exp(-2 * st_[None, :] * taus[:, None])).sum(1) + g2_


ratios = []
for s_init in (1e-2, 1e-3, 1e-4):
    ratios.append(s_init * t_escape_quadrature(s_init ** 3, Mstar, u0, v0, eta * sig_star[0]))
print(f"  I = {I:.6f};  s0 * t_esc for s0 = 1e-2, 1e-3, 1e-4: {np.round(ratios, 6)}")
check("s0 t_esc -> I as s0 -> 0, with an O(s0) correction",
      abs(ratios[-1] - I) < 2e-3 * I and abs(ratios[-1] - I) < abs(ratios[0] - I))
# direct simulation for s0 = 1e-2 agrees with the quadrature
s_init = 1e-2
ev = lambda t, th: np.linalg.norm(white.unpack(th)[2]) ** 3 - eta * sig_star[0]
ev.terminal, ev.direction = True, 1
simA, simB, simC = s_init * np.outer(u0, wl), s_init * np.outer(v0, wr), s_init * np.outer(wl, wr)
sim = white.run(simA, simB, simC, 2000.0, events=ev)
t_sim = sim.t_events[0][0]
check("direct simulation (s0 = 1e-2) reproduces the quadrature escape time",
      abs(t_sim - ratios[0] / s_init) < 1e-4 * t_sim, f"{t_sim:.4f} vs {ratios[0] / s_init:.4f}")
upper = np.arcsinh(np.sqrt(g2) / abs(al[0])) / (sig_star[0] * np.sqrt(g2))
check("bounds 1/sigma*_1 <= I <= arsinh(gamma/|alpha_1|) / (sigma*_1 gamma)", 1 / sig_star[0] <= I <= upper,
      f"{1 / sig_star[0]:.4f} <= {I:.4f} <= {upper:.4f}")

# alignment-then-amplification decomposition, and escape before alignment at moderate s0
taus = np.linspace(0, 40, 400001)
Zs = Zfun_vec(taus, Mstar, u0, v0)
w1 = al[0] ** 2 * np.exp(2 * st[0] * taus) / Zs                       # weight of the leading mode


def escape_and_alignment(sig0, eps):
    pr_ = rank_one_prediction(Mstar, u0, v0, sig0, 40.0)
    Jv, tv = pr_.sol(taus)
    sig = Zs ** 1.5 / (1 / sig0 + 3 * Jv)
    ia, ie = np.argmax(w1 >= 1 - eps), np.argmax(sig >= eta * sig_star[0])
    return tv[ia], tv[ie], sig[ia] ** (1 / 3)


for eps in (1e-2, 1e-3):
    t_al, t_es, s_al = escape_and_alignment(1e-30, eps)
    lhs, rhs = t_es - t_al, 1 / (sig_star[0] * s_al)
    check(f"small s0: after alignment (w_1 = 1 - {eps:g}), the remaining time is 1/(sigma*_1 s_align) (1 + O(eps))",
          t_al < t_es and abs(lhs - rhs) < 2 * eps * rhs, f"{lhs:.4e} vs {rhs:.4e}")
t_al, t_es, _ = escape_and_alignment(1e-12, 1e-3)
check("moderate s0 (1e-4): the channel escapes before it is aligned to w_1 = 1 - 1e-3", t_es < t_al,
      f"t_esc = {t_es:.1f}, t_align = {t_al:.1f}")

# random high-dimensional initialization: sigma*_1 I = log(1/alpha_1^2)/2 + O_P(1), at most log 2 + o(1) above
for d in (10 ** 2, 10 ** 3, 10 ** 4):
    vals = []
    for trial in range(200):
        p_, q_ = rng.standard_normal(3) / np.sqrt(d), rng.standard_normal(3) / np.sqrt(d)
        al_, be_ = (p_ + q_) / 2, (p_ - q_) / 2
        g2_ = 1 - np.sum(al_ ** 2 + be_ ** 2)
        Ib = quad(lambda tau: Zfun(tau, sig_star, al_, be_, g2_) ** -0.5, 0, np.inf, limit=400)[0]
        vals.append(sig_star[0] * Ib - 0.5 * np.log(1 / al_[0] ** 2))
    v = np.array(vals)
    q05, q50, q95 = np.quantile(v, [0.05, 0.5, 0.95])
    print(f"  d = {d}: sigma*_1 I - log(1/alpha_1^2)/2: median {q50:.2f}, 5% {q05:.2f}, 95% {q95:.2f}")
    check(f"random start, d = {d}: sigma*_1 I = log(1/alpha_1^2)/2 + O(1), never above it by more than log 2 + 0.05",
          -1.0 < q05 and q95 < np.log(2) + 0.05 and v.max() < np.log(2) + 0.05)

# ---------------------------------------------------------------------------
section("5. Local geometry: Hessian spectra at the critical points M_S")


def hessian(flow, th, h=1e-5):
    n = len(th)
    J = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = h
        J[:, i] = (flow.rhs(0, th + e) - flow.rhs(0, th - e)) / (2 * h)
    return -(J + J.T) / 2


Uf, sf, Vft = np.linalg.svd(Mstar)
Uf, Vf = Uf, Vft.T                          # complete singular bases of M*
s_full = np.zeros(max(d12, d34))
s_full[:3] = sig_star


def predicted_rates(S):
    """Nonzero eigenvalues of the linearized flow (= minus Hessian) at M_S, from the section."""
    rates = []
    for i in S:
        si = sig_star[i]
        rates.append(-3 * si ** (4 / 3))
        for j in range(3):                                    # teacher modes outside S
            if j not in S:
                rates += [si ** (1 / 3) * (sig_star[j] - si), -si ** (1 / 3) * (sig_star[j] + si)]
        rates += [-si ** (4 / 3)] * ((d12 - 3) + (d34 - 3))   # left and right null directions of M*
    for a, i in enumerate(S):
        for j in S[a + 1:]:
            kij = sig_star[i] ** (4 / 3) + (sig_star[i] * sig_star[j]) ** (2 / 3) + sig_star[j] ** (4 / 3)
            rates += [-kij, -kij]
    return np.sort(rates)


for S in ([1], [0, 2], [1, 2]):
    k = len(S)
    s = sig_star[S] ** (1 / 3)
    WL_, WR_ = orth(DL, k), orth(DR, k)
    th = white.pack(*balanced(Uf[:, S], Vf[:, S], WL_, WR_, s))
    ev_ = np.linalg.eigvalsh(-hessian(white, th))
    nonzero = np.sort(ev_[np.abs(ev_) > 1e-6])
    pr_ = predicted_rates(S)
    ok = len(nonzero) == len(pr_) and np.allclose(nonzero, pr_, atol=1e-5)
    n_unst = int(np.sum(nonzero > 0))
    check(f"S = {[i + 1 for i in S]}: nonzero linear rates match the predictions ({n_unst} unstable)", ok,
          f"{len(nonzero)} vs {len(pr_)} rates")

# ---------------------------------------------------------------------------
section("6. Plateaus of a rank-two student: homogeneity and the deflated normal form")
U2i, V2i, WL2, WR2 = orth(d12, 2), orth(d34, 2), orth(DL, 2), orth(DR, 2)
dirs = np.array([1.0, 0.8])
first = []
for s_init in (1e-2, 3e-3, 1e-3):
    ev1 = lambda t, th: (Us[:, 0] @ (lambda A_, B_, C_: A_ @ C_ @ B_.T)(*white.unpack(th)) @ Vs[:, 0]) - 0.5 * sig_star[0]
    ev1.terminal, ev1.direction = True, 1
    r = white.run(*balanced(U2i, V2i, WL2, WR2, s_init * dirs), 1e5, events=ev1, rtol=1e-10, atol=1e-14)
    first.append(s_init * r.t_events[0][0])
print(f"  s0 * t_1 for s0 = 1e-2, 3e-3, 1e-3: {np.round(first, 4)}")
check("homogeneity: s0 t_1 converges as s0 -> 0 (scale invariance of the early phase)",
      abs(first[2] - first[1]) < abs(first[1] - first[0]) and abs(first[2] - first[1]) < 0.02 * first[2])

# deflated normal form for the second (dormant) channel
s_init = 3e-3
coup = lambda th, i: Us[:, i] @ (lambda A_, B_, C_: A_ @ C_ @ B_.T)(*white.unpack(th)) @ Vs[:, i]
ev_a = lambda t, th: coup(th, 0) - 0.995 * sig_star[0]
ev_a.terminal, ev_a.direction = True, 1
r1 = white.run(*balanced(U2i, V2i, WL2, WR2, s_init * dirs), 1e5, events=ev_a, rtol=1e-10, atol=1e-14)
ta, tha = r1.t_events[0][0], r1.y_events[0][0]
ev_b = lambda t, th: np.linalg.svd((lambda A_, B_, C_: A_ @ C_ @ B_.T)(*white.unpack(th)), compute_uv=False)[1] - 0.5 * sig_star[1]
ev_b.terminal, ev_b.direction = True, 1
r2 = solve_ivp(white.rhs, (ta, 1e5), tha, method="DOP853", rtol=1e-10, atol=1e-14, events=ev_b)
tb = r2.t_events[0][0]
A_, B_, C_ = white.unpack(tha)
Ma = A_ @ C_ @ B_.T
Ua, sva, Va = svd_r(Ma, 2)
Ydefl = (np.eye(d12) - np.outer(Ua[:, 0], Ua[:, 0])) @ Mstar @ (np.eye(d34) - np.outer(Va[:, 0], Va[:, 0]))
t_pred = t_escape_quadrature(sva[1], Ydefl, Ua[:, 1], Va[:, 1], 0.5 * sig_star[1])
print(f"  at t_a = {ta:.1f}: sigma_2 = {sva[1]:.2e} (sigma_1 = {sva[0]:.3f}); "
      f"remaining time simulated {tb - ta:.2f}, normal form {t_pred:.2f}")
check("plateau after the first acquisition: deflated rank-one normal form predicts the second escape",
      abs((tb - ta) - t_pred) < 0.03 * (tb - ta))

# ---------------------------------------------------------------------------
section("7. Near the aligned manifold and near the optimum")
n3 = 3
Ut3, Vt3 = orth(n3, n3), orth(n3, n3)
st3 = np.array([1.5, 0.9, 0.4])
Mst3 = Ut3 @ np.diag(st3) @ Vt3.T
F3 = lambda M: Kop(M, Mst3 - M)
Jm = np.zeros((9, 9))
for i in range(9):
    E = np.zeros(9)
    E[i] = 1e-6
    Jm[:, i] = (F3(Mst3 + E.reshape(3, 3)) - F3(Mst3 - E.reshape(3, 3))).ravel() / 2e-6
kst = st3[:, None] ** (4 / 3) + (st3[:, None] * st3[None, :]) ** (2 / 3) + st3[None, :] ** (4 / 3)
check("near the optimum: the linearized closed flow has eigenvalues -kappa*_ij",
      np.allclose(np.sort(np.linalg.eigvals(Jm).real), np.sort(-kst.ravel()), atol=1e-6))

# rates of xi^+ and xi^- near an aligned configuration with amplitudes off the teacher values
mu = np.array([1.0, 0.3, 0.6])
eps_ang = 1e-5
skew = lambda a, i, j: (lambda K: K - K.T)(np.zeros((n3, n3)) + a * np.eye(n3)[:, [i]] @ np.eye(n3)[[j], :])
ok = True
for (i, j) in ((0, 1), (0, 2), (2, 1)):
    for sign in (+1, -1):
        X, Y = skew(eps_ang, i, j), skew(sign * eps_ang, i, j)
        U_, V_ = Ut3 @ expm(X), Vt3 @ expm(Y)
        M = U_ @ np.diag(mu) @ V_.T
        Mdot = F3(M)
        hh = 1e-7

        def match(Xn, X0):                 # reorder and re-sign singular vectors to follow X0
            idx = np.argmax(np.abs(X0.T @ Xn), axis=1)
            Xn = Xn[:, idx]
            return Xn * np.sign(np.sum(Xn * X0, axis=0))

        Up, _, Vp = svd_r(M + hh * Mdot, 3)
        Um, _, Vm = svd_r(M - hh * Mdot, 3)
        Udot = (match(Up, U_) - match(Um, U_)) / (2 * hh)
        Vdot = (match(Vp, V_) - match(Vm, V_)) / (2 * hh)
        x_dot, y_dot = (U_.T @ Udot)[i, j], (V_.T @ Vdot)[i, j]
        kij = mu[i] ** (4 / 3) + (mu[i] * mu[j]) ** (2 / 3) + mu[j] ** (4 / 3)
        if sign > 0:
            rate, pred_rate = (x_dot + y_dot) / (2 * eps_ang), -kij * (st3[i] - st3[j]) / (mu[i] - mu[j])
        else:
            rate, pred_rate = (x_dot - y_dot) / (2 * eps_ang), -kij * (st3[i] + st3[j]) / (mu[i] + mu[j])
        ok &= abs(rate - pred_rate) < 1e-3 * abs(pred_rate)
        print(f'    pair {(i + 1, j + 1)}, {"xi+" if sign > 0 else "xi-"}: rate {rate:+.5f}, predicted {pred_rate:+.5f}')
check("near-aligned rates of xi^+ (gaps) and xi^- (sums), including the mode-swapped pair", ok)

# ---------------------------------------------------------------------------
section("8. Non-whitened inputs: early phase driven by the moment matrix")
u0n, v0n = unit(rng.standard_normal(d12)), unit(rng.standard_normal(d34))
Mmom = S12 @ Mstar @ S34
stm, alm, bem, g2m, _, _ = dilation_coefficients(Mmom, u0n, v0n)
Im = quad(lambda tau: Zfun(tau, stm, alm, bem, g2m) ** -0.5, 0, np.inf, limit=400)[0]
vals = []
for s_init in (1e-2, 3e-3, 1e-3):
    level = 0.05
    evn = lambda t, th: np.linalg.norm(nonwhite.unpack(th)[2]) - level ** (1 / 3)
    evn.terminal, evn.direction = True, 1
    r = nonwhite.run(s_init * np.outer(u0n, wl), s_init * np.outer(v0n, wr), s_init * np.outer(wl, wr), 1e5,
                     events=evn, rtol=1e-10, atol=1e-14)
    vals.append(s_init * r.t_events[0][0])
print(f"  I(moment matrix) = {Im:.4f}; s0 * t_esc = {np.round(vals, 4)}")
check("non-whitened rank-one channel: s0 t_esc -> I computed from Sigma_12 M* Sigma_34",
      abs(vals[-1] - Im) < abs(vals[0] - Im) and abs(vals[-1] - Im) < 0.01 * Im)

# ---------------------------------------------------------------------------
section("9. Rank-one channel of the seven-core tree: tensor power flow")
d, m = 4, 2
a_ = [orth(d, 2) for _ in range(4)]
sig_t = np.array([2.0, 1.2])
Tstar = sum(sig_t[k] * np.einsum("i,j,k,l->ijkl", a_[0][:, k], a_[1][:, k], a_[2][:, k], a_[3][:, k]) for k in range(2))
shapes7 = [(d, m), (d, m), (m, m, 2), (2, 2), (m, m, 2), (d, m), (d, m)]   # L1, L2, Gu, C, Gv, L3, L4
sizes7 = [int(np.prod(s)) for s in shapes7]
EIN = "ia,jb,abp,pq,cdq,kc,ld->ijkl"
grads_spec = ["ijkl,jb,abp,pq,cdq,kc,ld->ia", "ijkl,ia,abp,pq,cdq,kc,ld->jb", "ijkl,ia,jb,pq,cdq,kc,ld->abp",
              "ijkl,ia,jb,abp,cdq,kc,ld->pq", "ijkl,ia,jb,abp,pq,kc,ld->cdq", "ijkl,ia,jb,abp,pq,cdq,ld->kc",
              "ijkl,ia,jb,abp,pq,cdq,kc->ld"]


def unpack7(th):
    parts = np.split(th, np.cumsum(sizes7)[:-1])
    return [p.reshape(s) for p, s in zip(parts, shapes7)]


def rhs7(_, th):
    cores = unpack7(th)
    R = np.einsum(EIN, *cores, optimize=True) - Tstar
    out = []
    for g, spec in enumerate(grads_spec):
        others = [c for i, c in enumerate(cores) if i != g]
        out.append(-np.einsum(spec, R, *others, optimize=True).ravel())
    return np.concatenate(out)


e1 = np.eye(m)[0]
ells = [unit(rng.standard_normal(d)) for _ in range(4)]
s7 = 0.45
cores0 = [s7 * np.outer(ells[0], e1), s7 * np.outer(ells[1], e1), s7 * np.einsum("a,b,p->abp", e1, e1, np.eye(2)[0]),
          s7 * np.outer(np.eye(2)[0], np.eye(2)[0]), s7 * np.einsum("a,b,p->abp", e1, e1, np.eye(2)[0]),
          s7 * np.outer(ells[2], e1), s7 * np.outer(ells[3], e1)]
T7 = 30.0
full7 = solve_ivp(rhs7, (0, T7), np.concatenate([c.ravel() for c in cores0]), method="DOP853", rtol=1e-11,
                  atol=1e-13, dense_output=True)


def red_rhs(_, y):                               # reduced system: 4 leaf vectors and the common scale s
    ls, s = [y[4 * i:4 * i + 4] for i in range(4)], y[-1]
    rho = np.einsum("ijkl,i,j,k,l->", Tstar, *ls)
    out = []
    for g in range(4):
        spec = ["ijkl,j,k,l->i", "ijkl,i,k,l->j", "ijkl,i,j,l->k", "ijkl,i,j,k->l"][g]
        v = np.einsum(spec, Tstar, *[ls[i] for i in range(4) if i != g])
        out.append(s ** 5 * (v - (v @ ls[g]) * ls[g]))
    return np.concatenate(out + [[s ** 6 * (rho - s ** 7)]])


red = solve_ivp(red_rhs, (0, T7), np.concatenate(ells + [[s7]]), method="DOP853", rtol=1e-11, atol=1e-13,
                dense_output=True)
err7, rank_err = 0.0, 0.0
for t in np.linspace(0, T7, 61):
    T_full = np.einsum(EIN, *unpack7(full7.sol(t)), optimize=True)
    y = red.sol(t)
    T_red = y[-1] ** 7 * np.einsum("i,j,k,l->ijkl", *[y[4 * i:4 * i + 4] for i in range(4)])
    err7 = max(err7, np.abs(T_full - T_red).max())
    rank_err = max(rank_err, np.linalg.svd(T_full.reshape(d * d, d * d), compute_uv=False)[1])
check("seven-core rank-one channel: T = s^7 l1 x l2 x l3 x l4, with l_a' = s^5 (I - l_a l_a^T) T*(l_-a)",
      err7 < 1e-8 and rank_err < 1e-10, f"max error {err7:.1e}")

# alignment time in tau, rank-one teacher, equal overlaps: m = 4 groups of dimension d (seven-core tree),
# p0 = d^{-1/2}, versus m = 2 groups that each gather two legs of dimension d (three-core model), P0 = 1/d
times = []
for dd in (16, 64, 256):
    p0 = 1 / np.sqrt(dd)
    P0 = p0 ** 2
    t_mat = quad(lambda P: 1 / (P * (1 - P ** 2)), P0, 0.9)[0]
    t_ten = quad(lambda p: 1 / (p ** 3 * (1 - p ** 2)), p0, 0.9)[0]
    times.append((dd, t_mat, t_ten))
print("  (d, tau_align matrix, tau_align tensor):", [(dd, round(a, 2), round(b, 1)) for dd, a, b in times])
check("m = 4: tau_align ~ d_g/2 (sigma = 1); m = 2 with d_g = d^2: tau_align ~ log d",
      all(abs(b / (dd / 2) - 1) < 0.25 and abs(a - np.log(dd)) < 1.0 for dd, a, b in times))

# ---------------------------------------------------------------------------
n_fail = sum(not ok for _, ok in RESULTS)
print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed.")
for name, ok in RESULTS:
    if not ok:
        print(f"  FAILED: {name}")
sys.exit(1 if n_fail else 0)
