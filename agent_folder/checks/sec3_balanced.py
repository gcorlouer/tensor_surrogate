#!/usr/bin/env python3
"""Checks of the balanced-manifold part of Section 3 ("Training dynamics of TTNs") of ttns_surrogate.tex.

Exact SymPy checks of the algebra (balanced factors, closed equation for M, loss decay,
spectral form of the tangent operator, aligned scalar dynamics), with exact rational
orthonormal frames, and SciPy checks of the dynamics (aligned modes, escape time,
convergence rate, decay of extra modes, closed equation against the factor flow,
conservation of the bond rank).

    python3 sec3_balanced.py
"""
import random
import sys

import numpy as np
import sympy as sp
from scipy.integrate import quad, solve_ivp

random.seed(2)
RESULTS = []


def check(name, ok):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}", flush=True)


def section(title):
    print(f"\n== {title}", flush=True)


def is_zero(M):
    entries = M if isinstance(M, sp.MatrixBase) else [M]
    return all(sp.expand(e) == 0 for e in entries)


def frob(X, Y):
    return sum(X.multiply_elementwise(Y))


def ex(M):
    """Expand every entry, so that later products do not build nested expressions."""
    return M.applyfunc(sp.expand)


def cayley(n):
    """Exact rational orthogonal matrix (I - K)(I + K)^{-1}, K skew-symmetric."""
    K = sp.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            v = sp.Rational(random.randint(-3, 3), random.randint(1, 3))
            K[i, j], K[j, i] = v, -v
    Q = (sp.eye(n) - K) * (sp.eye(n) + K).inv()
    assert is_zero(Q.T * Q - sp.eye(n))
    return Q


# Dimensions of Section 3: d12 = 6, d34 = 4, bond dimensions D_L = 2, D_R = 3; k = rank C.
d12, d34, DL, DR, k = 6, 4, 2, 3, 2
Ufull, Vfull, WLfull, WRfull = cayley(d12), cayley(d34), cayley(DL), cayley(DR)
U, V, WL, WR = Ufull[:, :k], Vfull[:, :k], WLfull[:, :k], WRfull[:, :k]
s = sp.symbols(f"s1:{k + 1}", positive=True)
S = sp.diag(*s)
A, C, B = ex(U * S * WL.T), ex(WL * S * WR.T), ex(V * S * WR.T)
M = ex(A * C * B.T)

section("Balanced factors (exact, symbolic s_i)")
check("A = U S W_L^T, C = W_L S W_R^T, B = V S W_R^T are balanced: A^T A = C C^T, B^T B = C^T C",
      is_zero(A.T * A - C * C.T) and is_zero(B.T * B - C.T * C))
check("M = A C B^T = U S^3 V^T, so sigma_i = s_i^3", is_zero(M - U * S ** 3 * V.T))
L1, R1 = ex(U * S ** 2 * U.T), ex(V * S ** 2 * V.T)  # (M M^T)^{1/3}, (M^T M)^{1/3}
L2, R2 = ex(U * S ** 4 * U.T), ex(V * S ** 4 * V.T)  # (M M^T)^{2/3}, (M^T M)^{2/3}
Lh, Rh = ex(U * S * U.T), ex(V * S * V.T)            # (M M^T)^{1/6}, (M^T M)^{1/6}
check("A A^T = U S^2 U^T with (U S^2 U^T)^3 = M M^T; B B^T = V S^2 V^T with (V S^2 V^T)^3 = M^T M",
      is_zero(A * A.T - L1) and is_zero(L1 ** 3 - M * M.T) and is_zero(B * B.T - R1) and is_zero(R1 ** 3 - M.T * M))
check("A C C^T A^T = (A A^T)^2 and B C^T C B^T = (B B^T)^2 on the balanced manifold",
      is_zero(A * C * C.T * A.T - (A * A.T) ** 2) and is_zero(B * C.T * C * B.T - (B * B.T) ** 2))

section("Closed equation for M and loss decay (generic symbolic R = grad_M L)")
Rg = sp.Matrix(d12, d34, lambda i, j: sp.Symbol(f"R{i}_{j}"))
Adot, Bdot, Cdot = ex(-Rg * ex(B * C.T)), ex(-Rg.T * ex(A * C)), ex(-A.T * Rg * B)
Mdot = ex(Adot * ex(C * B.T) + A * ex(Cdot * B.T) + ex(A * C) * Bdot.T)
check("Mdot = -(MM^T)^{2/3} R - (MM^T)^{1/3} R (M^TM)^{1/3} - R (M^TM)^{2/3}",
      is_zero(Mdot - (-L2 * Rg - L1 * Rg * R1 - Rg * R2)))
dLdt = -(frob(Adot, Adot) + frob(Bdot, Bdot) + frob(Cdot, Cdot))
check("dL/dt = -||(MM^T)^{1/3}R||^2 - ||(MM^T)^{1/6} R (M^TM)^{1/6}||^2 - ||R (M^TM)^{1/3}||^2",
      is_zero(dLdt + frob(L1 * Rg, L1 * Rg) + frob(Lh * Rg * Rh, Lh * Rg * Rh) + frob(Rg * R1, Rg * R1)))


def Kop(X):
    return A * C * C.T * A.T * X + A * A.T * X * B * B.T + X * B * C.T * C * B.T


left = [s[i] ** 2 if i < k else 0 for i in range(d12)]   # sigma_i^{2/3} of u_i (0 beyond the rank)
right = [s[j] ** 2 if j < k else 0 for j in range(d34)]  # sigma_j^{2/3} of v_j
check("K[u_i v_j^T] = kappa_ij u_i v_j^T for complete singular bases, kappa_ij = sigma_i^{4/3} + "
      "sigma_i^{2/3} sigma_j^{2/3} + sigma_j^{4/3} (sigma = 0 beyond the rank)",
      all(is_zero(Kop(Ufull[:, i] * Vfull[:, j].T)
                  - (left[i] ** 2 + left[i] * right[j] + right[j] ** 2) * Ufull[:, i] * Vfull[:, j].T)
          for i in range(d12) for j in range(d34)))
check("u_i^T Mdot v_i = -3 sigma_i^{4/3} u_i^T R v_i for the nonzero singular values",
      all(is_zero((Ufull[:, i].T * Mdot * Vfull[:, i])[0] + 3 * s[i] ** 4 * (Ufull[:, i].T * Rg * Vfull[:, i])[0])
          for i in range(k)))
check("K vanishes on u_i v_j^T when both singular values are zero",
      all(is_zero(Kop(Ufull[:, i] * Vfull[:, j].T)) for i in range(k, d12) for j in range(k, d34)))

section("Aligned modes (exact): the factors keep their form")
sst = sp.symbols(f"t1:{k + 1}", nonnegative=True)        # teacher singular values sigma*_i
# Eigenvalues of Sigma_12 on u_i and of Sigma_34 on v_i: symbolic on the k aligned modes; numbers
# elsewhere, where they cannot enter the result.
lL = list(sp.symbols(f"lL1:{k + 1}", positive=True)) + [sp.Integer(i + 2) for i in range(d12 - k)]
lR = list(sp.symbols(f"lR1:{k + 1}", positive=True)) + [sp.Integer(i + 3) for i in range(d34 - k)]
S12, S34 = ex(Ufull * sp.diag(*lL) * Ufull.T), ex(Vfull * sp.diag(*lR) * Vfull.T)
Mstar = ex(U * sp.diag(*sst) * V.T)
Ral = ex(ex(S12 * (M - Mstar)) * S34)
Sdot = sp.diag(*[lL[i] * lR[i] * s[i] ** 2 * (sst[i] - s[i] ** 3) for i in range(k)])
check("Adot = U Sdot W_L^T, Cdot = W_L Sdot W_R^T, Bdot = V Sdot W_R^T with sdot_i = l^L_i l^R_i s_i^2 (sigma*_i - s_i^3)",
      is_zero(-Ral * ex(B * C.T) - U * Sdot * WL.T) and is_zero(-A.T * ex(Ral * B) - WL * Sdot * WR.T)
      and is_zero(-Ral.T * ex(A * C) - V * Sdot * WR.T))

# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------
def factor_flow(A0, B0, C0, grad_M, T):
    shapes = [A0.shape, B0.shape, C0.shape]
    sizes = [A0.size, B0.size, C0.size]

    def unpack(th):
        parts = np.split(th, np.cumsum(sizes)[:-1])
        return [p.reshape(sh) for p, sh in zip(parts, shapes)]

    def rhs(_, th):
        A_, B_, C_ = unpack(th)
        Rm = grad_M(A_ @ C_ @ B_.T)
        return np.concatenate([(-Rm @ B_ @ C_.T).ravel(), (-Rm.T @ A_ @ C_).ravel(), (-A_.T @ Rm @ B_).ravel()])

    th0 = np.concatenate([A0.ravel(), B0.ravel(), C0.ravel()])
    sol = solve_ivp(rhs, (0, T), th0, method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True)
    return sol, unpack


def psd_power(G, p, rtol=1e-10):
    """Power of a PSD matrix; eigenvalues below rtol * max are rounding noise on the kernel and set to 0
    (their fractional powers would otherwise be of order (1e-16)^p)."""
    lam, Q = np.linalg.eigh((G + G.T) / 2)
    lam = np.where(lam > rtol * lam.max(), lam, 0.0)
    return (Q * lam ** p) @ Q.T


f = lambda X: np.array(X, dtype=float)
Un, Vn, WLn, WRn = f(U), f(V), f(WL), f(WR)
Ufn, Vfn = f(Ufull), f(Vfull)

section("Numerics: aligned modes, extra mode, escape time, convergence rate")
lam_L = np.array([1.5, 0.7, 1.0, 2.0, 0.5, 1.2])
lam_R = np.array([0.8, 1.3, 0.6, 1.1])
S12n, S34n = (Ufn * lam_L) @ Ufn.T, (Vfn * lam_R) @ Vfn.T
sig_star = np.array([1.0, 0.0])            # mode 2 has no teacher counterpart
s0 = np.array([0.05, 0.6])
Mstar_n = Un @ np.diag(sig_star) @ Vn.T
sol, unpack = factor_flow(Un @ np.diag(s0) @ WLn.T, Vn @ np.diag(s0) @ WRn.T, WLn @ np.diag(s0) @ WRn.T,
                          lambda Mm: S12n @ (Mm - Mstar_n) @ S34n, T=60.0)
ts = np.linspace(0, 60, 601)
c = lam_L[:k] * lam_R[:k]
scalar = solve_ivp(lambda t, y: c * y ** 2 * (sig_star - y ** 3), (0, 60), s0, method="DOP853",
                   rtol=1e-12, atol=1e-14, dense_output=True)
frame_dev, sv_err, extra_err = 0.0, 0.0, 0.0
for t in ts:
    A_, B_, C_ = unpack(sol.sol(t))
    D_A, D_B, D_C = Un.T @ A_ @ WLn, Vn.T @ B_ @ WRn, WLn.T @ C_ @ WRn
    frame_dev = max(frame_dev, *(np.abs(Dm - np.diag(np.diag(Dm))).max() for Dm in (D_A, D_B, D_C)),
                    np.abs(A_ - Un @ D_A @ WLn.T).max(), np.abs(C_ - WLn @ D_C @ WRn.T).max())
    s_t = scalar.sol(t)
    sv_err = max(sv_err, np.abs(np.diag(D_A) - s_t).max(), np.abs(np.diag(D_C) - s_t).max(), np.abs(np.diag(D_B) - s_t).max())
    sigma2_exact = ((s0[1] ** 3) ** (-4 / 3) + 4 * c[1] * t) ** (-3 / 4)
    extra_err = max(extra_err, abs(np.diag(D_A)[1] ** 3 - sigma2_exact))
print(f"  max frame deviation {frame_dev:.1e}; max |s_i(t) - scalar ODE| {sv_err:.1e}; "
      f"max |sigma_2(t) - closed form| {extra_err:.1e}")
check("aligned: frames fixed, the three factors share s_i(t), which solves sdot = l^L l^R s^2 (sigma* - s^3)",
      frame_dev < 1e-9 and sv_err < 1e-6)
check("extra mode (sigma* = 0): sigma(t) = (sigma(0)^{-4/3} + 4 l^L l^R t)^{-3/4}", extra_err < 1e-9)

ratios = []
for s_init in (1e-2, 1e-3, 1e-4):
    s_half = 0.5 ** (1 / 3)  # time to reach sigma = sigma*/2, with sigma* = 1 and l^L l^R = 1
    t_half, _ = quad(lambda x: 1 / (x ** 2 * (1 - x ** 3)), s_init, s_half, epsabs=1e-13, epsrel=1e-13, limit=200)
    ratios.append(t_half * s_init)
print(f"  sigma* s(0) t_escape for s(0) = 1e-2, 1e-3, 1e-4: {[round(r, 5) for r in ratios]}")
check("escape time ~ 1/(sigma* s(0)) as s(0) -> 0", all(abs(r - 1) < 2 * si for r, si in zip(ratios, (1e-2, 1e-3, 1e-4))))

sigma_star, cc = 2.0, 1.3
# Integrate delta = sigma* - sigma itself, so that its relative accuracy survives as delta -> 0.
conv = solve_ivp(lambda t, y: -3 * cc * (sigma_star - y) ** (4 / 3) * y, (0, 5), [0.1 * sigma_star],
                 method="DOP853", rtol=1e-13, atol=1e-300, dense_output=True)
t1, t2 = 4.0, 5.0
rate = -(np.log(conv.sol(t2)[0]) - np.log(conv.sol(t1)[0])) / (t2 - t1)
print(f"  convergence rate {rate:.6f} vs 3 l^L l^R sigma*^(4/3) = {3 * cc * sigma_star ** (4 / 3):.6f}")
check("exponential convergence at rate 3 l^L l^R sigma*^{4/3}", abs(rate / (3 * cc * sigma_star ** (4 / 3)) - 1) < 1e-4)

section("Numerics: closed equation for M against the factor flow (generic quadratic loss)")
rng = np.random.default_rng(0)


def orth(n, kk):
    return np.linalg.qr(rng.standard_normal((n, n)))[0][:, :kk]


def closed_rhs(Mm, grad_M):
    Rm = grad_M(Mm)
    Lc, Rc = psd_power(Mm @ Mm.T, 1 / 3), psd_power(Mm.T @ Mm, 1 / 3)
    return -Lc @ Lc @ Rm - Lc @ Rm @ Rc - Rm @ Rc @ Rc


for (n, m, DLx, DRx, kk) in ((3, 3, 3, 3, 3), (d12, d34, DL, DR, k)):
    Gm = rng.standard_normal((n * m, n * m))
    Gamma = Gm @ Gm.T / (n * m) + 0.1 * np.eye(n * m)   # Sigma_[4] of an arbitrary (dependent) law
    Ms = rng.standard_normal((n, m))
    grad = lambda Mm: (Gamma @ (Mm - Ms).ravel()).reshape(n, m)
    sv = rng.uniform(0.5, 1.2, kk)
    Ux, Vx, WLx, WRx = orth(n, kk), orth(m, kk), orth(DLx, kk), orth(DRx, kk)
    A0, B0, C0 = Ux @ np.diag(sv) @ WLx.T, Vx @ np.diag(sv) @ WRx.T, WLx @ np.diag(sv) @ WRx.T
    T = 3.0
    full_rank = kk == min(n, m)
    solf, unp = factor_flow(A0, B0, C0, grad, T)
    # The closed equation is integrated only at full rank, where it is locally Lipschitz.
    solc = solve_ivp(lambda t, y: closed_rhs(y.reshape(n, m), grad).ravel(), (0, T), (A0 @ C0 @ B0.T).ravel(),
                     method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True) if full_rank else None
    dev, bal, root, rhs_dev = 0.0, 0.0, 0.0, 0.0
    for t in np.linspace(0, T, 301):
        A_, B_, C_ = unp(solf.sol(t))
        Mf = A_ @ C_ @ B_.T
        if full_rank:
            dev = max(dev, np.abs(Mf - solc.sol(t).reshape(n, m)).max())
        Rm = grad(Mf)
        Mdot_f = (-Rm @ B_ @ C_.T) @ C_ @ B_.T + A_ @ (-A_.T @ Rm @ B_) @ B_.T + A_ @ C_ @ (-Rm.T @ A_ @ C_).T
        rhs_dev = max(rhs_dev, np.abs(Mdot_f - closed_rhs(Mf, grad)).max())
        bal = max(bal, np.abs(A_.T @ A_ - C_ @ C_.T).max(), np.abs(B_.T @ B_ - C_.T @ C_).max())
        root = max(root, np.abs(A_ @ A_.T - psd_power(Mf @ Mf.T, 1 / 3)).max(),
                   np.abs(B_ @ B_.T - psd_power(Mf.T @ Mf, 1 / 3)).max())
    label = "full rank" if full_rank else f"rank {kk} < {min(n, m)}"
    print(f"  {label}: balance drift {bal:.1e}, max |AA^T - (MM^T)^(1/3)| = {root:.1e}, "
          f"max |Mdot - closed right-hand side| = {rhs_dev:.1e}"
          + (f", max |M_factor - M_closed| = {dev:.1e}" if full_rank else ""))
    check(f"{label}: along the factor flow, balance holds, AA^T = (MM^T)^(1/3), BB^T = (M^TM)^(1/3), "
          f"and Mdot equals the closed right-hand side", bal < 1e-9 and root < 1e-6 and rhs_dev < 1e-6)
    if full_rank:
        check(f"{label}: integrating the closed equation reproduces the factor flow", dev < 1e-7)

section("Numerics: the bond rank is conserved on the balanced manifold")
Ms = (orth(d12, 2) * np.array([1.5, 0.8])) @ orth(d34, 2).T      # rank-2 teacher, generic position
grad_w = lambda Mm: Mm - Ms                                      # whitened loss
u1, v1, wl1, wr1 = orth(d12, 1), orth(d34, 1), orth(DL, 1), orth(DR, 1)
A0, B0, C0 = 0.3 * u1 @ wl1.T, 0.3 * v1 @ wr1.T, 0.3 * wl1 @ wr1.T   # balanced, rank one
T = 600.0


def sigma_2(th, unr):
    A_, B_, C_ = unr(th)
    return np.linalg.svd(A_ @ C_ @ B_.T, compute_uv=False)[1]


for name, pert in (("balanced, rank one", 0.0), ("perturbed by 3e-2", 3e-2)):
    A1 = A0 + pert * rng.standard_normal(A0.shape)
    B1 = B0 + pert * rng.standard_normal(B0.shape)
    C1 = C0 + pert * rng.standard_normal(C0.shape)
    solr, unr = factor_flow(A1, B1, C1, grad_w, T)
    sig2 = [sigma_2(solr.sol(t), unr) for t in np.linspace(0, T, 601)]
    A_, B_, C_ = unr(solr.sol(T))
    final_sv = np.linalg.svd(A_ @ C_ @ B_.T, compute_uv=False)[:2]
    print(f"  {name}: max sigma_2(t) = {max(sig2):.1e}; final singular values {np.round(final_sv, 6)} "
          f"(teacher 1.5, 0.8)")
    if pert == 0.0:
        check("exactly balanced rank-one initialization: rank M(t) stays one; the second teacher mode is never learned",
              max(sig2) < 1e-9 and abs(final_sv[0] - 1.5) < 1e-6)
    else:
        check("a small unbalanced perturbation lets the second mode be learned", abs(final_sv[1] - 0.8) < 1e-3)

# ---------------------------------------------------------------------------
n_fail = sum(not ok for _, ok in RESULTS)
print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed.")
for name, ok in RESULTS:
    if not ok:
        print(f"  FAILED: {name}")
sys.exit(1 if n_fail else 0)
