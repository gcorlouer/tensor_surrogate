#!/usr/bin/env python3
"""Exact checks of Section 3 ("Training dynamics of TTNs") of ttns_surrogate.tex.

SymPy port of sec3_gradient_flow.wl, written while the Mathematica licence was
unavailable. Every check is an exact identity: polynomials are expanded with
rational coefficients, and expectations are exact, either by enumerating a
finite support or from Gaussian moments. Labels such as (d2-white-flow) refer
to the equation labels in ttns_surrogate.tex. Notation follows the paper:
Sigma_12 = E[z z^T], Sigma_34 = E[w w^T], Sigma_[4] = E[(z (x) w)(z (x) w)^T],
and vec flattens row by row (lexicographic order).

    python3 sec3_gradient_flow.py
"""
import itertools
import random
import sys
import time

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

random.seed(1)
Q = sp.Rational

# Dimensions are distinct where possible, so that transposition errors show up.
d1, d2, d3, d4 = 2, 3, 2, 2
DL, DR = 2, 3
dims = (d1, d2, d3, d4)
d12, d34 = d1 * d2, d3 * d4
n_x = sum(dims)

RESULTS = []


def check(name, ok):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}", flush=True)


def is_zero(M):
    entries = M if isinstance(M, sp.MatrixBase) else [M]
    return all(sp.expand(e) == 0 for e in entries)


def kron(*Ms):
    out = Ms[0]
    for M in Ms[1:]:
        out = sp.kronecker_product(out, M)
    return out


def vec(M):
    """Row-by-row (lexicographic) vec, as in the 'Vectorized form' paragraph."""
    return M.reshape(M.rows * M.cols, 1)


def unvec(v, rows, cols):
    return v.reshape(rows, cols)


def frob(X, Y):
    return sum(X.multiply_elementwise(Y))


def sym_matrix(name, rows, cols):
    return sp.Matrix(rows, cols, lambda i, j: sp.Symbol(f"{name}{i}_{j}"))


def rand_int_matrix(rows, cols, lo=-2, hi=2):
    return sp.Matrix(rows, cols, lambda i, j: random.randint(lo, hi))


def section(title):
    print(f"\n== {title}", flush=True)


LEX = list(itertools.product(*map(range, dims)))  # lexicographic order of (i1, i2, i3, i4)

# ---------------------------------------------------------------------------
# Model of Section 3.1
# ---------------------------------------------------------------------------
x = [sp.symbols(f"x{a + 1}_0:{dims[a]}") for a in range(4)]
xs = [v for block in x for v in block]
xb = [sp.Matrix(block) for block in x]
z = kron(xb[0], xb[1])  # x1 (x) x2
w = kron(xb[2], xb[3])  # x3 (x) x4
Xi = kron(*xb)          # x1 (x) x2 (x) x3 (x) x4

core_A = [[[sp.Symbol(f"a{al}_{i1}{i2}") for i2 in range(d2)] for i1 in range(d1)] for al in range(DL)]
core_B = [[[sp.Symbol(f"b{be}_{i3}{i4}") for i4 in range(d4)] for i3 in range(d3)] for be in range(DR)]
C = sym_matrix("c", DL, DR)
A = sp.Matrix(d12, DL, lambda r, al: core_A[al][r // d2][r % d2])
B = sp.Matrix(d34, DR, lambda r, be: core_B[be][r // d4][r % d4])
pA, pB, pC = list(A), list(B), list(C)
params = pA + pB + pC

# Teacher with smaller bond dimensions D*_L = 1, D*_R = 2.
A_t, C_t, B_t = rand_int_matrix(d12, 1), rand_int_matrix(1, 2), rand_int_matrix(d34, 2)
M_star = A_t * C_t * B_t.T
T_star = {(i1, i2, i3, i4): M_star[i1 * d2 + i2, i3 * d4 + i4] for i1, i2, i3, i4 in LEX}

M = A * C * B.T
Delta_theta = M - M_star
D = sym_matrix("D", d12, d34)  # generic coefficient error
e_D = (z.T * D * w)[0]

section("3.1 Forward model")
hL = sp.Matrix([sum(core_A[al][i1][i2] * x[0][i1] * x[1][i2] for i1 in range(d1) for i2 in range(d2))
                for al in range(DL)])
hR = sp.Matrix([sum(core_B[be][i3][i4] * x[2][i3] * x[3][i4] for i3 in range(d3) for i4 in range(d4))
                for be in range(DR)])
g = sum(C[al, be] * hL[al] * hR[be] for al in range(DL) for be in range(DR))
T = {(i1, i2, i3, i4): sum(core_A[al][i1][i2] * C[al, be] * core_B[be][i3][i4]
                           for al in range(DL) for be in range(DR))
     for i1, i2, i3, i4 in LEX}
check("h^L = A^T z, h^R = B^T w   (d2-hL, d2-hR, d2-matrix-forward)",
      is_zero(hL - A.T * z) and is_zero(hR - B.T * w))
check("g = <T, x1 (x) x2 (x) x3 (x) x4>   (d2-output-index, d2-coefficient-tensor)",
      is_zero(g - sum(T[k] * Xi[n] for n, k in enumerate(LEX))))
check("g = z^T A C B^T w   (d2-matrix-forward)", is_zero(g - (z.T * A * C * B.T * w)[0]))
check("T^<{1,2}> (lexicographic) = A C B^T   (d2-matricization)",
      is_zero(sp.Matrix(d12, d34, lambda r, s: T[(r // d2, r % d2, s // d4, s % d4)]) - M))
check("vec(M) = vec(T) and vec(z w^T) = z (x) w = x1 (x) x2 (x) x3 (x) x4 (row-by-row vec)",
      is_zero(vec(M) - sp.Matrix([T[k] for k in LEX])) and is_zero(vec(z * w.T) - kron(z, w))
      and is_zero(kron(z, w) - Xi))
check("Lemma cut at the left bond: F_e = A, H_e = B C^T", is_zero(M - A * (B * C.T).T))


# ---------------------------------------------------------------------------
# Input laws with exact moments
# ---------------------------------------------------------------------------
class Law:
    whitened = independent = cut_independent = False

    def expect(self, expr):
        poly = sp.Poly(sp.expand(expr), *xs)
        return sp.expand(sum(coeff * self.moment(mon) for mon, coeff in poly.terms()))


class DiscreteLaw(Law):
    """Finite support: list of (probability, point in R^9)."""

    def __init__(self, name, atoms):
        self.name, self.atoms, self._cache = name, atoms, {}
        assert sum(p for p, _ in atoms) == 1

    def moment(self, k):
        if k not in self._cache:
            self._cache[k] = sum(p * sp.prod([v ** e for v, e in zip(pt, k)]) for p, pt in self.atoms)
        return self._cache[k]


def product_law(name, blocks):
    atoms = [(sp.prod([c[0] for c in combo]), [v for c in combo for v in c[1]])
             for combo in itertools.product(*blocks)]
    return DiscreteLaw(name, atoms)


class GaussianLaw(Law):
    """X = mu + L g with g ~ N(0, I_9): moments from E[g^n] = (n-1)!!."""
    DF = {0: 1, 2: 1, 4: 3, 6: 15, 8: 105, 10: 945}

    def __init__(self, name, mu, Lc):
        self.name, self.mu, self.Lc = name, sp.Matrix(mu), Lc
        self.g = sp.symbols(f"g0:{n_x}")
        self.lin = [sp.Poly(mu[i] + sum(Lc[i, j] * self.g[j] for j in range(n_x)), *self.g, domain="QQ")
                    for i in range(n_x)]
        self._pcache = {(0,) * n_x: sp.Poly(1, *self.g, domain="QQ")}
        self._cache = {}

    def _prod(self, k):
        if k not in self._pcache:
            i = max(j for j, kj in enumerate(k) if kj > 0)
            km = list(k)
            km[i] -= 1
            self._pcache[k] = self._prod(tuple(km)) * self.lin[i]
        return self._pcache[k]

    def moment(self, k):
        if k not in self._cache:
            tot = sp.Integer(0)
            for mon, coeff in self._prod(tuple(k)).terms():
                if all(n % 2 == 0 for n in mon):
                    tot += coeff * sp.prod([self.DF[n] for n in mon])
            self._cache[k] = tot
        return self._cache[k]


def rademacher_block(d):
    return [(Q(1, 2 ** d), list(s)) for s in itertools.product([-1, 1], repeat=d)]


def homogenized_block(d):
    """xi = (1, u) with u Rademacher: E[xi xi^T] = I but E[xi] = e_1 != 0."""
    return [(Q(1, 2 ** (d - 1)), [1] + list(s)) for s in itertools.product([-1, 1], repeat=d - 1)]


laws = []

L1 = GaussianLaw("L1 indep. N(0,I)", [0] * n_x, sp.eye(n_x))
L1.whitened = L1.independent = L1.cut_independent = True
laws.append(L1)

L2 = product_law("L2 indep. Rademacher", [rademacher_block(d) for d in dims])
L2.whitened = L2.independent = L2.cut_independent = True
laws.append(L2)

L3 = product_law("L3 indep. homogenized (1,u), non-centred", [homogenized_block(d) for d in dims])
L3.whitened = L3.independent = L3.cut_independent = True
laws.append(L3)

# Sigma_a = S_a^2 with S_a symmetric positive definite, so that sqrt(Sigma_a) = S_a exactly.
S = [sp.Matrix([[2, 1], [1, 1]]),
     sp.Matrix([[2, 1, 0], [1, 2, 1], [0, 1, 2]]),
     sp.Matrix([[1, Q(1, 2)], [Q(1, 2), 1]]),
     sp.Matrix([[3, 1], [1, 1]])]
assert all(s.is_positive_definite for s in S)
L4 = GaussianLaw("L4 indep. N(0, S_a^2)", [0] * n_x, sp.diag(*S))
L4.independent = L4.cut_independent = True
laws.append(L4)

L4b = GaussianLaw("L4b indep. Gaussian, non-centred",
                  [1, -1, 0, 2, 1, -1, 0, 1, 1],
                  sp.diag(sp.Matrix([[1, 0], [1, 2]]), sp.Matrix([[1, 0, 0], [-1, 1, 0], [2, 1, 1]]),
                          sp.Matrix([[2, 0], [1, 1]]), sp.Matrix([[1, 0], [-2, 1]])))
L4b.independent = L4b.cut_independent = True
laws.append(L4b)

L5 = product_law("L5 indep. discrete, singular Sigma_1, Sigma_2", [
    [(Q(1, 2), [u, u]) for u in (-1, 1)],                                     # Sigma_1 singular
    [(Q(1, 4), [u1, u2, u1 + u2]) for u1 in (-1, 1) for u2 in (-1, 1)],       # Sigma_2 singular
    [(Q(1, 3), [2, 1]), (Q(1, 3), [-1, 1]), (Q(1, 3), [-1, -2])],             # centred, asymmetric
    [(Q(1, 4), [1, 0]), (Q(1, 4), [-1, 0]), (Q(1, 4), [0, 3]), (Q(1, 4), [0, -3])]])
L5.independent = L5.cut_independent = True
laws.append(L5)

L12 = sp.Matrix(5, 5, lambda i, j: (1 if i == j else random.choice([-1, 0, 1])) if j <= i else 0)
L34 = sp.Matrix(4, 4, lambda i, j: (1 if i == j else random.choice([-1, 0, 1])) if j <= i else 0)
L12[2, 0], L34[2, 0] = 1, 1  # make sure x1 and x2 (and x3 and x4) are correlated
L6 = GaussianLaw("L6 Gaussian, (X1,X2) indep. of (X3,X4), X1~X2 correlated", [0] * n_x, sp.diag(L12, L34))
L6.cut_independent = True
laws.append(L6)

L7c = sp.Matrix(n_x, n_x, lambda i, j: (1 if i == j else random.choice([-1, 0, 1])) if j <= i else 0)
L7c[5, 0] = L7c[7, 2] = 1  # correlations across the cut
L7 = GaussianLaw("L7 Gaussian, dependent across the cut, non-centred", [1, 0, -1, 0, 1, 0, 0, 1, -1], L7c)
laws.append(L7)


# ---------------------------------------------------------------------------
# Population loss, residual moment and gradients, law by law (Sections 3.2-3.4)
# ---------------------------------------------------------------------------
def index_gradients(Dt):
    """(d2-index-grad-A)-(d2-index-grad-C): contractions of a 4-index error tensor Dt."""
    gA = [sum(Dt[(i1, i2, i3, i4)] * C[al, be] * core_B[be][i3][i4]
              for be in range(DR) for i3 in range(d3) for i4 in range(d4))
          for al in range(DL) for i1 in range(d1) for i2 in range(d2)]
    gB = [sum(Dt[(i1, i2, i3, i4)] * core_A[al][i1][i2] * C[al, be]
              for al in range(DL) for i1 in range(d1) for i2 in range(d2))
          for be in range(DR) for i3 in range(d3) for i4 in range(d4)]
    gC = [sum(Dt[(i1, i2, i3, i4)] * core_A[al][i1][i2] * core_B[be][i3][i4]
              for i1, i2, i3, i4 in LEX)
          for al in range(DL) for be in range(DR)]
    return gA, gB, gC


core_syms_A = [core_A[al][i1][i2] for al in range(DL) for i1 in range(d1) for i2 in range(d2)]
core_syms_B = [core_B[be][i3][i4] for be in range(DR) for i3 in range(d3) for i4 in range(d4)]
core_syms_C = [C[al, be] for al in range(DL) for be in range(DR)]

# (d2-index-error), built from the cores directly.
Delta4 = {k: T[k] - T_star[k] for k in T}
check("(d2-index-error) is the 4-index form of Delta = A C B^T - M*",
      all(sp.expand(Delta4[(i1, i2, i3, i4)] - Delta_theta[i1 * d2 + i2, i3 * d4 + i4]) == 0
          for i1, i2, i3, i4 in Delta4))

store = {}
for law in laws:
    t0 = time.time()
    section(law.name)
    E = law.expect
    L_D = Q(1, 2) * E(e_D ** 2)
    R = sp.Matrix(d12, d34, lambda i, j: E(e_D * z[i] * w[j]))
    S12 = sp.Matrix(d12, d12, lambda i, j: E(z[i] * z[j]))
    S34 = sp.Matrix(d34, d34, lambda i, j: E(w[i] * w[j]))
    Sig = [sp.Matrix(dims[a], dims[a], lambda i, j: E(x[a][i] * x[a][j])) for a in range(4)]
    mean = sp.Matrix([E(v) for v in xs])
    phi = vec(z * w.T)
    S1234 = sp.Matrix(d12 * d34, d12 * d34, lambda p, q: E(phi[p] * phi[q]))
    vD = vec(D)
    print(f"  (moments computed in {time.time() - t0:.1f}s; E[X] = {list(mean)})")

    # Valid for every law.
    check("grad_M L = R_Delta := E[e z w^T]   (d2-residual-moment, d2-general-coeff-gradient)",
          is_zero(sp.Matrix(d12, d34, lambda i, j: sp.diff(L_D, D[i, j])) - R))
    check("R_Delta is the {1,2}-matricization of E[e x1 (x) x2 (x) x3 (x) x4]",
          is_zero(unvec(sp.Matrix([E(e_D * Xi[n]) for n in range(len(LEX))]), d12, d34) - R))
    check("vec(R) = Sigma_[4] vec(Delta) and L = 1/2 vec(Delta)^T Sigma_[4] vec(Delta)   (d2-vectorized-loss)",
          is_zero(vec(R) - S1234 * vD) and is_zero(L_D - Q(1, 2) * (vD.T * S1234 * vD)[0]))

    sub = dict(zip(list(D), list(Delta_theta)))
    L_theta = L_D.xreplace(sub)
    R_theta = R.xreplace(sub)
    grads = [sp.diff(L_theta, s) for s in params]
    formula = list(R_theta * B * C.T) + list(R_theta.T * A * C) + list(A.T * R_theta * B)
    check("end-to-end: d/d(core) of L(A C B^T - M*) = (R B C^T, R^T A C, A^T R B)   (d2-general-gradients)",
          all(sp.expand(gr - h) == 0 for gr, h in zip(grads, formula)))
    true = dict(zip(params, grads))

    if law.whitened:
        check("E[z z^T] = I, E[w w^T] = I   (d2-z-white, d2-w-white)",
              is_zero(S12 - sp.eye(d12)) and is_zero(S34 - sp.eye(d34)))
        check("L = 1/2 ||Delta||_F^2   (d2-white-loss)", is_zero(L_D - Q(1, 2) * frob(D, D)))
        check("E[e z w^T] = Delta   (d2-residual-white)", is_zero(R - D))
        check("Sigma_[4] = I", is_zero(S1234 - sp.eye(d12 * d34)))
        flowA = (M_star - M) * B * C.T
        flowB = (M_star - M).T * A * C
        flowC = A.T * (M_star - M) * B
        check("gradient flow = (d2-white-flow)",
              all(sp.expand(-gr - h) == 0 for gr, h in zip(grads, list(flowA) + list(flowB) + list(flowC))))
        gA, gB, gC = index_gradients(Delta4)
        check("index-form gradients   (d2-index-grad-A, -B, -C)",
              all(sp.expand(true[s] - gi) == 0 for s, gi in
                  zip(core_syms_A + core_syms_B + core_syms_C, gA + gB + gC)))
        Err = {k: -v for k, v in Delta4.items()}
        fA, fB, fC = index_gradients(Err)
        check("index-form flow with E := T* - T(theta)   (d2-index-flow)",
              all(sp.expand(-true[s] - fi) == 0 for s, fi in
                  zip(core_syms_A + core_syms_B + core_syms_C, fA + fB + fC)))
        if any(m != 0 for m in mean):
            print("  note: this law is NOT centred; only E[X X^T] = I is used.")

    if law.independent:
        check("Sigma_12 = E[z z^T] = Sigma_1 (x) Sigma_2, Sigma_34 = Sigma_3 (x) Sigma_4   (d2-gram-LR)",
              is_zero(S12 - kron(Sig[0], Sig[1])) and is_zero(S34 - kron(Sig[2], Sig[3])))
        check("L = 1/2 tr(Delta^T Sigma_12 Delta Sigma_34)   (d2-nonwhite-loss)",
              is_zero(L_D - Q(1, 2) * (D.T * S12 * D * S34).trace()))
        check("grad_M L = E[e z w^T] = Sigma_12 Delta Sigma_34   (d2-coeff-gradient-nonwhite, d2-residual-nonwhite)",
              is_zero(R - S12 * D * S34))
        E_S = S12 * (M_star - M) * S34
        check("gradient flow = (d2-nonwhite-flow), and E_Sigma = -R_Delta",
              all(sp.expand(-gr - h) == 0 for gr, h in
                  zip(grads, list(E_S * B * C.T) + list(E_S.T * A * C) + list(A.T * E_S * B)))
              and is_zero(E_S + R_theta))
        check("Sigma_[4] = Sigma_12 (x) Sigma_34 = Sigma_1 (x) Sigma_2 (x) Sigma_3 (x) Sigma_4   (d2-gamma-product)",
              is_zero(S1234 - kron(S12, S34)) and is_zero(S1234 - kron(*Sig)))
        DS = {(i1, i2, i3, i4): sum(Sig[0][i1, j1] * Sig[1][i2, j2] * Sig[2][i3, j3] * Sig[3][i4, j4]
                                    * Delta4[(j1, j2, j3, j4)] for j1, j2, j3, j4 in LEX)
              for i1, i2, i3, i4 in LEX}
        check("(d2-weighted-index-error) matricizes to Sigma_12 Delta Sigma_34",
              all(sp.expand(DS[(i1, i2, i3, i4)] - (S12 * Delta_theta * S34)[i1 * d2 + i2, i3 * d4 + i4]) == 0
                  for i1, i2, i3, i4 in DS))
        gA, gB, gC = index_gradients(DS)
        check("index-form gradients with Delta replaced by Delta_Sigma",
              all(sp.expand(true[s] - gi) == 0 for s, gi in
                  zip(core_syms_A + core_syms_B + core_syms_C, gA + gB + gC)))
        if any(m != 0 for m in mean):
            print("  note: this law is NOT centred; Sigma_a := E[X X^T] (second moment), not the covariance.")

    if law.cut_independent and not law.independent:
        check("blocks dependent within a side: Sigma_12 != Sigma_1 (x) Sigma_2", not is_zero(S12 - kron(Sig[0], Sig[1])))
        check("...yet Proposition d2-nonwhite-flow holds with Sigma_12 := E[z z^T], Sigma_34 := E[w w^T]",
              is_zero(L_D - Q(1, 2) * (D.T * S12 * D * S34).trace()) and is_zero(R - S12 * D * S34))
        check("...and Sigma_[4] = Sigma_12 (x) Sigma_34, which is not Sigma_1 (x) ... (x) Sigma_4",
              is_zero(S1234 - kron(S12, S34)) and not is_zero(S1234 - kron(*Sig)))

    if not law.cut_independent:
        check("dependence across the cut: R_Delta != Sigma_12 Delta Sigma_34", not is_zero(R - S12 * D * S34))
        check("dependence across the cut: Sigma_[4] is not Sigma_12 (x) Sigma_34", not is_zero(S1234 - kron(S12, S34)))

    store[law.name] = dict(L_D=L_D, R=R, S12=S12, S34=S34, Sig=Sig, S1234=S1234,
                           L_theta=L_theta, R_theta=R_theta, grads=grads)
    print(f"  ({time.time() - t0:.1f}s)")

# Monte Carlo sanity check of the Gaussian moment code on the most general law.
section("Monte Carlo sanity check of the exact Gaussian moments (L7)")
rng = np.random.default_rng(0)
Dnum = rng.standard_normal((d12, d34))
exact = float(store[L7.name]["L_D"].xreplace(dict(zip(list(D), Dnum.ravel().tolist()))))
n_mc = 2_000_000
Xs = np.array(L7.mu, dtype=float).ravel() + rng.standard_normal((n_mc, n_x)) @ np.array(L7.Lc, dtype=float).T
blocks = np.split(Xs, np.cumsum(dims)[:-1], axis=1)
Zs = np.einsum("ni,nj->nij", blocks[0], blocks[1]).reshape(n_mc, d12)
Ws = np.einsum("ni,nj->nij", blocks[2], blocks[3]).reshape(n_mc, d34)
es = np.einsum("ni,ij,nj->n", Zs, Dnum, Ws)
mc, se = 0.5 * np.mean(es ** 2), 0.5 * np.std(es ** 2) / np.sqrt(n_mc)
print(f"  exact L = {exact:.4f}, Monte Carlo L = {mc:.4f} +- {se:.4f}")
check("exact Gaussian moments agree with Monte Carlo (within 4 standard errors)", abs(exact - mc) < 4 * se)

# ---------------------------------------------------------------------------
# Remark (whitened analysis coordinates), law L4 with Sigma_a = S_a^2
# ---------------------------------------------------------------------------
section("Remark: whitened analysis coordinates (L4)")
st = store[L4.name]
SqL, SqR = kron(S[0], S[1]), kron(S[2], S[3])
check("Sigma_12^{1/2} = S_1 (x) S_2 and Sigma_34^{1/2} = S_3 (x) S_4 (positive definite square roots)",
      is_zero(SqL * SqL - st["S12"]) and is_zero(SqR * SqR - st["S34"])
      and SqL.is_positive_definite and SqR.is_positive_definite)
check("L = 1/2 ||Sigma_12^{1/2} Delta Sigma_34^{1/2}||_F^2   (d2-nonwhite-loss)",
      is_zero(st["L_D"] - Q(1, 2) * frob(SqL * D * SqR, SqL * D * SqR)))
At, Bt, Mt = SqL * A, SqR * B, SqL * M_star * SqR
check("Atilde C Btilde^T - Mtilde* = Sigma_12^{1/2} Delta Sigma_34^{1/2}",
      is_zero(At * C * Bt.T - Mt - SqL * Delta_theta * SqR))
gradA = st["R_theta"] * B * C.T
gradB = st["R_theta"].T * A * C
gradC = A.T * st["R_theta"] * B
resid_t = At * C * Bt.T - Mt
gradAt, gradBt, gradCt = resid_t * Bt * C.T, resid_t.T * At * C, At.T * resid_t * Bt
check("gradient flow in (A,B,C): d/dt Atilde = -Sigma_12 grad_Atilde Ltilde, d/dt Btilde = -Sigma_34 grad_Btilde Ltilde, "
      "d/dt C = -grad_C Ltilde",
      is_zero(SqL * gradA - st["S12"] * gradAt) and is_zero(SqR * gradB - st["S34"] * gradBt)
      and is_zero(gradC - gradCt))
pt = {s: Q(random.randint(-4, 4), random.randint(1, 3)) for s in params}
check("...which differs from Euclidean gradient flow in (Atilde, Btilde, C)",
      (SqL * gradA - gradAt).xreplace(pt) != sp.zeros(d12, DL))

# ---------------------------------------------------------------------------
# Section 3.5: differential, adjoint, tangent operator (generic matrices)
# ---------------------------------------------------------------------------
section("3.5 Differential, adjoint and tangent operator (generic symbolic A, B, C, X)")
Ag, Bg, Cg = sym_matrix("A", d12, DL), sym_matrix("B", d34, DR), sym_matrix("C", DL, DR)
Xg = sym_matrix("X", d12, d34)
dAg, dBg, dCg = sym_matrix("dA", d12, DL), sym_matrix("dB", d34, DR), sym_matrix("dC", DL, DR)
t = sp.Symbol("t")


def DPhi(A_, B_, C_, dA, dB, dC):
    return dA * C_ * B_.T + A_ * C_ * dB.T + A_ * dC * B_.T  # (d2-DPhi)


def DPhi_adj(A_, B_, C_, X_):
    return X_ * B_ * C_.T, X_.T * A_ * C_, A_.T * X_ * B_  # (d2-DPhi-adjoint)


def Kop(A_, B_, C_, X_):
    return A_ * C_ * C_.T * A_.T * X_ + A_ * A_.T * X_ * B_ * B_.T + X_ * B_ * C_.T * C_ * B_.T  # (d2-K-operator)


Phi_t = (Ag + t * dAg) * (Cg + t * dCg) * (Bg + t * dBg).T
check("d/dt Phi(A+t dA, B+t dB, C+t dC)|_0 = DPhi   (d2-DPhi; dM in d2-differential-proof)",
      is_zero(sp.diff(Phi_t, t).subs(t, 0) - DPhi(Ag, Bg, Cg, dAg, dBg, dCg)))
adj = DPhi_adj(Ag, Bg, Cg, Xg)
check("<X, DPhi[dA,dB,dC]> = <X B C^T, dA> + <X^T A C, dB> + <A^T X B, dC>   (d2-DPhi-adjoint)",
      is_zero(frob(Xg, DPhi(Ag, Bg, Cg, dAg, dBg, dCg)) - frob(adj[0], dAg) - frob(adj[1], dBg) - frob(adj[2], dCg)))
check("DPhi DPhi^* [X] = A C C^T A^T X + A A^T X B B^T + X B C^T C B^T   (d2-M-flow, d2-K-operator)",
      is_zero(DPhi(Ag, Bg, Cg, *adj) - Kop(Ag, Bg, Cg, Xg)))
check("<X, K[X]> = ||X B C^T||^2 + ||X^T A C||^2 + ||A^T X B||^2   (d2-K-positive)",
      is_zero(frob(Xg, Kop(Ag, Bg, Cg, Xg)) - frob(adj[0], adj[0]) - frob(adj[1], adj[1]) - frob(adj[2], adj[2])))
Jg = vec(Ag * Cg * Bg.T).jacobian(list(Ag) + list(Bg) + list(Cg))
check("J J^T vec(X) = vec(K[X]) (row-by-row vec)", is_zero(Jg * (Jg.T * vec(Xg)) - vec(Kop(Ag, Bg, Cg, Xg))))

# Tangent kernel: <grad g(x), grad g(x')> = <z w^T, K[z' w'^T]> for two inputs, symbolic in the cores.
x_i = [random.randint(-3, 3) for _ in range(n_x)]
x_j = [random.randint(-3, 3) for _ in range(n_x)]
zi, wi = z.xreplace(dict(zip(xs, x_i))), w.xreplace(dict(zip(xs, x_i)))
zj, wj = z.xreplace(dict(zip(xs, x_j))), w.xreplace(dict(zip(xs, x_j)))
g_i, g_j = (zi.T * M * wi)[0], (zj.T * M * wj)[0]
ntk = sum(sp.diff(g_i, s) * sp.diff(g_j, s) for s in params)
check("tangent kernel: <grad g(x_i), grad g(x_j)> = <z_i w_i^T, K[z_j w_j^T]>   (after d2-K-positive)",
      is_zero(ntk - frob(zi * wi.T, Kop(A, B, C, zj * wj.T))))
check("grad_theta g(x) = DPhi^*[z w^T]",
      all(sp.expand(sp.diff(g_i, s) - h) == 0 for s, h in
          zip(params, [e for blk in DPhi_adj(A, B, C, zi * wi.T) for e in list(blk)])))

section("3.5 Coefficient-space flow and loss decay at an exact random point (law L7)")
st = store[L7.name]
pt = {s: Q(random.randint(-4, 4), random.randint(1, 3)) for s in params}
grad_pt = sp.Matrix([gr.xreplace(pt) for gr in st["grads"]])
theta_dot = -grad_pt
J_pt = vec(M).jacobian(params).xreplace(pt)
m_dot = J_pt * theta_dot
R_pt = st["R_theta"].xreplace(pt)
A_pt, B_pt, C_pt = A.xreplace(pt), B.xreplace(pt), C.xreplace(pt)
check("dm/dt = -J J^T Sigma_[4] (m - m*)   (d2-vectorized-flow)",
      m_dot == -J_pt * J_pt.T * st["S1234"] * vec(Delta_theta.xreplace(pt)))
check("dM/dt = -K[R_Delta]   (d2-M-flow)", unvec(m_dot, d12, d34) == -Kop(A_pt, B_pt, C_pt, R_pt))
dLdt = (grad_pt.T * theta_dot)[0]
check("dL/dt = -sum ||grad||^2 = -<R, K[R]> <= 0   (d2-loss-decay)",
      dLdt == -frob(R_pt, Kop(A_pt, B_pt, C_pt, R_pt)) and dLdt <= 0)

# ---------------------------------------------------------------------------
# Proposition (balance invariants)
# ---------------------------------------------------------------------------
section("Balance invariants (generic symbolic R)")
Rg = sym_matrix("R", d12, d34)
Adot, Bdot, Cdot = -Rg * Bg * Cg.T, -Rg.T * Ag * Cg, -Ag.T * Rg * Bg
dAtA = Adot.T * Ag + Ag.T * Adot
dBtB = Bdot.T * Bg + Bg.T * Bdot
check("d/dt A^T A = -C B^T R^T A - A^T R B C^T = d/dt C C^T",
      is_zero(dAtA - (-Cg * Bg.T * Rg.T * Ag - Ag.T * Rg * Bg * Cg.T))
      and is_zero(dAtA - (Cdot * Cg.T + Cg * Cdot.T)))
check("d/dt B^T B = -C^T A^T R B - B^T R^T A C = d/dt C^T C",
      is_zero(dBtB - (-Cg.T * Ag.T * Rg * Bg - Bg.T * Rg.T * Ag * Cg))
      and is_zero(dBtB - (Cdot.T * Cg + Cg.T * Cdot)))
Xl, Yr = rand_int_matrix(DL, DL), rand_int_matrix(DR, DR)
while Xl.det() == 0 or Yr.det() == 0:
    Xl, Yr = rand_int_matrix(DL, DL), rand_int_matrix(DR, DR)
check("gauge action (A,C,B) -> (AX, X^{-1} C Y^{-T}, BY) leaves M = A C B^T unchanged",
      is_zero((Ag * Xl) * (Xl.inv() * Cg * Yr.inv().T) * (Bg * Yr).T - Ag * Cg * Bg.T))

section("Balance invariants and loss decay along numerically integrated flows")


def integrate(law_name, T_end=60.0, scale=0.3, seed=0):
    G = np.array(store[law_name]["S1234"], dtype=float)
    G /= np.linalg.eigvalsh(G).max()
    Ms = np.array(M_star, dtype=float)
    Ms /= np.linalg.norm(Ms)
    sizes = [d12 * DL, d34 * DR, DL * DR]

    def unpack(th):
        a_, b_, c_ = np.split(th, np.cumsum(sizes)[:-1])
        return a_.reshape(d12, DL), b_.reshape(d34, DR), c_.reshape(DL, DR)

    def loss(th):
        A_, B_, C_ = unpack(th)
        v = (A_ @ C_ @ B_.T - Ms).ravel()  # row-by-row vec
        return 0.5 * v @ G @ v

    def rhs(_, th):
        A_, B_, C_ = unpack(th)
        Rm = (G @ (A_ @ C_ @ B_.T - Ms).ravel()).reshape(d12, d34)
        return np.concatenate([(-Rm @ B_ @ C_.T).ravel(), (-Rm.T @ A_ @ C_).ravel(), (-A_.T @ Rm @ B_).ravel()])

    def invariants(th):
        A_, B_, C_ = unpack(th)
        return A_.T @ A_ - C_ @ C_.T, B_.T @ B_ - C_.T @ C_

    th0 = scale * np.random.default_rng(seed).standard_normal(sum(sizes))
    sol = solve_ivp(rhs, (0, T_end), th0, method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True)
    ts = np.linspace(0, T_end, 2001)
    ths = sol.sol(ts).T
    Ls = np.array([loss(th) for th in ths])
    Q1_0, Q2_0 = invariants(th0)
    drift = max(max(np.abs(invariants(th)[0] - Q1_0).max(), np.abs(invariants(th)[1] - Q2_0).max()) for th in ths)
    print(f"  {law_name}: loss {Ls[0]:.3e} -> {Ls[-1]:.3e}; max invariant drift {drift:.1e} "
          f"(invariant size {max(np.abs(Q1_0).max(), np.abs(Q2_0).max()):.1e})")
    return drift, np.diff(Ls).max()


for name in (L4.name, L7.name):
    drift, max_increase = integrate(name)
    check(f"{name}: invariants conserved (drift < 1e-8) and loss non-increasing", drift < 1e-8 and max_increase < 1e-12)

# ---------------------------------------------------------------------------
# Remarks: empirical loss and tied lower cores
# ---------------------------------------------------------------------------
section("Remark: empirical loss")
n = 5
samples = [[random.randint(-3, 3) for _ in range(n_x)] for _ in range(n)]
subs_s = [dict(zip(xs, s)) for s in samples]
e_s = [e_D.xreplace(sd) for sd in subs_s]
L_n = Q(1, 2 * n) * sum(e ** 2 for e in e_s)
Rhat = Q(1, n) * sum((e * z.xreplace(sd) * w.xreplace(sd).T for e, sd in zip(e_s, subs_s)), sp.zeros(d12, d34))
check("L_n := (1/2n) sum e_s^2 has grad_M L_n = Rhat", is_zero(sp.Matrix(d12, d34, lambda i, j: sp.diff(L_n, D[i, j])) - Rhat))
sub = dict(zip(list(D), list(Delta_theta)))
L_n_theta, Rhat_theta = L_n.xreplace(sub), Rhat.xreplace(sub)
check("parameter gradients of L_n = (Rhat B C^T, Rhat^T A C, A^T Rhat B)",
      all(sp.expand(sp.diff(L_n_theta, s) - h) == 0 for s, h in
          zip(params, list(Rhat_theta * B * C.T) + list(Rhat_theta.T * A * C) + list(A.T * Rhat_theta * B))))

section("Remark: tied lower cores (whitened loss, d_1 = d_3, d_2 = d_4, D_L = D_R)")
Sg, Cs = sym_matrix("S", 4, 2), sym_matrix("K", 2, 2)
Mt_star = rand_int_matrix(4, 4)
Dl = Sg * Cs * Sg.T - Mt_star
Lt = Q(1, 2) * frob(Dl, Dl)
gradS = sp.Matrix(4, 2, lambda i, j: sp.diff(Lt, Sg[i, j]))
check("grad of the shared core = grad_A L + grad_B L at A = B = S", is_zero(gradS - (Dl * Sg * Cs.T + Dl.T * Sg * Cs)))

# ---------------------------------------------------------------------------
n_fail = sum(not ok for _, ok in RESULTS)
print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed.")
for name, ok in RESULTS:
    if not ok:
        print(f"  FAILED: {name}")
sys.exit(1 if n_fail else 0)
