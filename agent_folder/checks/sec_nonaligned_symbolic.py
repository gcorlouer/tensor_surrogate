#!/usr/bin/env python3
"""Symbolic (SymPy) checks of the closed-form steps of the section on balanced dynamics without alignment.

    python3 sec_nonaligned_symbolic.py
"""
import sys

import sympy as sp

RESULTS = []


def check(name, ok):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}", flush=True)


mi, mj, si, sj = sp.symbols("mu_i mu_j sigma_i sigma_j", positive=True)
a, b, Fij, Fji = sp.symbols("a b F_ij F_ji")

# 1. Derivative of the SVD: F = U^T Mdot V = Omega_U Sigma + Sigma_dot - Sigma Omega_V, off-diagonal entries.
sol = sp.solve([sp.Eq(Fij, a * mj - mi * b), sp.Eq(Fji, -a * mi + mj * b)], [a, b], dict=True)[0]
check("(Omega_U)_ij = (mu_j F_ij + mu_i F_ji)/(mu_j^2 - mu_i^2), (Omega_V)_ij = (mu_i F_ij + mu_j F_ji)/(mu_j^2 - mu_i^2)",
      sp.simplify(sol[a] - (mj * Fij + mi * Fji) / (mj ** 2 - mi ** 2)) == 0
      and sp.simplify(sol[b] - (mi * Fij + mj * Fji) / (mj ** 2 - mi ** 2)) == 0)

# 2. Linearization near the aligned manifold: Theta = Sigma* - X Sigma* + Sigma* Y to first order.
x, y = sp.symbols("x y")
k = mi ** sp.Rational(4, 3) + (mi * mj) ** sp.Rational(2, 3) + mj ** sp.Rational(4, 3)
Th_ij, Th_ji = -x * sj + si * y, x * si - sj * y
xdot = k * (mj * Th_ij + mi * Th_ji) / (mj ** 2 - mi ** 2)
ydot = k * (mi * Th_ij + mj * Th_ji) / (mj ** 2 - mi ** 2)
check("xi^+ = x + y decays at kappa_ij (sigma_i - sigma_j)/(mu_i - mu_j)",
      sp.simplify((xdot + ydot).subs(y, x) + k * (si - sj) / (mi - mj) * 2 * x) == 0)
check("xi^- = x - y decays at kappa_ij (sigma_i + sigma_j)/(mu_i + mu_j)",
      sp.simplify((xdot - ydot).subs(y, -x) + k * (si + sj) / (mi + mj) * 2 * x) == 0)

# 3. Amplitude of a rank-one channel: mu(tau) = e^{3A}/(1/mu0 + 3 int e^{3A}) solves mu' = 3 mu (rho - mu), A' = rho.
tau, mu0 = sp.symbols("tau mu0", positive=True)
A = sp.Function("A")(tau)
Jf = sp.Function("J")(tau)                              # J' = e^{3A}
mu = sp.exp(3 * A) / (1 / mu0 + 3 * Jf)
lhs = sp.diff(mu, tau).subs(sp.Derivative(Jf, tau), sp.exp(3 * A))
check("Bernoulli solution of mu' = 3 mu (rho - mu) with rho = A'",
      sp.simplify(lhs - 3 * mu * (sp.diff(A, tau) - mu)) == 0)

# 4. Orientation flow: z = e^{tau L} y0, y = z/|z| solves y' = L y - (y^T L y) y; check with a 2x2 dilation.
s1, p0, q0 = sp.symbols("s1 p0 q0", real=True)
L = sp.Matrix([[0, s1], [s1, 0]])
z = (L * tau).exp() * sp.Matrix([p0, q0])
nz = sp.sqrt((z.T * z)[0])
yv = z / nz
res = sp.diff(yv, tau) - (L * yv - ((yv.T * L * yv)[0]) * yv)
check("normalized exponential solves Oja's flow (2x2 dilation)", all(sp.simplify(e) == 0 for e in res))
check("rho = (1/2) d/dtau log Z with Z = |z|^2",
      sp.simplify((yv.T * L * yv)[0] - sp.diff(sp.log((z.T * z)[0]), tau) / 2) == 0)

# 5. Escape-time integral: int_0^infty dtau / sqrt(g^2 + a^2 e^{2 s tau}) = arsinh(g/a)/(s g).
g, aa, ss = sp.symbols("g a s", positive=True)
xx = sp.symbols("x", positive=True)
# substitution x = a e^{s tau}/g turns the integrand into dx / (s g x sqrt(1 + x^2))
inner = sp.integrate(1 / (xx * sp.sqrt(1 + xx ** 2)), (xx, aa / g, sp.oo))
check("int_0^oo (g^2 + a^2 e^{2 s tau})^{-1/2} dtau = arsinh(g/a)/(s g)",
      sp.simplify(sp.simplify(inner / (ss * g)) - sp.asinh(g / aa) / (ss * g)) == 0
      or abs(float((inner / (ss * g) - sp.asinh(g / aa) / (ss * g)).subs({g: 0.7, aa: 0.2, ss: 1.3}))) < 1e-12)

# 6. Rates at a critical point: mode i learned (s^3 = sigma_i), rotation towards teacher mode j.
s = sp.symbols("s", positive=True)
eu, ev = sp.symbols("e_u e_v")
J = sp.Matrix([[-s * si, s * sj], [s * sj, -s * si]])   # (e_u, e_v)' = s (sigma_j e_v - sigma_i e_u, ...)
eig = J.subs(s, si ** sp.Rational(1, 3)).eigenvals()
check("rotation rates sigma_i^{1/3}(sigma_j - sigma_i) and -sigma_i^{1/3}(sigma_j + sigma_i)",
      set(sp.simplify(e) for e in eig) == {sp.simplify(si ** sp.Rational(1, 3) * (sj - si)),
                                           sp.simplify(-si ** sp.Rational(1, 3) * (sj + si))})
check("amplitude rate: d/ds [s^2 (sigma - s^3)] at s^3 = sigma equals -3 sigma^{4/3}",
      sp.simplify(sp.diff(s ** 2 * (si - s ** 3), s).subs(s, si ** sp.Rational(1, 3)) + 3 * si ** sp.Rational(4, 3)) == 0)

# 7. Rank-one channel of an N-core tree: mu = s^N, dtau = s^{N-2} dt gives dmu/dtau = N mu (rho - mu).
N, rho = sp.symbols("N rho", positive=True)
sdot = s ** (N - 1) * (rho - s ** N)
check("N cores: mu = s^N, tau' = s^{N-2} gives dmu/dtau = N mu (rho - mu)",
      sp.simplify(sp.diff(s ** N, s) * sdot / s ** (N - 2) - N * s ** N * (rho - s ** N)) == 0)

n_fail = sum(not ok for _, ok in RESULTS)
print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed.")
sys.exit(1 if n_fail else 0)
