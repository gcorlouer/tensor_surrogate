#!/usr/bin/env python3
"""Symbolic checks (SymPy) for the note saddle_to_saddle.tex.

    python3 s2s_symbolic.py

1. Two-channel manifold on two small trees (an MPS with an interior core of the Steiner subtree that carries a site,
   and an HT tree with an internal core): the gradient of the loss vanishes off the manifold, and the loss restricted
   to the manifold is the reduced loss of the note.
2. Conservation laws and reduced equations on the manifold.
3. Clock identities.
4. Orientation flow: the invariants (1 - p_g^2) e^{2 int rho}, the change of variables giving I_kappa, and Lambda for
   two leg groups (antiderivative).
5. Rebalancing law and the change of clock rate of a later candidate.
6. Frozen-background transition law.
7. Rotation block of the Hessian at the saddle of a K-core MPS (K = 3, 4, 5).
"""
import itertools
import sys

import sympy as sp

RESULTS = []


def check(name, ok):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}", flush=True)


def zero(e):
    return sp.simplify(sp.expand(e)) == 0


# ---------------------------------------------------------------------------------------------------------------
print("== 1. Two-channel manifold: invariance and reduced loss")


class SymTTN:
    def __init__(self, n, edges, phys, chi):
        self.n, self.edges, self.phys, self.chi = n, edges, phys, chi
        self.pnodes = sorted(phys)
        self.inc = {v: [e for e in edges if v in e] for v in range(n)}
        self.shapes = {v: ((phys[v],) if v in phys else ()) + (chi,) * len(self.inc[v]) for v in range(n)}

    def entry_index(self, v, i_phys, beta):
        return ((i_phys[self.pnodes.index(v)],) if v in self.phys else ()) + \
            tuple(beta[self.edges.index(e)] for e in self.inc[v])

    def configs(self):
        return itertools.product(itertools.product(*[range(self.phys[v]) for v in self.pnodes]),
                                 itertools.product(*[range(self.chi)] * len(self.edges)))

    def W(self, cores):
        W = {}
        for i, beta in self.configs():
            term = sp.Integer(1)
            for v in range(self.n):
                c = cores[v].get(self.entry_index(v, i, beta), 0)
                if c == 0:
                    term = 0
                    break
                term *= c
            if term != 0:
                W[i] = W.get(i, 0) + term
        return W

    def gradient(self, cores, R):
        """dL/d(core entry) = sum over configurations of R[i] times the product of the other cores."""
        G = {v: {} for v in range(self.n)}
        for i, beta in self.configs():
            r = R.get(i, 0)
            if r == 0:
                continue
            vals = [cores[v].get(self.entry_index(v, i, beta), 0) for v in range(self.n)]
            nz = [v for v in range(self.n) if vals[v] != 0]
            if len(nz) < self.n - 1:
                continue
            for v in range(self.n):
                others = [vals[u] for u in range(self.n) if u != v]
                if any(o == 0 for o in others):
                    continue
                idx = self.entry_index(v, i, beta)
                G[v][idx] = G[v].get(idx, 0) + r * sp.Mul(*others)
        return G

    def all_entries(self, v):
        return itertools.product(*[range(s) for s in self.shapes[v]])


def two_channel_case(name, net, q, pir, S, S_edges, Vr):
    """Ansatz with separate amplitudes on every core (unbalanced), and dormant vectors v_u on the cores of S that
    carry a site (zero on the learned index for u in V_r) or scalars beta_u on the others."""
    sj, sr = sp.symbols("s_j s_r", positive=True)
    amp = {v: sp.Symbol(f"x{v}" if v in S else f"a{v}", positive=True) for v in range(net.n)}
    dorm = {}
    for u in S:
        if u in net.phys:
            dorm[u] = [sp.Integer(0) if (u in Vr and i == q[u]) else sp.Symbol(f"v{u}_{i}", real=True)
                       for i in range(net.phys[u])]
        else:
            dorm[u] = sp.Symbol(f"beta{u}", real=True)
    cores = {v: {} for v in range(net.n)}
    for v in range(net.n):
        cores[v][((q[v],) if v in net.phys else ()) + (0,) * len(net.inc[v])] = amp[v]
    on_support = {v: set(cores[v]) for v in range(net.n)}
    for u in S:
        bidx = tuple(1 if e in S_edges else 0 for e in net.inc[u])
        if u in net.phys:
            for i in range(net.phys[u]):
                cores[u][(i,) + bidx] = dorm[u][i]
                if not (u in Vr and i == q[u]):
                    on_support[u].add((i,) + bidx)
        else:
            cores[u][bidx] = dorm[u]
            on_support[u].add(bidx)
    Wstar = {tuple(q[v] for v in net.pnodes): sj, tuple(pir[v] for v in net.pnodes): sr}
    W = net.W(cores)
    R = {i: W.get(i, 0) - Wstar.get(i, 0) for i in set(W) | set(Wstar)}
    G = net.gradient(cores, R)
    off = [(v, idx) for v in range(net.n) for idx in net.all_entries(v) if idx not in on_support[v]]
    check(f"{name}: the gradient vanishes at all {len(off)} entries off the manifold",
          all(zero(G[v].get(idx, 0)) for v, idx in off))
    # reduced loss: (1/2)(omega_j - s_j)^2 + (1/2) A^2 prod |v_u|^2 - s_r A prod <v_u, e_{pi_r(u)}> + s_r^2/2
    omega_j = sp.Mul(*amp.values())
    A = sp.Mul(*[amp[v] for v in range(net.n) if v not in S]) * \
        sp.Mul(*[dorm[u] for u in S if u not in net.phys])
    norms = sp.Mul(*[sum(c ** 2 for c in dorm[u]) for u in S if u in net.phys])
    overl = sp.Mul(*[dorm[u][pir[u]] for u in S if u in net.phys])
    L_red = (omega_j - sj) ** 2 / 2 + A ** 2 * norms / 2 - sr * A * overl + sr ** 2 / 2
    L = sum(r ** 2 for r in R.values()) / 2
    check(f"{name}: the loss on the manifold is the reduced loss", zero(L - L_red))
    return G, cores


# (a) MPS with 4 cores, d = 3, Steiner subtree {0, 1, 2}, V_r = {0, 2}: core 1 is in S but not in V_r
mps4 = SymTTN(4, [(0, 1), (1, 2), (2, 3)], {0: 3, 1: 3, 2: 3, 3: 3}, 2)
two_channel_case("MPS, S = {0,1,2}, V_r = {0,2}", mps4, {0: 0, 1: 0, 2: 0, 3: 0}, {0: 1, 1: 0, 2: 2, 3: 0},
                 {0, 1, 2}, {(0, 1), (1, 2)}, {0, 2})
# (b) HT tree with 4 leaves, d = 3, V_r = {leaf 3, leaf 4}, S = {1, 3, 4}: the internal core 1 has no site
ht = SymTTN(7, [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)], {3: 3, 4: 3, 5: 3, 6: 3}, 2)
two_channel_case("HT tree, S = {1,3,4}, V_r = {3,4}", ht, {3: 0, 4: 0, 5: 0, 6: 0}, {3: 1, 4: 2, 5: 0, 6: 0},
                 {1, 3, 4}, {(1, 3), (1, 4)}, {3, 4})

# ---------------------------------------------------------------------------------------------------------------
print("== 2. Conservation laws and reduced equations")
sj, sr = sp.symbols("s_j s_r", positive=True)
a1, a2, x1, x2, x3, beta = sp.symbols("a1 a2 x1 x2 x3 beta", positive=True)
v1 = sp.symbols("v1_0:3", real=True)
v3 = sp.symbols("v3_0:3", real=True)
e1, e2 = sp.Matrix([0, 1, 0]), sp.Matrix([0, 0, 1])
V1, V3 = sp.Matrix(v1), sp.Matrix(v3)
# generic reduced loss: cores a1, a2 off S; x1, x2, x3 on S; dormant vectors V1, V3 and scalar beta on S
A = a1 * a2 * beta
omega_j = a1 * a2 * x1 * x2 * x3
L = (omega_j - sj) ** 2 / 2 + A ** 2 * (V1.dot(V1)) * (V3.dot(V3)) / 2 - sr * A * V1.dot(e1) * V3.dot(e2)
vars_ = [a1, a2, x1, x2, x3, beta] + list(v1) + list(v3)
flow = {s: -sp.diff(L, s) for s in vars_}


def ddt(expr):
    return sum(sp.diff(expr, s) * flow[s] for s in vars_)


check("edge leaving S: d/dt (a^2 - x^2 - |v|^2) = 0 (here a1, x1 and the dormant vector V1)",
      zero(ddt(a1 ** 2 - x1 ** 2 - V1.dot(V1))))
check("edge leaving S at a core without site: d/dt (a^2 - x^2 - beta^2) = 0", zero(ddt(a2 ** 2 - x2 ** 2 - beta ** 2)))
check("edges inside S: d/dt (x_u^2 - x_v^2) = 0 and d/dt (|v_u|^2 - beta^2) = 0",
      zero(ddt(x1 ** 2 - x3 ** 2)) and zero(ddt(V1.dot(V1) - beta ** 2)) and zero(ddt(V3.dot(V3) - beta ** 2)))
# balanced reduced equations: x_v = x, |v_u| = beta = b, a_v = a
a, x, b = sp.symbols("a x b", positive=True)
K, k = 5, 3
P1, P3 = sp.symbols("p1 p3", real=True)
wj, wr = a ** (K - k) * x ** k, a ** (K - k) * b ** k
rho = sr * P1 * P3
sub = {a1: a, a2: a, x1: x, x2: x, x3: x, beta: b}
w1 = sp.symbols("w1_0:3", real=True)
w3 = sp.symbols("w3_0:3", real=True)
sub.update({v1[i]: b * w1[i] for i in range(3)})
sub.update({v3[i]: b * w3[i] for i in range(3)})
norm1 = {w1[0] ** 2 + w1[1] ** 2 + w1[2] ** 2: 1}


def on_manifold(e):
    e = sp.expand(e.subs(sub))
    e = e.subs(w1[0] ** 2, 1 - w1[1] ** 2 - w1[2] ** 2).subs(w3[0] ** 2, 1 - w3[1] ** 2 - w3[2] ** 2)
    return sp.expand(e.subs({w1[1]: P1 if False else w1[1]}))


ok = zero(on_manifold(flow[x1]) - on_manifold(sp.expand(wj * (sj - wj) / x)))
ok &= zero(on_manifold(flow[a1]) - on_manifold(sp.expand(
    (wj * (sj - wj) + wr * (sr * w1[1] * w3[2] - wr)) / a)))
ok &= zero(on_manifold(flow[beta]) - on_manifold(sp.expand(wr * (sr * w1[1] * w3[2] - wr) / b)))
check("balanced manifold: x' = omega_j (s_j - omega_j)/x, b' = omega_r (rho - omega_r)/b, "
      "a' = [omega_j (s_j - omega_j) + omega_r (rho - omega_r)]/a", ok)
# orientation: the tangential part of the dormant vector's velocity, divided by b, is (omega_r/b^2) (I - w w^T) grad rho
Wv = sp.Matrix(w1)
vel = sp.Matrix([flow[v1[i]] for i in range(3)]).applyfunc(lambda e: e.subs(sub))
proj = (sp.eye(3) - Wv * Wv.T)
grad_rho = sr * w3[2] * e1
lhs = (proj * vel / b).applyfunc(sp.expand)
rhs = (wr / b ** 2 * proj * grad_rho).applyfunc(sp.expand)
unit_ideal = [sum(c ** 2 for c in w1) - 1, sum(c ** 2 for c in w3) - 1]     # |w_1| = |w_3| = 1
gens = list(w1) + list(w3)
check("balanced manifold: w' = (omega_r / b^2)(I - w w^T) grad_w rho (modulo |w| = 1)",
      all(sp.reduced(sp.expand(e), unit_ideal, *gens, order="lex")[1] == 0 for e in (lhs - rhs)))

# ---------------------------------------------------------------------------------------------------------------
print("== 3. Clock identities")
kk, KK = sp.symbols("k K", positive=True)
xs, bs, as_, Rh = sp.symbols("x b a rho", positive=True)
wj = as_ ** (KK - kk) * xs ** kk
wr = as_ ** (KK - kk) * bs ** kk
xdot, bdot = wj * (sj - wj) / xs, wr * (Rh - wr) / bs
check("d x^-(k-2)/dt = -(k-2)(s_j - omega_j) a^(K-k)",
      sp.simplify(sp.diff(xs ** (-(kk - 2)), xs) * xdot + (kk - 2) * (sj - wj) * as_ ** (KK - kk)) == 0)
check("d b^-(k-2)/dt = -(k-2)(rho - omega_r) a^(K-k)",
      sp.simplify(sp.diff(bs ** (-(kk - 2)), bs) * bdot + (kk - 2) * (Rh - wr) * as_ ** (KK - kk)) == 0)
check("d log b/dt = (rho - omega_r) a^(K-2) for k = 2", sp.simplify((bdot / bs - (Rh - wr) * as_ ** (KK - kk)).subs(kk, 2)) == 0)
b0 = sp.Symbol("b0", positive=True)
# dG/dt with G' = exp(-(k-2) A) dtau/dt, exp(A) = b/b0, dtau/dt = a^(K-k) b^(k-2)
check("dG/dt = b(0)^(k-2) dC/dt, hence G(tau(t)) = b(0)^(k-2) C(t)",
      sp.simplify((b0 / bs) ** (kk - 2) * as_ ** (KK - kk) * bs ** (kk - 2) - b0 ** (kk - 2) * as_ ** (KK - kk)) == 0)

# ---------------------------------------------------------------------------------------------------------------
print("== 4. Orientation flow of a rank-one dormant channel")
m = 4
ps = sp.symbols(f"p1:{m + 1}", real=True)
E = sp.Symbol("E", positive=True)
P = sp.Mul(*ps)
pdot = [sr * P / ps[g] * (1 - ps[g] ** 2) for g in range(m)]
Edot = -2 * sr * P * E
ok = all(zero(sp.diff((1 - ps[g] ** 2) / E, ps[g]) * pdot[g] + sp.diff((1 - ps[g] ** 2) / E, E) * Edot) for g in range(m))
check("m = 4: (1 - p_g^2) e^{2 int rho} is constant along p_g' = s_r (P/p_g)(1 - p_g^2)", ok)
wv = sp.Matrix(sp.symbols("w0:4", real=True))
tv = sp.Matrix([1, 0, 0, 0])
c_ = sp.Symbol("c")
g = c_ * tv
check("p' = <e, (I - w w^T) g> = c (1 - p^2) for g = c e (the projected flow of one group)",
      zero(((tv.T * (sp.eye(4) - wv * wv.T) * g)[0] - c_ * (1 - wv[0] ** 2)).subs(wv[0], wv[0])))
u = sp.Symbol("u", positive=True)
kap = sp.Symbol("kappa", positive=True)
cs = sp.symbols(f"c1:{m + 1}", positive=True)
Q = sp.Mul(*[sp.sqrt(1 - cg * E) for cg in cs])
integrand_E = E ** (kap / 2) / (2 * sr * E * Q)            # E^{kappa/2} dtau, with dtau = -dE/(2 s_r E Q)
integrand_u = (integrand_E.subs(E, u ** 2) * 2 * u)
target_u = u ** (kap - 1) / (sr * sp.Mul(*[sp.sqrt(1 - cg * u ** 2) for cg in cs]))
check("E = u^2 turns int E^{kappa/2} dtau into (1/s_r) int_0^1 u^{kappa-1} prod (1 - c_g u^2)^{-1/2} du",
      sp.simplify(integrand_u - target_u) == 0)
check("aligned limit: I_kappa = 1/(kappa s_r)",
      sp.simplify(sp.integrate(target_u.subs({cg: 0 for cg in cs}), (u, 0, 1)) - 1 / (kap * sr)) == 0)
c1, c2, t = sp.symbols("c1 c2 t", positive=True)
Qt = 1 - (c1 + c2) * t + c1 * c2 * t ** 2
F = -sp.log((2 - (c1 + c2) * t + 2 * sp.sqrt(Qt)) / t)
check("antiderivative: d/dt[-log((2 - (c1+c2) t + 2 sqrt(Q))/t)] = 1/(t sqrt(Q)), Q = (1 - c1 t)(1 - c2 t)",
      sp.simplify(sp.diff(F, t) - 1 / (t * sp.sqrt(Qt))) == 0)
p1, p2 = sp.symbols("p1 p2", positive=True)
# Lambda = (1/2) int_0^1 [1/(t sqrt Q) - 1/t] dt = (1/2)[F + log t]_0^1 ... evaluated with c_g = 1 - p_g^2
Gt = sp.simplify(F - sp.log(t))                              # antiderivative of 1/(t sqrt(Q)) - 1/t
G1 = Gt.subs(t, 1)
G0 = sp.limit(Gt, t, 0, "+")
Lam = sp.simplify(((G1 - G0) / 2).subs({c1: 1 - p1 ** 2, c2: 1 - p2 ** 2}))
check("m = 2: Lambda = log(2/(p_1 + p_2))", sp.simplify(sp.expand_log(Lam - sp.log(2 / (p1 + p2)), force=True)) == 0
      or abs(float((Lam - sp.log(2 / (p1 + p2))).subs({p1: 0.37, p2: 0.81}))) < 1e-14)

# ---------------------------------------------------------------------------------------------------------------
print("== 5. Rebalancing law and clock rates")
for (K_, k_) in ((7, 3), (6, 2), (9, 5), (4, 4)):
    astar = (sj ** sp.Rational(2, k_) + sr ** sp.Rational(2, k_)) ** sp.Rational(k_, 2 * K_)
    xstar = (sj / astar ** (K_ - k_)) ** sp.Rational(1, k_)
    bstar = (sr / astar ** (K_ - k_)) ** sp.Rational(1, k_)
    ok = zero(sp.powsimp(astar ** 2 - xstar ** 2 - bstar ** 2, force=True)) or \
        abs(float((astar ** 2 - xstar ** 2 - bstar ** 2).subs({sj: 1.3, sr: 0.4}))) < 1e-14
    ok &= sp.simplify(astar ** (K_ - k_) * xstar ** k_ - sj) == 0 and sp.simplify(astar ** (K_ - k_) * bstar ** k_ - sr) == 0
    check(f"K = {K_}, k = {k_}: (a*, x*, b*) solve a^(K-k) x^k = s_j, a^(K-k) b^k = s_r, a^2 = x^2 + b^2", ok)
# clock rate of a candidate with subtree S' after the rebalancing caused by a candidate with subtree S
K_, k_ = 9, 3
astar = (sj ** sp.Rational(2, k_) + sr ** sp.Rational(2, k_)) ** sp.Rational(k_, 2 * K_)
xstar = (sj / astar ** (K_ - k_)) ** sp.Rational(1, k_)
f = (1 + (sr / sj) ** sp.Rational(2, k_)) ** sp.Rational(k_, 2 * K_)
ok = True
for (kp, n_out, n_S) in ((4, 5, 0), (3, 3, 3), (5, 2, 2), (4, 3, 2)):       # (|S'|, cores off S u S', |S \ S'|)
    assert n_out + n_S + kp <= K_ and n_out + n_S == K_ - kp
    rate_after = astar ** n_out * xstar ** n_S
    rate_before = sj ** sp.Rational(K_ - kp, K_)
    e = n_out - sp.Rational(n_S * (K_ - k_), k_)
    val = (rate_after / rate_before - f ** e).subs({sj: sp.Rational(13, 10), sr: sp.Rational(2, 5)})
    ok &= abs(float(val)) < 1e-13
check("clock rate after the rebalancing: R = f^(n_out - n_S (K-k)/k), f = (1 + (s_r/s_j)^(2/k))^(k/(2K))", ok)

# ---------------------------------------------------------------------------------------------------------------
print("== 6. Frozen-background transition law")
cc, bb = sp.symbols("c b", positive=True)
kq = sp.Symbol("k", positive=True)
om = cc * bb ** kq
bdot = cc * bb ** (kq - 1) * (sr - om)
check("omega = c b^k, b' = c b^(k-1)(s_r - omega)  =>  omega' = k c^(2/k) omega^(2-2/k) (s_r - omega)",
      sp.simplify(sp.diff(om, bb) * bdot - kq * cc ** (2 / kq) * om ** (2 - 2 / kq) * (sr - om)) == 0)

# ---------------------------------------------------------------------------------------------------------------
print("== 7. Rotation block at the saddle of a K-core MPS with bond dimension one")
for K_ in (3, 4, 5):
    zs = sp.symbols(f"z0:{K_}", real=True)
    al = sj ** sp.Rational(1, K_)
    # cores (al, z_v) on the basis (e_0, e_1); target s_j e_0...0 + s_r e_1 0...0 1; W has entries prod of choices
    L = 0
    for bits in itertools.product((0, 1), repeat=K_):
        Wb = sp.Mul(*[(zs[v] if bits[v] else al) for v in range(K_)])
        tgt = sj if not any(bits) else (sr if bits == (1,) + (0,) * (K_ - 2) + (1,) else 0)
        L += (Wb - tgt) ** 2 / 2
    H = sp.hessian(L, zs).subs({z: 0 for z in zs})
    M1 = sp.zeros(K_, K_)
    M1[0, K_ - 1] = M1[K_ - 1, 0] = sr
    check(f"K = {K_}: H_rot = s_j^((K-2)/K) (s_j I - M_1)",
          (H - sj ** sp.Rational(K_ - 2, K_) * (sj * sp.eye(K_) - M1)).applyfunc(sp.simplify) == sp.zeros(K_, K_))

n_fail = sum(not ok for _, ok in RESULTS)
print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed.")
for name, ok in RESULTS:
    if not ok:
        print(f"  FAILED: {name}")
sys.exit(1 if n_fail else 0)
