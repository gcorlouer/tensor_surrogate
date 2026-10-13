#!/usr/bin/env python3
"""Numerical checks for the note saddle_to_saddle.tex.

Gradient flows of tree tensor networks (TTNs) on several trees are compared with the reduced equations and the
asymptotic formulas of the note.

    python3 s2s_checks.py              # all checks
    python3 s2s_checks.py 1 3 5        # selected sections
    python3 s2s_checks.py --quick      # skip the slow sections (10 and 11)

Sections
    1  two-channel manifold: the full flow stays on it and obeys the reduced equations
    2  shared clock: exact identities
    3  orientation flow of a rank-one dormant channel: closed form, I_kappa and Lambda
    4  escape from the saddle (reduced equations)
    5  plateau from a small initialization (reduced equations)
    6  plateau on the full HT tree, non-aligned dormant channel
    7  transition: rebalancing law, duration, adiabatic and frozen approximations
    8  rotation block of the Hessian at the saddle
    9  two candidates at a saddle: locality against weight, crossover
   10  same target on three trees, aligned small initialization: order of acquisition and plateau lengths
   11  same target on three trees, random small initialization
   12  path reuse: after a second component is learned, a third one can branch off it
   13  the condition on the extra components W' in the invariance of the two-channel manifold
   14  perturbations off the two-channel manifold
   15  bounded remainders of the escape and plateau asymptotics
   16  a dormant channel that sees several components (three-core model): residual competition
"""
import itertools
import string
import sys
import time

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

RESULTS = []
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
QUICK = "--quick" in sys.argv
SECTIONS = {int(a) for a in ARGS} if ARGS else set(range(1, 17)) - ({10, 11} if QUICK else set())


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   ({detail})" if detail else ""), flush=True)


def section(n, title):
    print(f"\n== {n}. {title}", flush=True)


def unit(x):
    x = np.asarray(x, float)
    return x / np.linalg.norm(x)


# ---------------------------------------------------------------------------------------------------------------
# Tree tensor networks
# ---------------------------------------------------------------------------------------------------------------
class TTN:
    """Cores on the nodes of a tree; node v carries the sites node_sites[v] (grouped into one physical leg of
    dimension d**len(sites)) or none. W has one leg per physical node, in increasing node order."""

    def __init__(self, n_nodes, edges, node_sites, d, chi):
        self.n, self.edges, self.d = n_nodes, list(edges), d
        self.node_sites = {v: list(s) for v, s in node_sites.items()}
        self.phys = {v: d ** len(s) for v, s in self.node_sites.items()}
        self.pnodes = sorted(self.phys)
        self.chi = chi if isinstance(chi, dict) else {e: chi for e in self.edges}
        letters = iter(string.ascii_letters)
        self.el = {e: next(letters) for e in self.edges}
        self.pl = {v: next(letters) for v in self.pnodes}
        self.inc = {v: [e for e in self.edges if v in e] for v in range(self.n)}
        self.subs, self.shapes = [], []
        for v in range(self.n):
            s, shp = "", []
            if v in self.phys:
                s += self.pl[v]
                shp.append(self.phys[v])
            for e in self.inc[v]:
                s += self.el[e]
                shp.append(self.chi[e])
            self.subs.append(s)
            self.shapes.append(tuple(shp))
        self.out = "".join(self.pl[v] for v in self.pnodes)
        self.Wshape = tuple(self.phys[v] for v in self.pnodes)
        self.spec = ",".join(self.subs) + "->" + self.out
        dummy = [np.ones(s) for s in self.shapes]
        self.path = np.einsum_path(self.spec, *dummy, optimize="optimal")[0]
        self.gspec, self.gpath = [], []
        for v in range(self.n):
            others = [self.subs[u] for u in range(self.n) if u != v]
            sp = ",".join([self.out] + others) + "->" + self.subs[v]
            self.gspec.append(sp)
            dd = [np.ones(self.Wshape)] + [dummy[u] for u in range(self.n) if u != v]
            self.gpath.append(np.einsum_path(sp, *dd, optimize="optimal")[0])
        self.sizes = [int(np.prod(s)) for s in self.shapes]

    def W(self, cores):
        return np.einsum(self.spec, *cores, optimize=self.path)

    def unpack(self, th):
        return [p.reshape(s) for p, s in zip(np.split(th, np.cumsum(self.sizes)[:-1]), self.shapes)]

    def pack(self, cores):
        return np.concatenate([c.ravel() for c in cores])

    def grads(self, cores, R):
        return [np.einsum(self.gspec[v], R, *[cores[u] for u in range(self.n) if u != v], optimize=self.gpath[v])
                for v in range(self.n)]

    def rhs_fn(self, Wstar):
        def rhs(_, th):
            cores = self.unpack(th)
            return -np.concatenate([g.ravel() for g in self.grads(cores, self.W(cores) - Wstar)])
        return rhs

    def node_index(self, pi):
        """Physical index of every physical node for a string pi of site indices."""
        out = {}
        for v, sites in self.node_sites.items():
            i = 0
            for s in sites:
                i = i * self.d + pi[s]
            out[v] = i
        return out

    def basis(self, pi):
        idx = self.node_index(pi)
        U = np.zeros(self.Wshape)
        U[tuple(idx[v] for v in self.pnodes)] = 1.0
        return U


def mps(n, d, chi):
    return TTN(n, [(i, i + 1) for i in range(n - 1)], {i: [i] for i in range(n)}, d, chi)


def ht4(d, chi):
    """Hierarchical Tucker tree with four leaves: root 0, internal nodes 1 = (0,1) and 2 = (2,3), leaves 3-6."""
    return TTN(7, [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)], {3: [0], 4: [1], 5: [2], 6: [3]}, d, chi)


def binary4(d, chi):
    """Binary tree with four sites: two bottom cores with two sites each and a root (the three-core model)."""
    return TTN(3, [(0, 1), (0, 2)], {1: [0, 1], 2: [2, 3]}, d, chi)


def steiner(edges, terminals):
    """Smallest subtree containing the terminal nodes, by repeatedly pruning non-terminal leaves."""
    nodes, E = set(itertools.chain(*edges)), set(edges)
    changed = True
    while changed:
        changed = False
        for v in list(nodes):
            if v not in terminals and sum(1 for e in E if v in e) <= 1:
                nodes.discard(v)
                E = {e for e in E if v not in e}
                changed = True
    return nodes, E


def differing_nodes(net, pi_a, pi_b):
    ia, ib = net.node_index(pi_a), net.node_index(pi_b)
    return {v for v in net.phys if ia[v] != ib[v]}


def set_channel(net, cores, idx, S_nodes, edge_index, amp, w=None):
    """Write one channel: at every node of S_nodes, the entry with physical index idx[v] (or the vector w[v]) and
    bond index edge_index.get(e, 0) on each incident edge, with amplitude amp (a scalar or a dict)."""
    for v in S_nodes:
        a = amp[v] if isinstance(amp, dict) else amp
        eidx = tuple(edge_index.get(e, 0) for e in net.inc[v])
        if v in net.phys:
            if w is not None and v in w:
                cores[v][(slice(None),) + eidx] = a * np.asarray(w[v], float)
            else:
                cores[v][(idx[v],) + eidx] = a
        else:
            cores[v][eidx] = a


def two_channel_cores(net, q, S_nodes, S_edges, a, x, b, w):
    """Learned channel (index 0 on all edges, physical indices q) with amplitude a off S and x on S, plus the dormant
    channel on S (index 1 on the edges of S, index 0 on the edges that leave S), amplitude b, directions w."""
    cores = [np.zeros(s) for s in net.shapes]
    set_channel(net, cores, q, range(net.n), {}, {v: (x if v in S_nodes else a) for v in range(net.n)})
    set_channel(net, cores, q, S_nodes, {e: 1 for e in S_edges}, b, w)
    return cores


# ---------------------------------------------------------------------------------------------------------------
# Reduced equations of the two-channel manifold
# ---------------------------------------------------------------------------------------------------------------
class Reduced:
    """State y = (a, x, b, w_1, ..., w_m). K cores, Steiner subtree of k cores, m of which carry a physical leg,
    with target directions targets[g] (unit vectors) for the dormant channel."""

    def __init__(self, K, k, sj, sr, targets):
        self.K, self.k, self.sj, self.sr = K, k, sj, sr
        self.targets = [np.asarray(t, float) for t in targets]
        self.nw = sum(len(t) for t in self.targets)
        self.split = np.cumsum([len(t) for t in self.targets])[:-1]

    def ws(self, y):
        return np.split(np.asarray(y[3:3 + self.nw]), self.split)

    def overlaps(self, y):
        return np.array([w @ t for w, t in zip(self.ws(y), self.targets)])

    def signal(self, y):
        p = self.overlaps(y)
        P = np.prod(p)
        others = [np.prod(np.delete(p, g)) for g in range(len(p))]
        return self.sr * P, [self.sr * o * t for o, t in zip(others, self.targets)], P

    def amplitudes(self, y):
        A = y[0] ** (self.K - self.k)
        return A * y[1] ** self.k, A * y[2] ** self.k

    def rhs(self, t, y, clock=False):
        a, x, b = y[0], y[1], y[2]
        rho, grads, _ = self.signal(y)
        A = a ** (self.K - self.k)
        wj, wr = A * x ** self.k, A * b ** self.k
        fj, fr = wj * (self.sj - wj), wr * (rho - wr)
        out = [np.array([(fj + fr) / a, fj / x, fr / b])]
        out += [(wr / b ** 2) * (g - (g @ w) * w) for w, g in zip(self.ws(y), grads)]
        v = np.concatenate(out)
        return v / A if clock else v

    def overlap_r(self, y):
        """<U_r, W> = omega_r * P."""
        return self.amplitudes(y)[1] * self.signal(y)[2]


def kappa_integral(sr, p, kappa):
    """I_kappa = int_0^infty E^{kappa/2} dtau for the rank-one orientation flow with initial overlaps p (P > 0):
    (1/s_r) int_0^1 u^{kappa-1} prod_g (1 - c_g u^2)^{-1/2} du, c_g = 1 - p_g^2."""
    c = 1 - np.asarray(p, float) ** 2
    # the integrand can be sharply peaked at u = 1 when some overlap is small: split the interval there
    pts = [1 - 10.0 ** -j for j in range(1, 9)]
    return quad(lambda u: u ** (kappa - 1) / np.sqrt(np.prod(1 - c * u * u)), 0, 1, limit=1000, points=pts)[0] / sr


def lambda_shift(p):
    """Lambda = int_0^1 (prod_g (1 - c_g u^2)^{-1/2} - 1) du / u (P > 0)."""
    c = 1 - np.asarray(p, float) ** 2
    return quad(lambda u: (1 / np.sqrt(np.prod(1 - c * u * u)) - 1) / u, 0, 1, limit=500)[0]


def saddle_state(K, k, sj, delta):
    """Learned channel at rest with a dormant amplitude delta on the balanced manifold: a^2 = x^2 + delta^2 and
    a^{K-k} x^k = s_j."""
    if K == k:
        return np.sqrt(sj ** (2 / K) + delta ** 2), sj ** (1 / K)
    x = brentq(lambda x: (x * x + delta * delta) ** ((K - k) / 2) * x ** k - sj, 1e-12, sj ** (1 / K) + 1e-12,
               xtol=1e-15, rtol=1e-15)
    return np.sqrt(x * x + delta * delta), x


def integrate_clock(red, y0, event, C_max, rtol=1e-11, atol=1e-16):
    """Integrate the reduced equations in the clock C (dC = a^{K-k} dt), with physical time as an extra state."""
    K, k = red.K, red.k

    def rhs(C, z):
        return np.append(red.rhs(C, z[:-1], clock=True), 1 / z[0] ** (K - k))

    def ev(C, z):
        return event(z[:-1])
    ev.terminal, ev.direction = True, 1
    s = solve_ivp(rhs, (0, C_max), np.append(y0, 0.0), method="LSODA", rtol=rtol, atol=atol, events=ev)
    if len(s.t_events[0]) == 0:
        raise RuntimeError("event not reached")
    return s.t_events[0][0], s.y_events[0][0][:-1], s.y_events[0][0][-1]


def plateau(red, x0, b0, w0, eta=0.5, a0=None):
    """From a small initialization: clock and physical time between omega_j = eta s_j and <U_r, W> = eta s_r."""
    K, k, sj, sr = red.K, red.k, red.sj, red.sr
    a0 = np.sqrt(x0 ** 2 + b0 ** 2) if a0 is None else a0
    y0 = np.concatenate([[a0, x0, b0]] + [np.asarray(w, float) for w in w0])
    big = x0 ** -(k - 2) if k > 2 else np.log(1 / x0)
    C_max = 50 * big / min(sj, sr) + 1e3
    Cj, yj, _ = integrate_clock(red, y0, lambda y: red.amplitudes(y)[0] - eta * sj, C_max)
    dC, yr, dt = integrate_clock(red, yj, lambda y: red.overlap_r(y) - eta * sr, C_max)
    return Cj, dC, dt, yj, yr


def escape(red, delta, w0, eta=0.5):
    """From the balanced saddle with a dormant amplitude delta: clock and physical time to <U_r, W> = eta s_r."""
    a, x = saddle_state(red.K, red.k, red.sj, delta)
    y0 = np.concatenate([[a, x, delta]] + [np.asarray(w, float) for w in w0])
    big = delta ** -(red.k - 2) if red.k > 2 else np.log(1 / delta)
    C, y, t = integrate_clock(red, y0, lambda y: red.overlap_r(y) - eta * red.sr, 50 * big / red.sr + 1e3)
    return C, t, y


def rebalanced(K, k, sj, sr):
    a = (sj ** (2 / k) + sr ** (2 / k)) ** (k / (2 * K))
    return a, (sj / a ** (K - k)) ** (1 / k), (sr / a ** (K - k)) ** (1 / k)


# ---------------------------------------------------------------------------------------------------------------
if 1 in SECTIONS:
    section(1, "Two-channel manifold: the full flow stays on it and obeys the reduced equations")
    rng = np.random.default_rng(3)
    sj, sr = 1.0, 0.5
    cases = [("HT tree, V_r = {leaves of sites 0, 1}", ht4(3, 2), "0000", "1200", 200.0),
             ("MPS (6 cores), V_r = {1, 3}, middle core 2 not in V_r", mps(6, 3, 2), "000000", "010200", 200.0),
             ("MPS (6 cores), V_r = {2, 3}", mps(6, 3, 2), "000000", "001200", 200.0),
             ("binary tree (three-core model), V_r = both bottom cores", binary4(3, 2), "0000", "1221", 60.0)]
    for name, net, sj_str, sr_str, T in cases:
        pij, pir = [int(c) for c in sj_str], [int(c) for c in sr_str]
        Wstar = sj * net.basis(pij) + sr * net.basis(pir)
        q, qr = net.node_index(pij), net.node_index(pir)
        Vr = differing_nodes(net, pij, pir)
        S_nodes, S_edges = steiner(net.edges, Vr)
        K, k = net.n, len(S_nodes)
        groups = sorted(v for v in S_nodes if v in net.phys)
        w = {}
        for v in groups:
            vec = rng.standard_normal(net.phys[v])
            vec[qr[v]] = abs(vec[qr[v]]) + 0.3          # positive overlap with the target direction
            if v in Vr:
                vec[q[v]] = 0.0                          # orthogonal to the learned direction on V_r
            w[v] = unit(vec)
        x0, b0 = 0.35, 0.3
        a0 = np.sqrt(x0 ** 2 + b0 ** 2)
        th0 = net.pack(two_channel_cores(net, q, S_nodes, S_edges, a0, x0, b0, w))
        # entries that may be nonzero on the manifold
        support = [np.zeros(sh, bool) for sh in net.shapes]
        for v in range(net.n):
            support[v][((q[v],) if v in net.phys else ()) + (0,) * len(net.inc[v])] = True
        for v in S_nodes:
            eidx = tuple(1 if e in S_edges else 0 for e in net.inc[v])
            support[v][((slice(None),) if v in net.phys else ()) + eidx] = True
        off_support = ~net.pack(support)
        full = solve_ivp(net.rhs_fn(Wstar), (0, T), th0, method="DOP853", rtol=1e-13, atol=1e-15, dense_output=True)
        red = Reduced(K, k, sj, sr, [np.eye(net.phys[v])[qr[v]] for v in groups])
        y0 = np.concatenate([[a0, x0, b0]] + [w[v] for v in groups])
        rs = solve_ivp(red.rhs, (0, T), y0, method="DOP853", rtol=1e-13, atol=1e-15, dense_output=True)
        err, bal = 0.0, 0.0
        off = max(np.abs(full.sol(t)[off_support]).max() for t in np.linspace(0, T, 101))
        for t in np.linspace(0, T, 101):
            y = rs.sol(t)
            wj, wr = red.amplitudes(y)
            ws = red.ws(y)
            D = np.ones(())
            for v in net.pnodes:
                D = np.multiply.outer(D, ws[groups.index(v)] if v in groups else np.eye(net.phys[v])[q[v]])
            W_red = wj * net.basis(pij) + wr * D
            err = max(err, np.abs(net.W(net.unpack(full.sol(t))) - W_red).max())
            bal = max(bal, abs(y[0] ** 2 - y[1] ** 2 - y[2] ** 2))
        yT = rs.sol(T)
        check(f"{name} (K = {K}, k = {k}, m = {len(groups)}): entries off the manifold stay zero, and the full flow "
              f"equals the reduced flow", off == 0.0 and err < 1e-8 and bal < 1e-12,
              f"max off-manifold entry {off:.0e}, max |W_full - W_red| = {err:.1e}; at t = {T:g}: "
              f"omega_j = {red.amplitudes(yT)[0]:.3f}, <U_r,W> = {red.overlap_r(yT):.3f}")

# ---------------------------------------------------------------------------------------------------------------
if 2 in SECTIONS:
    section(2, "Shared clock: exact identities")
    K, k, sj, sr = 7, 3, 1.0, 0.5
    red = Reduced(K, k, sj, sr, [np.eye(3)[1], np.eye(3)[1]])
    w0 = [unit([0.0, 0.3, 1.0]), unit([0.0, 1.0, -0.4])]
    x0 = b0 = 0.08
    y0 = np.concatenate([[np.sqrt(x0 ** 2 + b0 ** 2), x0, b0]] + w0 + [[0.0, 0.0, 0.0, 0.0]])

    def rhs_aug(t, z):
        y = z[:-4]
        v = red.rhs(t, y)
        a, b = y[0], y[2]
        rho = red.signal(y)[0]
        wr = red.amplitudes(y)[1]
        dtau = a ** (K - k) * b ** (k - 2)
        A = z[-2]
        # C, tau, A = int (rho - omega_r) dtau, G = int exp(-(k-2) A) dtau
        return np.concatenate([v, [a ** (K - k), dtau, (rho - wr) * dtau, np.exp(-(k - 2) * A) * dtau]])

    sol = solve_ivp(rhs_aug, (0, 400), y0, method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True)
    eG, eb = 0.0, 0.0
    for t in np.linspace(1, 400, 400):
        z = sol.sol(t)
        C, tau, A, G = z[-4:]
        eG = max(eG, abs(G - b0 ** (k - 2) * C) / (b0 ** (k - 2) * C))
        eb = max(eb, abs(np.log(z[2] / b0) - A))
    zT = sol.sol(400)
    check("G(tau(t)) = b(0)^(k-2) C(t) and log(b/b(0)) = int (rho - omega_r) dtau along a trajectory",
          eG < 1e-9 and eb < 1e-9, f"errors {eG:.1e}, {eb:.1e}; final <U_r,W> = {red.overlap_r(zT[:-4]):.3f}")
    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(20):
        y = np.concatenate([rng.uniform(0.2, 1.5, 3)] + [unit(rng.standard_normal(3)) for _ in range(2)])
        v = red.rhs(0, y)
        a, x, b = y[:3]
        wj, wr = red.amplitudes(y)
        rho = red.signal(y)[0]
        lhs_x, rhs_x = -(k - 2) * x ** (-(k - 1)) * v[1], -(k - 2) * (sj - wj) * a ** (K - k)
        lhs_b, rhs_b = -(k - 2) * b ** (-(k - 1)) * v[2], -(k - 2) * (rho - wr) * a ** (K - k)
        worst = max(worst, abs(lhs_x - rhs_x), abs(lhs_b - rhs_b))
    check("d x^-(k-2)/dt = -(k-2)(s_j - omega_j) a^(K-k) and d b^-(k-2)/dt = -(k-2)(rho - omega_r) a^(K-k)",
          worst < 1e-12, f"max error {worst:.1e} at 20 random states")

# ---------------------------------------------------------------------------------------------------------------
if 3 in SECTIONS:
    section(3, "Orientation flow of a rank-one dormant channel: closed form")
    rng = np.random.default_rng(7)
    sr = 0.7

    def orient(p0, dims, T):
        targets = [np.eye(dd)[0] for dd in dims]
        ws0 = []
        for p, dd in zip(p0, dims):
            v = rng.standard_normal(dd)
            v[0] = 0.0
            v = unit(v) * np.sqrt(1 - p * p)
            v[0] = p
            ws0.append(v)
        split = np.cumsum(dims)[:-1]

        def rhs(tau, y):
            ws = np.split(y[:-1], split)
            p = np.array([w[0] for w in ws])
            out = []
            for g, w in enumerate(ws):
                grad = sr * np.prod(np.delete(p, g)) * targets[g]
                out.append(grad - (grad @ w) * w)
            return np.concatenate(out + [[sr * np.prod(p)]])
        s = solve_ivp(rhs, (0, T), np.concatenate(ws0 + [[0.0]]), method="DOP853", rtol=1e-12, atol=1e-14,
                      dense_output=True)
        return s, split

    for p0 in ([0.3, 0.5], [0.2, 0.45, 0.6], [0.3, 0.35, 0.5, 0.7]):
        m = len(p0)
        dims = list(rng.integers(3, 6, m))
        s, split = orient(p0, dims, 60.0)
        c = 1 - np.array(p0) ** 2
        err = 0.0
        for tau in np.linspace(0, 60, 301):
            y = s.sol(tau)
            p = np.array([w[0] for w in np.split(y[:-1], split)])
            err = max(err, np.abs((1 - p ** 2) - c * np.exp(-2 * y[-1])).max())
        check(f"m = {m}: 1 - p_g(tau)^2 = (1 - p_g(0)^2) E(tau), E = exp(-2 int rho)", err < 1e-9, f"max error {err:.1e}")
        for kappa in (1, 2, 3):
            num = quad(lambda tau: np.exp(-kappa * s.sol(tau)[-1]), 0, 60, limit=800)[0]
            form = kappa_integral(sr, p0, kappa)
            check(f"m = {m}, kappa = {kappa}: int_0^inf E^(kappa/2) dtau = (1/s_r) int_0^1 u^(kappa-1) "
                  f"prod (1 - c_g u^2)^(-1/2) du", abs(num - form) < 1e-7 * form, f"{num:.8f} vs {form:.8f}")
        lam_num = sr * 60 - s.sol(60)[-1]
        check(f"m = {m}: Lambda = lim (s_r tau - int rho) = int_0^1 (prod (1 - c_g u^2)^(-1/2) - 1) du/u",
              abs(lam_num - lambda_shift(p0)) < 1e-8, f"{lam_num:.8f} vs {lambda_shift(p0):.8f}")
        if m == 2:
            check("m = 2: Lambda = log(2/(p_1 + p_2))", abs(lambda_shift(p0) - np.log(2 / sum(p0))) < 1e-9)
            al, be = sum(p0) / 2, (p0[0] - p0[1]) / 2
            Z = lambda tau: al ** 2 * np.exp(2 * sr * tau) + be ** 2 * np.exp(-2 * sr * tau) + 1 - al ** 2 - be ** 2
            eZ = max(abs(np.exp(-2 * s.sol(tau)[-1]) * Z(tau) - 1) for tau in np.linspace(0, 30, 61))
            check("m = 2: E = 1/Z with Z of the dilation solution (Proposition 4.2 of the main note)", eZ < 1e-9)

    # equal overlaps: hypergeometric form; random-like overlaps p^2 = 1/d: large-dimension asymptotics
    from scipy.special import hyp2f1
    worst = 0.0
    for m in (2, 3, 5):
        for kappa in (1, 2, 3):
            for p in (0.2, 0.6):
                form = hyp2f1(m / 2, kappa / 2, kappa / 2 + 1, 1 - p * p) / (kappa * sr)
                worst = max(worst, abs(kappa_integral(sr, [p] * m, kappa) / form - 1))
    check("equal overlaps p: I_kappa = 2F1(m/2, kappa/2; kappa/2 + 1; 1 - p^2) / (kappa s_r)", worst < 1e-8,
          f"max relative error {worst:.1e}")
    rows = []
    for m in (3, 4):
        for kappa in (1, 3):
            d_ = 1e6
            val = sr * kappa_integral(sr, [d_ ** -0.5] * m, kappa)
            rows.append(abs(val / (d_ ** ((m - 2) / 2) / (m - 2)) - 1))
    check("overlaps p^2 = 1/d, m >= 3: s_r I_kappa = d^((m-2)/2)/(m-2) (1 + o(1)) (d = 1e6)", max(rows) < 5e-3,
          f"max relative deviation {max(rows):.1e}")
    diffs = [sr * kappa_integral(sr, [d_ ** -0.5] * 2, 1) - 0.5 * np.log(d_) for d_ in (1e4, 1e6)]
    check("overlaps p^2 = 1/d, m = 2: s_r I_kappa = (1/2) log d + O(1)", abs(diffs[1] - diffs[0]) < 1e-2,
          f"s_r I - log(d)/2 = {diffs[0]:.4f}, {diffs[1]:.4f}")

    # negative product of the initial overlaps: two branches
    for p0 in ([0.4, -0.15], [0.5, 0.3, -0.2]):
        m = len(p0)
        s, split = orient(p0, [4] * m, 80.0)
        c = 1 - np.array(p0) ** 2
        Emax = 1 / c.max()
        f = lambda E: E ** -0.5 / np.sqrt(np.abs(np.prod(1 - c * E)))
        form = (quad(f, 1, Emax, limit=500)[0] + quad(f, 0, Emax, limit=500)[0]) / (2 * sr)
        num = quad(lambda tau: np.exp(-s.sol(tau)[-1]), 0, 80, limit=1000)[0]
        check(f"m = {m}, P(0) < 0: I_1 = (1/(2 s_r)) (int_1^Emax + int_0^Emax) E^(-1/2) |prod (1 - c_g E)|^(-1/2) dE",
              abs(num - form) < 1e-6 * form, f"{num:.7f} vs {form:.7f}")

# ---------------------------------------------------------------------------------------------------------------
if 4 in SECTIONS:
    section(4, "Escape from the saddle (reduced equations)")
    sj, sr = 1.0, 0.6
    for (K, k, p0) in ((7, 3, [0.3, 0.55]), (6, 3, [0.4, 0.5, 0.6]), (9, 5, [1.0, 1.0]), (8, 4, [0.35, 0.6, 0.5, 0.8])):
        m = len(p0)
        red = Reduced(K, k, sj, sr, [np.eye(3)[1]] * m)
        w0 = [unit([0.0, p, np.sqrt(1 - p * p)]) for p in p0]
        I = kappa_integral(sr, p0, k - 2)
        pred_C, pred_t = I, I / sj ** ((K - k) / K)
        deltas = (1e-2, 1e-3) if k == 3 else (3e-2, 1e-2, 3e-3)
        rows = []
        for de in deltas:
            C, t, _ = escape(red, de, w0)
            rows.append((de, C * de ** (k - 2), t * de ** (k - 2)))
        print(f"  K = {K}, k = {k}, p(0) = {p0}: delta^(k-2) C_esc = {[f'{r[1]:.5f}' for r in rows]}, "
              f"delta^(k-2) t_esc = {[f'{r[2]:.5f}' for r in rows]} -> I = {pred_C:.5f}, I/s_j^((K-k)/K) = {pred_t:.5f}")
        ok = abs(rows[-1][1] - pred_C) < 1e-3 * pred_C and abs(rows[-1][2] - pred_t) < 2e-3 * pred_t \
            and abs(rows[-1][1] - pred_C) < abs(rows[0][1] - pred_C) + 1e-12
        check(f"K = {K}, k = {k}, m = {m}: delta^(k-2) t_esc -> I_(k-2) / s_j^((K-k)/K)", ok)
    # k = 2: misalignment shifts the escape by Lambda / (s_r s_j^((K-2)/K))
    K, k = 6, 2
    p0 = [0.15, 0.35]
    red = Reduced(K, k, sj, sr, [np.eye(4)[1]] * 2)
    w_mis = [unit([0.0, p, np.sqrt(1 - p * p), 0.0]) for p in p0]
    w_al = [np.eye(4)[1]] * 2
    diffs = []
    for de in (1e-3, 1e-5, 1e-7):
        _, t1, _ = escape(red, de, w_mis)
        _, t0, _ = escape(red, de, w_al)
        diffs.append((t1 - t0) * sr * sj ** ((K - 2) / K))
    lam = np.log(2 / sum(p0))
    print(f"  k = 2: (t_mis - t_al) s_r s_j^((K-2)/K) = {np.round(diffs, 6)} -> Lambda = log(2/(p1+p2)) = {lam:.6f}")
    check("k = 2: misalignment delays the escape by Lambda/(s_r s_j^((K-2)/K)), Lambda = log(2/(p_1+p_2))",
          abs(diffs[-1] - lam) < 1e-3 and abs(diffs[-1] - lam) <= abs(diffs[0] - lam) + 1e-9)
    # unbalanced, aligned dormant amplitudes on the full HT tree: budget (1/(2 s_r)) int prod (beta + b_v(0)^2)^(-1/2)
    net = ht4(2, 2)
    pij, pir = [0, 0, 0, 0], [1, 1, 0, 0]
    S_nodes, S_edges = steiner(net.edges, {3, 4})
    errs = []
    for de in (1e-2, 1e-3):
        b0 = {1: de, 3: 2.0 * de, 4: 0.6 * de}
        cores = [np.zeros(sh) for sh in net.shapes]
        set_channel(net, cores, net.node_index(pij), range(7), {}, sj ** (1 / 7))
        set_channel(net, cores, net.node_index(pir), S_nodes, {e: 1 for e in S_edges}, b0)
        evu = lambda t, th: np.sum(net.W(net.unpack(th)) * net.basis(pir)) - 0.5 * sr
        evu.terminal, evu.direction = True, 1
        t_full = solve_ivp(net.rhs_fn(sj * net.basis(pij) + sr * net.basis(pir)), (0, 1e8), net.pack(cores),
                           method="LSODA", rtol=1e-10, atol=1e-16, events=evu).t_events[0][0]
        # (1/(2 s_r)) int_0^inf prod_v (beta + b_v^2)^(-1/2) d beta: finite part plus the tail, expanded in 1/beta
        bsq = np.array([bb ** 2 for bb in b0.values()])
        beta1 = 1e6 * bsq.max()
        head = quad(lambda be: np.prod((be + bsq) ** -0.5), 0, beta1, limit=500,
                    points=list(bsq.max() * np.logspace(-2, 5, 8)))[0]
        kk = len(bsq)
        tail = beta1 ** (1 - kk / 2) / (kk / 2 - 1) - 0.5 * bsq.sum() * beta1 ** (-kk / 2) / (kk / 2)
        B = (head + tail) / (2 * sr)
        errs.append(abs(t_full * sj ** (4 / 7) / B - 1))
    check("unbalanced aligned dormant amplitudes (HT tree, b(0) = delta (1, 2, 0.6)): t_esc s_j^((K-k)/K) -> "
          "(1/(2 s_r)) int prod (beta + b_v(0)^2)^(-1/2) d beta, error O(delta)", errs[-1] < 5e-3 and errs[0] / errs[1] > 5,
          f"relative errors {errs[0]:.1e} (delta = 1e-2), {errs[1]:.1e} (delta = 1e-3)")
    # k = 2, unequal aligned dormant amplitudes (b_1, b_2) act as their arithmetic mean
    netm = mps(6, 2, 2)
    pij2, pir2 = [0] * 6, [0, 0, 1, 1, 0, 0]
    Wm = sj * netm.basis(pij2) + sr * netm.basis(pir2)
    evm = lambda t, th: np.sum(netm.W(netm.unpack(th)) * netm.basis(pir2)) - 0.5 * sr
    evm.terminal, evm.direction = True, 1

    def t_esc_mps(b2, b3):
        cores = [np.zeros(sh) for sh in netm.shapes]
        set_channel(netm, cores, netm.node_index(pij2), range(6), {}, sj ** (1 / 6))
        set_channel(netm, cores, netm.node_index(pir2), [2, 3], {(2, 3): 1}, {2: b2, 3: b3})
        return solve_ivp(netm.rhs_fn(Wm), (0, 1e6), netm.pack(cores), method="LSODA", rtol=1e-10, atol=1e-16,
                         events=evm).t_events[0][0]
    gaps = [t_esc_mps(de, 3 * de) - t_esc_mps(2 * de, 2 * de) for de in (1e-3, 1e-5)]
    check("k = 2, unequal dormant amplitudes (delta, 3 delta) escape like equal ones at their mean 2 delta",
          abs(gaps[1]) < 0.1 * abs(gaps[0]) and abs(gaps[1]) < 1e-3, f"time differences {gaps[0]:.2e}, {gaps[1]:.2e}")
    _, t3, _ = escape(red, 1e-3, w_al)
    _, t5, _ = escape(red, 1e-5, w_al)
    slope = (t5 - t3) / np.log(100)
    check("k = 2: t_esc grows like log(1/delta) / (s_r s_j^((K-2)/K))",
          abs(slope * sr * sj ** ((K - 2) / K) - 1) < 1e-3, f"slope x s_r s_j^((K-2)/K) = {slope * sr:.5f}")

# ---------------------------------------------------------------------------------------------------------------
if 5 in SECTIONS:
    section(5, "Plateau from a small initialization (reduced equations)")
    sj, sr = 1.0, 0.5
    for (K, k) in ((7, 3), (6, 4), (9, 5)):
        red = Reduced(K, k, sj, sr, [np.eye(2)[1]] * 2)
        w0 = [np.eye(2)[1]] * 2
        coefC = (1 / sr - 1 / sj) / (k - 2)
        coeft = coefC / sj ** ((K - k) / K)
        sigs = (1e-2, 1e-3) if k == 3 else ((3e-2, 1e-2, 3e-3) if k == 4 else (6e-2, 3e-2, 1.5e-2))
        rows = []
        for sig in sigs:
            _, dC, dt, _, _ = plateau(red, sig, sig, w0)
            rows.append((sig, dC * sig ** (k - 2), dt * sig ** (k - 2)))
        print(f"  aligned, K = {K}, k = {k}: sigma^(k-2) dC = {[f'{r[1]:.5f}' for r in rows]} -> {coefC:.5f}; "
              f"sigma^(k-2) dt = {[f'{r[2]:.5f}' for r in rows]} -> {coeft:.5f}")
        check(f"aligned, K = {K}, k = {k}: sigma^(k-2) Delta C -> (1/s_r - 1/s_j)/(k-2), error decreasing",
              abs(rows[-1][1] - coefC) < 2e-3 * coefC and abs(rows[-1][1] - coefC) < abs(rows[0][1] - coefC))
        check(f"aligned, K = {K}, k = {k}: sigma^(k-2) Delta t -> (1/s_r - 1/s_j)/((k-2) s_j^((K-k)/K)), "
              f"error decreasing",
              abs(rows[-1][2] - coeft) < 3e-3 * coeft and abs(rows[-1][2] - coeft) < abs(rows[0][2] - coeft))
    K, k = 6, 2
    red = Reduced(K, k, sj, sr, [np.eye(2)[1]] * 2)
    d1 = plateau(red, 1e-3, 1e-3, [np.eye(2)[1]] * 2)[2]
    d2 = plateau(red, 1e-5, 1e-5, [np.eye(2)[1]] * 2)[2]
    slope = (d2 - d1) / np.log(1e2)
    pred = (1 / sr - 1 / sj) / sj ** ((K - 2) / K)
    check("aligned, k = 2: Delta t grows like log(1/sigma), slope (1/s_r - 1/s_j)/s_j^((K-2)/K)",
          abs(slope - pred) < 2e-3 * pred, f"slope {slope:.5f} vs {pred:.5f}")
    # non-aligned dormant channels
    for (K, k, p0, d) in ((7, 3, [0.25, 0.4], 3), (6, 3, [0.3, 0.5, 0.45], 3), (8, 4, [0.3, 0.4, 0.5], 4)):
        m = len(p0)
        red = Reduced(K, k, sj, sr, [np.eye(d)[1]] * m)
        w0 = [unit(np.r_[0.0, p, np.sqrt(1 - p * p), np.zeros(d - 3)]) for p in p0]
        I = kappa_integral(sr, p0, k - 2)
        coefC = I - 1 / ((k - 2) * sj)
        coeft = coefC / sj ** ((K - k) / K)
        sigs = (1e-2, 1e-3) if k == 3 else (3e-2, 1e-2, 3e-3)
        rows = []
        for sig in sigs:
            _, dC, dt, _, _ = plateau(red, sig, sig, w0)
            rows.append((sig, dC * sig ** (k - 2), dt * sig ** (k - 2)))
        print(f"  non-aligned, K = {K}, k = {k}, m = {m}, p(0) = {p0}: I = {I:.4f} (aligned {1 / ((k - 2) * sr):.4f}); "
              f"sigma^(k-2) dC = {[f'{r[1]:.4f}' for r in rows]} -> {coefC:.4f}; "
              f"sigma^(k-2) dt = {[f'{r[2]:.4f}' for r in rows]} -> {coeft:.4f}")
        check(f"non-aligned, K = {K}, k = {k}, m = {m}: sigma^(k-2) Delta C -> I_(k-2) - 1/((k-2) s_j)",
              abs(rows[-1][1] - coefC) < 2e-3 * coefC and abs(rows[-1][1] - coefC) < abs(rows[0][1] - coefC))
        check(f"non-aligned, K = {K}, k = {k}, m = {m}: sigma^(k-2) Delta t -> (I_(k-2) - 1/((k-2) s_j))/s_j^((K-k)/K)",
              abs(rows[-1][2] - coeft) < 3e-3 * coeft)

# ---------------------------------------------------------------------------------------------------------------
if 6 in SECTIONS:
    section(6, "Plateau on the full HT tree with a non-aligned dormant channel")
    sj, sr, d, sig = 1.0, 0.5, 3, 1e-2
    net = ht4(d, 2)
    pij, pir = [0, 0, 0, 0], [1, 1, 0, 0]
    Wstar = sj * net.basis(pij) + sr * net.basis(pir)
    q = net.node_index(pij)
    S_nodes, S_edges = steiner(net.edges, {3, 4})
    p0 = [0.25, 0.4]
    w = {3: unit([0.0, p0[0], np.sqrt(1 - p0[0] ** 2)]), 4: unit([0.0, p0[1], -np.sqrt(1 - p0[1] ** 2)])}
    th0 = net.pack(two_channel_cores(net, q, S_nodes, S_edges, np.sqrt(2) * sig, sig, sig, w))
    Uj, Ur = net.basis(pij), net.basis(pir)
    rhs = net.rhs_fn(Wstar)

    def arc(s, th):
        F = rhs(s, th)
        return F / np.linalg.norm(F)
    ev1 = lambda s, th: np.sum(net.W(net.unpack(th)) * Uj) - 0.5 * sj
    ev1.terminal, ev1.direction = True, 1
    ph1 = solve_ivp(arc, (0, 50), th0, method="DOP853", rtol=1e-11, atol=1e-14, events=ev1)
    ev2 = lambda t, th: np.sum(net.W(net.unpack(th)) * Ur) - 0.5 * sr
    ev2.terminal, ev2.direction = True, 1
    ph2 = solve_ivp(rhs, (0, 1e5), ph1.y_events[0][0], method="LSODA", rtol=1e-10, atol=1e-14, events=ev2)
    dt_full = ph2.t_events[0][0]
    red = Reduced(7, 3, sj, sr, [np.eye(d)[1]] * 2)
    _, dC, dt_red, _, _ = plateau(red, sig, sig, [w[3], w[4]])
    I = kappa_integral(sr, p0, 1)
    print(f"  full HT: Delta t = {dt_full:.4f}; reduced: {dt_red:.4f}; leading order sigma^-1 (I - 1/s_j)/s_j^(4/7) "
          f"= {(I - 1 / sj) / sig:.4f}")
    check("full HT tree (sigma = 1e-2) reproduces the reduced plateau", abs(dt_full - dt_red) < 1e-5 * dt_red,
          f"relative difference {abs(dt_full - dt_red) / dt_red:.1e}")

# ---------------------------------------------------------------------------------------------------------------
if 7 in SECTIONS:
    section(7, "Transition: rebalancing law, duration, approximations")
    sj = 1.0
    for (K, k, sr) in ((7, 3, 0.5), (6, 2, 0.3), (9, 5, 0.8)):
        red = Reduced(K, k, sj, sr, [np.eye(2)[1]] * 2)
        a, x = saddle_state(K, k, sj, 1e-3)
        _, _, y = escape(red, 1e-3, [np.eye(2)[1]] * 2)
        s = solve_ivp(red.rhs, (0, 300), y, method="LSODA", rtol=1e-11, atol=1e-14)
        aT, xT, bT = s.y[:3, -1]
        aP, xP, bP = rebalanced(K, k, sj, sr)
        check(f"K = {K}, k = {k}, s_r = {sr}: final state = rebalancing law a* = (s_j^(2/k) + s_r^(2/k))^(k/(2K))",
              max(abs(aT - aP), abs(xT - xP), abs(bT - bP)) < 1e-6,
              f"a* = {aP:.5f} (saddle {sj ** (1 / K):.5f}), x* = {xP:.5f}, b* = {bP:.5f}")
    # full HT network: final core amplitudes
    net = ht4(2, 2)
    sr = 0.5
    pij, pir = [0, 0, 0, 0], [1, 1, 0, 0]
    S_nodes, S_edges = steiner(net.edges, {3, 4})
    a, x = saddle_state(7, 3, sj, 1e-2)
    th0 = net.pack(two_channel_cores(net, net.node_index(pij), S_nodes, S_edges, a, x, 1e-2,
                                     {3: np.eye(2)[1], 4: np.eye(2)[1]}))
    s = solve_ivp(net.rhs_fn(sj * net.basis(pij) + sr * net.basis(pir)), (0, 600), th0, method="LSODA",
                  rtol=1e-11, atol=1e-14)
    cores = net.unpack(s.y[:, -1])
    aP, xP, bP = rebalanced(7, 3, sj, sr)
    got = (abs(cores[2][0, 0, 0]), abs(cores[1][0, 0, 0]), abs(cores[1][0, 1, 1]))   # node 2 off S; node 1 in S
    check("full HT tree: core amplitudes after the transition follow the rebalancing law",
          max(abs(got[0] - aP), abs(got[1] - xP), abs(got[2] - bP)) < 1e-6,
          f"(a, x, b) = ({got[0]:.5f}, {got[1]:.5f}, {got[2]:.5f}) vs ({aP:.5f}, {xP:.5f}, {bP:.5f})")

    def transition_profile(red, delta, T=200.0, n=20001):
        _, _, y = escape(red, delta, [np.eye(2)[1]] * 2, eta=0.01)
        s = solve_ivp(red.rhs, (0, T), y, method="LSODA", rtol=1e-11, atol=1e-14, dense_output=True)
        ts = np.linspace(0, T, n)
        return ts, np.array([red.amplitudes(s.sol(tt))[1] for tt in ts]) / red.sr

    def crossing(ts, om, level):
        i = np.argmax(om >= level)
        return ts[i - 1] + (level - om[i - 1]) * (ts[i] - ts[i - 1]) / (om[i] - om[i - 1])

    K, k, sr = 7, 3, 0.5
    red = Reduced(K, k, sj, sr, [np.eye(2)[1]] * 2)
    durs = []
    for de in (1e-3, 1e-4, 1e-5):
        ts, om = transition_profile(red, de)
        durs.append(crossing(ts, om, 0.9) - crossing(ts, om, 0.1))
    print(f"  K = 7, k = 3, s_r = 0.5: 10-90% duration of the transition for delta = 1e-3, 1e-4, 1e-5: "
          f"{np.round(durs, 5)}")
    check("the transition takes a time independent of delta, to O(delta^2)", abs(durs[-1] - durs[-2]) < 1e-3 * durs[-1])

    def durations(K, k, sj, sr, delta=1e-6):
        """10-90% duration of the transition: exact reduced flow, adiabatic reduction, frozen background."""
        red = Reduced(K, k, sj, sr, [np.eye(2)[1]] * 2)
        _, _, y = escape(red, delta, [np.eye(2)[1]] * 2, eta=0.01)

        def level_events(fun):
            evs = []
            for lev in (0.1, 0.9):
                f = (lambda lev: (lambda t, z: fun(z) - lev * sr))(lev)
                f.direction = 1
                evs.append(f)
            evs[1].terminal = True
            return evs
        s = solve_ivp(red.rhs, (0, 1e8), y, method="LSODA", rtol=1e-11, atol=1e-15,
                      events=level_events(lambda z: red.amplitudes(z)[1]))
        d_exact = s.t_events[1][0] - s.t_events[0][0]

        def a_of_b(b):
            xx = brentq(lambda x: (x * x + b * b) ** ((K - k) / 2) * x ** k - sj, 1e-9, 2, xtol=1e-15)
            return np.sqrt(xx * xx + b * b)
        om_b = lambda b: a_of_b(b) ** (K - k) * b ** k
        b0 = brentq(lambda b: om_b(b) - 0.01 * sr, 1e-8, 2, xtol=1e-15)
        sa = solve_ivp(lambda t, z: [a_of_b(z[0]) ** (K - k) * z[0] ** (k - 1) * (sr - om_b(z[0]))], (0, 1e8), [b0],
                       method="LSODA", rtol=1e-11, atol=1e-15, events=level_events(lambda z: om_b(z[0])))
        d_ad = sa.t_events[1][0] - sa.t_events[0][0]
        c = sj ** ((K - k) / K)
        d_frozen = quad(lambda u: 1 / (u ** (2 - 2 / k) * (1 - u)), 0.1, 0.9)[0] / (k * c ** (2 / k) * sr ** (2 - 2 / k))
        return d_exact, d_ad, d_frozen

    errs_ad, errs_fr = [], []
    for ratio in (0.6, 0.1, 0.01, 0.001):
        d_exact, d_ad, d_frozen = durations(7, 3, sj, ratio * sj)
        errs_ad.append(abs(d_ad / d_exact - 1))
        errs_fr.append(abs(d_frozen / d_exact - 1))
        print(f"  K = 7, k = 3, s_r/s_j = {ratio}: 10-90% duration exact {d_exact:.5g}, adiabatic {d_ad:.5g} "
              f"({100 * (d_ad / d_exact - 1):+.3f}%), frozen depth-k law {d_frozen:.5g} "
              f"({100 * (d_frozen / d_exact - 1):+.2f}%); (s_r/s_j)^(2/k) = {ratio ** (2 / 3):.3f}")
    check("the adiabatic reduction gives the duration of the transition within 0.5% for s_r/s_j <= 0.6",
          max(errs_ad) < 5e-3, f"max error {100 * max(errs_ad):.3f}%")
    check("the error of the frozen depth-k law scales like (s_r/s_j)^(2/k)",
          abs(np.log(errs_fr[2] / errs_fr[3]) / np.log(10) - 2 / 3) < 0.1,
          f"error ratio between s_r/s_j = 0.01 and 0.001: 10^{np.log(errs_fr[2] / errs_fr[3]) / np.log(10):.3f}")

# ---------------------------------------------------------------------------------------------------------------
if 8 in SECTIONS:
    section(8, "Rotation block at the saddle: H_rot = s_j^((K-2)/K) (s_j I - M_1)")

    def hessian(rhs, th, h=1e-5):
        n = len(th)
        J = np.zeros((n, n))
        for i in range(n):
            e = np.zeros(n)
            e[i] = h
            J[:, i] = (rhs(0, th + e) - rhs(0, th - e)) / (2 * h)
        return -(J + J.T) / 2

    for (K, sj) in ((5, 1.0), (4, 1.7)):
        net = mps(K, 2, 1)
        for s_r in (0.6 * sj, 1.5 * sj):
            pir = [1] + [0] * (K - 2) + [1]
            Wstar = sj * net.basis([0] * K) + s_r * net.basis(pir)
            cores = [np.zeros(s) for s in net.shapes]
            set_channel(net, cores, net.node_index([0] * K), range(K), {}, sj ** (1 / K))
            ev = np.linalg.eigvalsh(hessian(net.rhs_fn(Wstar), net.pack(cores)))
            pred = sj ** ((K - 2) / K) * (sj - s_r)
            if s_r > sj:
                check(f"K = {K}, s_j = {sj}, s_r = {s_r:.2f} > s_j: lowest Hessian eigenvalue s_j^((K-2)/K)(s_j - s_r)",
                      abs(ev.min() - pred) < 1e-6, f"{ev.min():.6f} vs {pred:.6f}")
            else:
                check(f"K = {K}, s_j = {sj}, s_r = {s_r:.2f} < s_j: Hessian positive semidefinite, and "
                      f"s_j^((K-2)/K)(s_j - s_r) is an eigenvalue", ev.min() > -1e-8 and np.abs(ev - pred).min() < 1e-6)

# ---------------------------------------------------------------------------------------------------------------
if 9 in SECTIONS:
    section(9, "Two candidates at a saddle: locality against weight")
    K, d, sj = 6, 2, 1.0
    net = mps(K, d, 2)
    pij = [0] * 6
    comps = {"r1": ([0, 0, 0, 1, 1, 0], 0.1), "r2": ([1, 0, 1, 0, 0, 0], 0.8)}
    Wstar = sj * net.basis(pij) + sum(s * net.basis(p) for p, s in comps.values())
    q = net.node_index(pij)
    info = {}
    for name, (pi, s) in comps.items():
        S_nodes, S_edges = steiner(net.edges, differing_nodes(net, pij, pi))
        info[name] = (pi, s, S_nodes, S_edges, net.node_index(pi))
    print(f"  r1: k = {len(info['r1'][2])}, s = 0.1; r2: k = {len(info['r2'][2])}, s = 0.8")

    def full_escape(delta, which):
        """Full flow from the balanced saddle with the dormant channels in `which` (amplitude delta)."""
        S_all = set().union(*[info[n][2] for n in which])
        a, x = saddle_state(K, len(S_all), sj, delta)
        cores = [np.zeros(s) for s in net.shapes]
        set_channel(net, cores, q, range(K), {}, {v: (x if v in S_all else a) for v in range(K)})
        for n in which:
            pi, s, S_nodes, S_edges, idx = info[n]
            set_channel(net, cores, idx, S_nodes, {e: 1 for e in S_edges}, delta)
        evs = []
        for n in ("r1", "r2"):
            U, s = net.basis(info[n][0]), info[n][1]
            f = (lambda U, s: (lambda t, th: np.sum(net.W(net.unpack(th)) * U) - 0.5 * s))(U, s)
            f.direction = 1
            evs.append(f)
        sol = solve_ivp(net.rhs_fn(Wstar), (0, 4e4), net.pack(cores), method="LSODA", rtol=1e-10, atol=1e-14,
                        events=evs)
        return [ev[0] if len(ev) else np.inf for ev in sol.t_events]

    rows, agree, close = [], 0, 0
    deltas = (0.1, 0.05, 0.03, 0.02, 0.01, 0.005)
    for de in deltas:
        t1 = full_escape(de, ["r1"])[0]
        t2 = full_escape(de, ["r2"])[1]
        tb = full_escape(de, ["r1", "r2"])
        first = "r1" if tb[0] < tb[1] else "r2"
        pred_first = "r1" if t1 < t2 else "r2"
        agree += first == pred_first
        close += abs(min(tb) - min(t1, t2)) < 2e-2 * min(t1, t2)
        rows.append((de, t1, t2, tb, first))
        print(f"  delta = {de:6.3f}: alone r1 {t1:9.2f}, alone r2 {t2:9.2f} | together r1 {tb[0]:9.2f}, "
              f"r2 {tb[1]:9.2f} -> first: {first}")
    red1 = Reduced(K, 2, sj, 0.1, [np.eye(2)[1]] * 2)
    red2 = Reduced(K, 3, sj, 0.8, [np.eye(2)[1], np.eye(2)[0], np.eye(2)[1]])
    errs = []
    for de, t1, t2, _, _ in rows:
        errs.append(abs(escape(red1, de, [np.eye(2)[1]] * 2)[1] - t1) / t1)
        errs.append(abs(escape(red2, de, [np.eye(2)[1], np.eye(2)[0], np.eye(2)[1]])[1] - t2) / t2)
    check("single candidates: the full flow reproduces the escape times of the reduced equations",
          max(errs) < 1e-6, f"max relative difference {max(errs):.1e}")
    check("together, the first escape is the candidate with the smaller single-candidate escape time",
          agree == len(deltas), f"{agree}/{len(deltas)}")
    check("together, the first escape time equals the single-candidate one within 2%", close == len(deltas),
          f"{close}/{len(deltas)}")
    firsts = [r[4] for r in rows]
    check("the stronger, less local r2 escapes first for large delta, the weaker, local r1 for small delta",
          firsts[0] == "r2" and firsts[-1] == "r1", " -> ".join(firsts))
    # after r1 is learned, the rebalancing slows the clock of the disjoint candidate r2 by R = f^(-k_2)
    k1, k2 = len(info["r1"][2]), len(info["r2"][2])
    R = (1 + (0.1 / sj) ** (2 / k1)) ** (-k1 * k2 / (2 * K))
    worst = 0.0
    for de, t1, t2, tb, first in rows:
        if first == "r1" and de <= 0.03:
            pred = tb[0] + (t2 - tb[0]) / R
            worst = max(worst, abs(tb[1] / pred - 1))
            print(f"  delta = {de}: r2 after r1 measured {tb[1]:.3f}, predicted t_r1 + (t_r2 alone - t_r1)/R = {pred:.3f}"
                  f" (R = {R:.5f}; without the clock change {t2:.3f})")
    check("the delay of r2 is predicted by the slower clock R = (1 + (s_r1/s_j)^(2/k_1))^(-k_1 k_2/(2K))",
          worst < 2e-3, f"max relative error {worst:.1e}")

# ---------------------------------------------------------------------------------------------------------------
TREES = {"MPS (4 cores)": lambda chi: mps(4, 2, chi), "binary tree (3 cores)": lambda chi: binary4(2, chi),
         "HT tree (7 cores)": lambda chi: ht4(2, chi)}
TARGET = {"j": ([0, 0, 0, 0], 1.0), "r1": ([0, 1, 1, 0], 0.3), "r2": ([1, 0, 0, 1], 0.6)}


def run_with_clocks(net, th0, T_end, clock_entries):
    """Full flow with, as extra states, the clock C_r of every candidate: the integral of the product of the
    learned entries that the dormant channel of r uses outside its Steiner subtree."""
    Wstar = sum(s * net.basis(p) for p, s in TARGET.values())
    base = net.rhs_fn(Wstar)
    n, names = len(th0), list(clock_entries)

    def rates(th):
        cores = net.unpack(th)
        return np.array([np.prod([abs(cores[v][idx]) for v, idx in clock_entries[nm]]) for nm in names])

    def rhs(t, z):
        return np.concatenate([base(t, z[:n]), rates(z[:n])])
    evs = []
    for p, s in TARGET.values():
        U = net.basis(p)
        f = (lambda U, s: (lambda t, z: np.sum(net.W(net.unpack(z[:n])) * U) - 0.5 * s))(U, s)
        f.direction = 1
        evs.append(f)
    sol = solve_ivp(rhs, (0, T_end), np.concatenate([th0, np.zeros(len(names))]), method="LSODA", rtol=1e-10,
                    atol=1e-15, events=evs, dense_output=True)
    ta = {nm: (ev[0] if len(ev) else np.inf) for nm, ev in zip(TARGET, sol.t_events)}
    clocks = {nm: (dict(zip(names, ye[0][n:])) if len(ye) else None) for nm, ye in zip(TARGET, sol.y_events)}
    return ta, clocks, (lambda t: dict(zip(names, rates(sol.sol(t)[:n]))))


def tree_info(net):
    pij = TARGET["j"][0]
    return {n: steiner(net.edges, differing_nodes(net, pij, TARGET[n][0])) for n in ("r1", "r2")}


def clock_entries_of(net, inf):
    qj = net.node_index(TARGET["j"][0])
    return {n: [(v, ((qj[v],) if v in net.phys else ()) + (0,) * len(net.inc[v])) for v in range(net.n)
                if v not in inf[n][0]] for n in inf}


def budget(k, s, sig):
    return np.log(1 / sig) / s if k == 2 else sig ** -(k - 2) / ((k - 2) * s)


def plateau_prediction(K, k, s, sj, sig):
    if k == 2:
        return (1 / s - 1 / sj) * np.log(1 / sig) / sj ** ((K - 2) / K)
    return sig ** -(k - 2) * (1 / s - 1 / sj) / ((k - 2) * sj ** ((K - k) / K))


def aligned_init(net, inf, sig):
    cores = [np.zeros(s) for s in net.shapes]
    set_channel(net, cores, net.node_index(TARGET["j"][0]), range(net.n), {}, sig)
    used = {e: 1 for e in net.edges}
    for n in ("r1", "r2"):
        S_nodes, S_edges = inf[n]
        eidx = {}
        for e in S_edges:
            eidx[e] = used[e]
            used[e] += 1
        set_channel(net, cores, net.node_index(TARGET[n][0]), S_nodes, eidx, sig)
    return net.pack(cores)


if 10 in SECTIONS:
    section(10, "Same target on three trees, aligned small initialization")
    sj = TARGET["j"][1]
    for tname, make in TREES.items():
        net = make(3)
        K = net.n
        inf = tree_info(net)
        ks = {n: len(inf[n][0]) for n in inf}
        sigs = {"MPS (4 cores)": (0.05, 0.025, 0.0125), "binary tree (3 cores)": (0.05, 0.025, 0.0125),
                "HT tree (7 cores)": (0.1, 0.07, 0.05)}[tname]
        pred_order = sorted(("r1", "r2"), key=lambda n: plateau_prediction(K, ks[n], TARGET[n][1], sj, sigs[-1]))
        print(f"  {tname}: k_r1 = {ks['r1']}, k_r2 = {ks['r2']}; predicted order j, {pred_order[0]}, {pred_order[1]}")
        rows = []
        for sig in sigs:
            t0 = time.time()
            ta, clocks, rate_at = run_with_clocks(net, aligned_init(net, inf, sig), 50 * sig ** -(K - 2) + 2e5,
                                                  clock_entries_of(net, inf))
            rows.append((sig, ta, clocks, rate_at))
            order = sorted(ta, key=ta.get)
            cl = ", ".join(f"C_{n}(t_{n}) = {clocks[n][n]:.5g} (budget {budget(ks[n], TARGET[n][1], sig):.5g})"
                           for n in ("r1", "r2"))
            print(f"    sigma = {sig:.4f}: t_j = {ta['j']:.5g} [x (K-2) s_j sigma^(K-2) = "
                  f"{ta['j'] * (K - 2) * sj * sig ** (K - 2):.3f}], Delta t_r1 = {ta['r1'] - ta['j']:.5g}, "
                  f"Delta t_r2 = {ta['r2'] - ta['j']:.5g}; order {', '.join(order)}; {cl}  [{time.time() - t0:.0f} s]")
        check(f"{tname}: acquisition order j, {pred_order[0]}, {pred_order[1]} at every sigma",
              all(sorted(ta, key=ta.get) == ["j"] + pred_order for _, ta, _, _ in rows))
        sig, ta, clocks, rate_at = rows[-1]
        for n in ("r1", "r2"):
            if ks[n] >= 3:
                B = budget(ks[n], TARGET[n][1], sig)
                if ks[n] >= 4:
                    check(f"{tname}: {n} (k = {ks[n]}) is acquired when its clock reaches its budget "
                          f"sigma^-(k-2)/((k-2) s_r), within 0.02%", abs(clocks[n][n] / B - 1) < 2e-4,
                          f"C = {clocks[n][n]:.6g}, budget {B:.6g}")
                else:
                    check(f"{tname}: {n} (k = 3) is acquired when its clock reaches its budget, up to O(1): "
                          f"|C - B| < 5", abs(clocks[n][n] - B) < 5, f"C = {clocks[n][n]:.5g}, budget {B:.5g}")
            else:
                (s1, _, c1, _), (s2, _, c2, _) = rows[0], rows[-1]
                slope = (c2[n][n] - c1[n][n]) / np.log(s1 / s2)
                check(f"{tname}: {n} (k = 2): the clock at acquisition grows like log(1/sigma)/s_r",
                      abs(slope * TARGET[n][1] - 1) < 0.03, f"slope x s_r = {slope * TARGET[n][1]:.4f}")
        if tname.startswith("binary"):
            for n in ("r1", "r2"):
                p_ = plateau_prediction(K, ks[n], TARGET[n][1], sj, sig)
                check(f"{tname}: no core is shared, and the plateau before {n} is sigma^-1 (1/s - 1/s_j)",
                      abs((ta[n] - ta["j"]) / p_ - 1) < 0.03, f"measured {ta[n] - ta['j']:.4g}, predicted {p_:.4g}")
        if tname.startswith("HT"):
            second, third = pred_order
            k2 = ks[second]
            S2, S3 = inf[second][0], inf[third][0]
            f = (1 + (TARGET[second][1] / sj) ** (2 / k2)) ** (k2 / (2 * K))
            R = f ** ((K - len(S2 | S3)) - len(S2 - S3) * (K - k2) / k2)
            before = rate_at(ta["j"] + 0.5 * (ta[second] - ta["j"]))[third]
            after = rate_at(ta[second] + 0.5 * (ta[third] - ta[second]))[third]
            check(f"{tname}: after {second} is learned, the clock of {third} runs slower by the factor R of the "
                  f"rebalancing law", abs(after / before / R - 1) < 5e-3,
                  f"measured ratio {after / before:.5f}, R = {R:.5f}")

# ---------------------------------------------------------------------------------------------------------------
if 11 in SECTIONS:
    section(11, "Random small initialization (learned component favoured)")
    sj = TARGET["j"][1]

    def random_init(net, sig, seed):
        r_ = np.random.default_rng(1000 + seed)
        cores = [sig * r_.standard_normal(s) / np.sqrt(2) for s in net.shapes]
        qj = net.node_index(TARGET["j"][0])
        for v in range(net.n):                      # favour the learned component: add sigma to its entries
            cores[v][((qj[v],) if v in net.phys else ()) + (0,) * len(net.inc[v])] += sig
        return net.pack(cores)

    for tname, chi, sig in (("MPS (4 cores)", 3, 0.02), ("binary tree (3 cores)", 3, 0.02),
                            ("HT tree (7 cores)", 3, 0.07), ("HT tree (7 cores)", 4, 0.07)):
        net = TREES[tname](chi)
        K = net.n
        inf = tree_info(net)
        ks = {n: len(inf[n][0]) for n in inf}
        rows = []
        for seed in range(5):
            ta, _, _ = run_with_clocks(net, random_init(net, sig, seed), 3e7, {})
            rows.append(ta)
            print(f"    {tname}, chi = {chi}, sigma = {sig}, seed {seed}: t_j = {ta['j']:.4g}, "
                  f"Delta t_r1 = {ta['r1'] - ta['j']:.4g}, Delta t_r2 = {ta['r2'] - ta['j']:.4g}; "
                  f"order {', '.join(sorted(ta, key=ta.get))}")
        if tname.startswith("MPS"):
            check("MPS: k_r1 = 2 < k_r2 = 4, and the weaker r1 is learned before r2 in every seed",
                  all(ta["j"] < ta["r1"] < ta["r2"] for ta in rows))
        elif tname.startswith("binary"):
            n_w = sum(ta["r2"] < ta["r1"] for ta in rows)
            print(f"    equal orders (k = 3): the stronger r2 is learned first in {n_w}/5 seeds")
        elif chi == 4:
            worst = max(max(ta["r1"], ta["r2"]) - ta["j"] for ta in rows)
            check("HT tree with a spare bond direction (chi = 4): every component is learned within "
                  "20 sigma^-3 after j", worst < 20 * sig ** -3, f"longest {worst:.4g}, sigma^-3 = {sig ** -3:.4g}")
        else:
            worst = max(max(ta["r1"], ta["r2"]) - ta["j"] for ta in rows)
            print(f"    HT tree, chi = 3 (the rank of the target on the root bonds): longest plateau {worst:.4g} "
                  f"= {worst * sig ** 3:.3g} sigma^-3")

# ---------------------------------------------------------------------------------------------------------------
if 12 in SECTIONS:
    section(12, "Path reuse: a dormant channel can branch off a learned component other than the first")
    K, d, sj, s1, s2 = 6, 2, 1.0, 0.3, 0.6
    net = mps(K, d, 3)
    pij, pi1, pi2 = [0] * 6, [1, 1, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0]
    Wstar = sj * net.basis(pij) + s1 * net.basis(pi1) + s2 * net.basis(pi2)
    U2 = net.basis(pi2)
    aP, xP, bP = rebalanced(K, 2, sj, s1)              # j and r1 learned, balanced, S_1 = {0, 1}
    print(f"  |St(V_r2(j))| = 4, |St(V_r2(r1))| = 3; rebalanced (a*, x*, b*) = ({aP:.4f}, {xP:.4f}, {bP:.4f})")

    def start(delta, path):
        cores = [np.zeros(sh) for sh in net.shapes]
        set_channel(net, cores, net.node_index(pij), range(K), {}, {v: (xP if v in (0, 1) else aP) for v in range(K)})
        set_channel(net, cores, net.node_index(pi1), [0, 1], {(0, 1): 1}, bP)
        if path == 3:      # branch off r1: new entries on cores 1, 2, 3, index 1 of edge (0, 1) is r1's
            set_channel(net, cores, net.node_index(pi2), [1, 2, 3], {(0, 1): 1, (1, 2): 1, (2, 3): 1}, delta)
        else:              # branch off j: new entries on cores 0, 1, 2, 3 with a fresh index on edge (0, 1)
            set_channel(net, cores, net.node_index(pi2), [0, 1, 2, 3], {(0, 1): 2, (1, 2): 1, (2, 3): 1}, delta)
        return net.pack(cores)

    ev = lambda t, th: np.sum(net.W(net.unpack(th)) * U2) - 0.5 * s2
    ev.terminal, ev.direction = True, 1
    deltas = (1e-2, 3e-3, 1e-3)
    res = {}
    for path in (3, 4):
        res[path] = [solve_ivp(net.rhs_fn(Wstar), (0, 1e9), start(de, path), method="LSODA", rtol=1e-10, atol=1e-16,
                               events=ev).t_events[0][0] for de in deltas]
    # without r1 in the target and with j alone learned, the branch off j has 4 new cores
    W2 = sj * net.basis(pij) + s2 * net.basis(pi2)
    a4, x4 = saddle_state(K, 4, sj, 1e-3)
    res["alone"] = []
    for de in deltas:
        a4, x4 = saddle_state(K, 4, sj, de)
        cores = [np.zeros(sh) for sh in net.shapes]
        set_channel(net, cores, net.node_index(pij), range(K), {}, {v: (x4 if v < 4 else a4) for v in range(K)})
        set_channel(net, cores, net.node_index(pi2), [0, 1, 2, 3], {(0, 1): 1, (1, 2): 1, (2, 3): 1}, de)
        res["alone"].append(solve_ivp(net.rhs_fn(W2), (0, 1e9), net.pack(cores), method="LSODA", rtol=1e-10,
                                      atol=1e-16, events=ev).t_events[0][0])
    expo = {key: np.log(v[-1] / v[0]) / np.log(10) for key, v in res.items()}
    pred3 = 1e-3 ** -1 / (s2 * bP * aP ** 2)
    pred4 = 1e-3 ** -2 / (2 * s2 * sj ** (2 / K))
    for key, label in ((3, "j and r1 learned, r2 dormant on the branch off r1 (3 new cores)"),
                       (4, "j and r1 learned, r2 dormant on the branch off j (4 new cores)"),
                       ("alone", "target without r1, j learned, r2 dormant on the branch off j (4 new cores)")):
        print(f"  {label}: t_esc = {[f'{t:.5g}' for t in res[key]]} for delta = 1e-2, 3e-3, 1e-3; "
              f"exponent {expo[key]:.3f}")
    check("branch off r1: t_esc ~ delta^-1, equal to budget / clock rate = 1/(delta s_r2 b* a*^2)",
          abs(expo[3] - 1) < 0.03 and abs(res[3][-1] / pred3 - 1) < 0.02,
          f"{res[3][-1]:.5g} vs {pred3:.5g}")
    check("branch off j while r1 is learned: the flow reroutes through r1, and t_esc ~ delta^-1",
          abs(expo[4] - 1) < 0.05)
    check("without r1, the same branch has order 4: t_esc ~ delta^-2, equal to 1/(2 delta^2 s_r2 s_j^(2/K))",
          abs(expo["alone"] - 2) < 0.03 and abs(res["alone"][-1] / pred4 - 1) < 0.02,
          f"{res['alone'][-1]:.5g} vs {pred4:.5g}")

# ---------------------------------------------------------------------------------------------------------------
def off_manifold_mask(net, q, S_nodes, S_edges):
    support = [np.zeros(sh, bool) for sh in net.shapes]
    for v in range(net.n):
        support[v][((q[v],) if v in net.phys else ()) + (0,) * len(net.inc[v])] = True
    for v in S_nodes:
        eidx = tuple(1 if e in S_edges else 0 for e in net.inc[v])
        support[v][((slice(None),) if v in net.phys else ()) + eidx] = True
    return ~net.pack(support)


if 13 in SECTIONS:
    section(13, "The condition on W': extra components invisible to the two-channel manifold")
    sj, sr = 1.0, 0.5
    net = ht4(3, 2)
    pij, pir = [0, 0, 0, 0], [1, 2, 0, 0]
    q = net.node_index(pij)
    S_nodes, S_edges = steiner(net.edges, {3, 4})
    off = off_manifold_mask(net, q, S_nodes, S_edges)
    w = {3: unit([0.0, 0.8, 0.6]), 4: unit([0.0, 0.5, 0.9])}
    th0 = net.pack(two_channel_cores(net, q, S_nodes, S_edges, np.sqrt(0.35 ** 2 + 0.3 ** 2), 0.35, 0.3, w))
    for extra, expect in (([0, 0, 1, 2], "invariant"), ([1, 2, 1, 0], "not invariant")):
        Vx = differing_nodes(net, pij, extra)
        n_out = len(Vx - S_nodes)
        Wstar = sj * net.basis(pij) + sr * net.basis(pir) + 0.4 * net.basis(extra)
        sol = solve_ivp(net.rhs_fn(Wstar), (0, 100), th0, method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True)
        dev = max(np.abs(sol.sol(t)[off]).max() for t in np.linspace(0, 100, 101))
        print(f"  extra component {''.join(map(str, extra))}: {n_out} core(s) of V_r' outside S; "
              f"max off-manifold entry {dev:.2e}")
        if expect == "invariant":
            check("an extra component that differs from U_j on two cores outside S leaves the manifold invariant",
                  dev == 0.0)
        else:
            check("an extra component that differs from U_j on one core outside S (and agrees with the dormant "
                  "channel on S) breaks the invariance", dev > 1e-3)

# ---------------------------------------------------------------------------------------------------------------
if 14 in SECTIONS:
    section(14, "Perturbations off the two-channel manifold")
    sj, sr, d = 1.0, 0.5, 3
    net = ht4(d, 3)                                    # chi = 3: room for a second dormant channel
    pij, pir = [0, 0, 0, 0], [1, 1, 0, 0]
    Wstar = sj * net.basis(pij) + sr * net.basis(pir)
    q = net.node_index(pij)
    S_nodes, S_edges = steiner(net.edges, {3, 4})
    p0 = [0.3, 0.45]
    w = {3: unit([0.0, p0[0], np.sqrt(1 - p0[0] ** 2)]), 4: unit([0.0, p0[1], np.sqrt(1 - p0[1] ** 2)])}
    delta = 1e-2
    a, x = saddle_state(7, 3, sj, delta)
    base = net.pack(two_channel_cores(net, q, S_nodes, S_edges, a, x, delta, w))
    Ur = net.basis(pir)
    ev = lambda t, th: np.sum(net.W(net.unpack(th)) * Ur) - 0.5 * sr
    ev.terminal, ev.direction = True, 1
    t_base = solve_ivp(net.rhs_fn(Wstar), (0, 1e6), base, method="LSODA", rtol=1e-11, atol=1e-16,
                       events=ev).t_events[0][0]
    T = t_base + 60.0
    ref = solve_ivp(net.rhs_fn(Wstar), (0, T), base, method="DOP853", rtol=1e-12, atol=1e-16, dense_output=True)
    rng = np.random.default_rng(21)
    xi = rng.standard_normal(len(base))
    xi /= np.linalg.norm(xi)
    rows = []
    for eps in (1e-8, 1e-6):
        pert = solve_ivp(net.rhs_fn(Wstar), (0, T), base + eps * xi, method="DOP853", rtol=1e-12, atol=1e-16,
                         dense_output=True)
        dW = max(np.abs(net.W(net.unpack(pert.sol(t))) - net.W(net.unpack(ref.sol(t)))).max()
                 for t in np.linspace(0, T, 401))
        t_p = solve_ivp(net.rhs_fn(Wstar), (0, 1e6), base + eps * xi, method="LSODA", rtol=1e-11, atol=1e-16,
                        events=ev).t_events[0][0]
        rows.append((eps, dW, abs(t_p - t_base)))
        print(f"  perturbation of norm {eps:.0e} on all {len(base)} entries: max |W_pert - W| = {dW:.2e} "
              f"(ratio {dW / eps:.2e}); escape time shift {abs(t_p - t_base):.2e} (escape at t = {t_base:.4g})")
    lin = abs(np.log(rows[1][1] / rows[0][1]) / np.log(100) - 1)
    check("perturbations off the manifold change W and the escape time linearly in their size (no instability "
          "off the manifold; the gain ~ t_esc/delta is the sensitivity of the escape time to the dormant amplitude)",
          lin < 0.05 and rows[1][1] / rows[1][0] < 1e3,
          f"ratio max|dW|/eps = {rows[1][1] / rows[1][0]:.2e}; log-log slope deviation {lin:.3f}")

# ---------------------------------------------------------------------------------------------------------------
if 15 in SECTIONS:
    section(15, "Bounded remainders of the asymptotics")
    sj, sr = 1.0, 0.6
    for (K, k, p0, deltas) in ((7, 3, [0.3, 0.55], (1e-2, 1e-3, 1e-4)), (8, 4, [0.35, 0.6, 0.5], (3e-2, 1e-2, 3e-3))):
        m = len(p0)
        red = Reduced(K, k, sj, sr, [np.eye(3)[1]] * m)
        w0 = [unit([0.0, p, np.sqrt(1 - p * p)]) for p in p0]
        I = kappa_integral(sr, p0, k - 2)
        remC, remt = [], []
        for de in deltas:
            C, t, _ = escape(red, de, w0)
            remC.append(C - de ** -(k - 2) * I)
            remt.append(t - de ** -(k - 2) * I / sj ** ((K - k) / K))
        print(f"  escape, K = {K}, k = {k}: C_esc - delta^-(k-2) I = {np.round(remC, 4)}; "
              f"t_esc - delta^-(k-2) I / s_j^((K-k)/K) = {np.round(remt, 4)}")
        check(f"escape, K = {K}, k = {k}: the clock remainder C_esc - delta^-(k-2) I stays bounded and converges",
              max(abs(r) for r in remC) < 10 and abs(remC[-1] - remC[-2]) < 0.05 * max(1, abs(remC[-1])))
        if k == 3:
            check("escape, k = 3: the physical-time remainder is also O(1)",
                  max(abs(r) for r in remt) < 10 and abs(remt[-1] - remt[-2]) < 0.05 * max(1, abs(remt[-1])))
    K, k, sr = 7, 3, 0.5
    red = Reduced(K, k, sj, sr, [np.eye(3)[1]] * 2)
    w0 = [unit([0.0, 0.25, np.sqrt(1 - 0.0625)]), unit([0.0, 0.4, np.sqrt(1 - 0.16)])]
    I = kappa_integral(sr, [0.25, 0.4], 1)
    rem = []
    for sig in (1e-2, 1e-3, 1e-4):
        _, dC, dt, _, _ = plateau(red, sig, sig, w0)
        rem.append(dC - (I - 1 / sj) / sig)
    print(f"  plateau, K = 7, k = 3, non-aligned: Delta C - sigma^-1 (I - 1/s_j) = {np.round(rem, 4)}")
    check("plateau: the clock remainder Delta C - sigma^-(k-2)(I - 1/((k-2) s_j)) stays bounded and converges",
          max(abs(r) for r in rem) < 10 and abs(rem[-1] - rem[-2]) < 0.05 * max(1, abs(rem[-1])))

# ---------------------------------------------------------------------------------------------------------------
if 16 in SECTIONS:
    section(16, "A dormant channel that sees several components: residual competition (three-core model)")
    sj = 1.0
    net = binary4(2, 2)                                # chi = 2: one dormant channel only
    comps = {"r2": ([0, 1, 0, 1], 0.8), "r3": ([1, 0, 1, 0], 0.6)}
    pij = [0, 0, 0, 0]
    Wstar = sj * net.basis(pij) + sum(sv * net.basis(pv) for pv, sv in comps.values())
    q = net.node_index(pij)
    rowc = {n: net.node_index(pv) for n, (pv, sv) in comps.items()}
    # dormant directions on the two bottom cores (dimension 4), orthogonal to q, with a larger overlap with r3
    wA = unit([0.0, 0.15, 0.9, 0.4])
    wB = unit([0.0, 0.2, 0.85, -0.3])
    print(f"  overlaps with r2: ({wA[rowc['r2'][1]]:.3f}, {wB[rowc['r2'][2]]:.3f}); "
          f"with r3: ({wA[rowc['r3'][1]]:.3f}, {wB[rowc['r3'][2]]:.3f})")
    S_nodes, S_edges = {0, 1, 2}, {(0, 1), (0, 2)}
    # Z of the main note for the deflated teacher seen by the dormant channel (k = K = 3, I = int Z^-1/2)
    terms = []
    for n, (pv, sv) in comps.items():
        pL, pR = wA[rowc[n][1]], wB[rowc[n][2]]
        terms.append((sv, (pL + pR) / 2, (pL - pR) / 2))
    gam2 = 1 - sum(al ** 2 + be ** 2 for _, al, be in terms)
    smax = max(sv for sv, _, _ in terms)

    def inv_sqrt_Z(tau):                              # Z^(-1/2), scaled to avoid overflow
        z = sum(al ** 2 * np.exp(2 * (sv - smax) * tau) + be ** 2 * np.exp(-2 * (sv + smax) * tau)
                for sv, al, be in terms) + gam2 * np.exp(-2 * smax * tau)
        return np.exp(-smax * tau) / np.sqrt(z)
    I_full = quad(inv_sqrt_Z, 0, np.inf, limit=500)[0]
    p3 = [wA[rowc["r3"][1]], wB[rowc["r3"][2]]]
    I_r3 = kappa_integral(0.6, [abs(pp) for pp in p3], 1) if p3[0] * p3[1] > 0 else np.nan
    U = {n: net.basis(pv) for n, (pv, sv) in comps.items()}
    rows = []
    for de in (1e-2, 1e-3, 1e-4):
        a_s = sj ** (1 / 3)
        cores = two_channel_cores(net, q, S_nodes, S_edges, a_s, a_s, de, {1: wA, 2: wB})
        evs = []
        for n, (pv, sv) in comps.items():
            f = (lambda U_, sv_: (lambda t, th: np.sum(net.W(net.unpack(th)) * U_) - 0.5 * sv_))(U[n], sv)
            f.direction, f.terminal = 1, True
            evs.append(f)
        sol = solve_ivp(net.rhs_fn(Wstar), (0, 1e8), net.pack(cores), method="LSODA", rtol=1e-11, atol=1e-16,
                        events=evs)
        hit = [(sol.t_events[i][0], n) for i, n in enumerate(comps) if len(sol.t_events[i])]
        t_first, who = min(hit)
        rows.append((de, t_first, who))
        print(f"  delta = {de:.0e}: first acquired {who} at t = {t_first:.5g}; delta t = {de * t_first:.5f} "
              f"(full residual I = {I_full:.5f}; r3 alone would give I = {I_r3:.5f})")
    check("the dormant channel learns the stronger r2 although it starts closer to r3", all(r[2] == "r2" for r in rows))
    check("its escape time is I/delta with I computed from the whole visible residual (Z of the main note)",
          abs(rows[-1][0] * rows[-1][1] / I_full - 1) < 2e-3, f"{rows[-1][0] * rows[-1][1]:.5f} vs {I_full:.5f}")

# ---------------------------------------------------------------------------------------------------------------
n_fail = sum(not ok for _, ok in RESULTS)
print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed.")
for name, ok in RESULTS:
    if not ok:
        print(f"  FAILED: {name}")
sys.exit(1 if n_fail else 0)
