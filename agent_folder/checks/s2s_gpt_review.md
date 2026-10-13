# GPT review of saddle_to_saddle.tex (gpt-5.6-sol, effort max) and response

1. **Critical — §7.1, Rule 7.1, R7 and the abstract** ([lines 457–481](/Users/guime/projects/tensor_surrogate/agent_folder/saddle_to_saddle.tex:457)). The multi-component “budget rule” is neither a leading-order consequence of the two-channel reduction nor true under A1–A7. A generic dormant channel sees the entire projected residual, not one rank-one component. For example, in the three-core model let
   \[
   M^\star=u_1v_1^\top+0.8u_2v_2^\top+0.6u_3v_3^\top
   \]
   and suppose mode \(1\) is learned. A generic channel nominally assigned to mode \(3\) evolves by Oja flow for \(0.8u_2v_2^\top+0.6u_3v_3^\top\); positive overlap with mode \(3\) does not prevent it from aligning to the stronger mode \(2\). Its clock budget is obtained from the full \(Z\) of TS (4.7), not \(I^{(3)}_\kappa\) computed from \(s_3U_3\). This competing signal survives as \(b\to0\), so it is not a “higher-order interaction between dormant channels.” The note’s own rerouting experiment gives a second contradiction: a channel initialized on a \(k=4\) branch, whose proposed initial budget scales as \(\delta^{-2}\), subsequently escapes as \(\delta^{-1}\). It plainly did not exhaust the fixed initial budget in (7.1). **Fix:** restrict the rule to residual-isolated invariant channels satisfying the analogue of the \(W'\) condition in Proposition 4.2, or replace \(I_\kappa^{(r)}\) by the orientation flow for the full projected residual. For rerouting, budgets must be created or recomputed at each saddle. R7 and the abstract should call the general sequence a conjectural heuristic, not a result.

2. **Major — Proposition 7.3 and Appendix A.5** ([lines 487–500](/Users/guime/projects/tensor_surrogate/agent_folder/saddle_to_saddle.tex:487)). The proof of the relative-order bound assumes facts not implied by the proposition. It obtains
   \[
   \Delta W=\epsilon^{k'}\sum_\gamma A_\gamma U_\gamma
   \]
   and then uses
   \[
   -\sum_\gamma A_\gamma s_\gamma\le -A_{\pi_r}s_r.
   \]
   This requires every relevant \(A_\gamma\ge0\). Knowing merely that each learned \(U_i\) has one channel with positive entries does not make the conditional environments selected by its boundary indices nonnegative: other paths can have negative entries and cancellations. Likewise, “an index that no channel uses” does not imply that all existing core entries involving that index vanish. Consequently, lower-order mixed contractions and signed completions are not excluded. **Fix:** assume a canonical sparse direct-sum realization whose entire support consists of isolated nonnegative learned channels, and require the fresh index to be absent from both endpoint-core supports. Alternatively, replace the positivity argument by an explicit hypothesis
   \(\langle W^\star-W(\theta_J),\Delta W\rangle>0\). Without such a condition, downgrade the proposition to a conjecture.

3. **Major — Proposition 3.1(ii)–(iii), Table 1** ([lines 187–198](/Users/guime/projects/tensor_surrogate/agent_folder/saddle_to_saddle.tex:187)). The deduction from SO omits two hypotheses. First, creation of a second mode requires an unused index on both bonds; with bond dimensions one and \(s_r<s_j\), the learned top rank-one approximation is a constrained minimum, not an order-three saddle. A7 may be intended to exclude this, but Proposition 3.1 must invoke it explicitly. Second, if some \(s_r=s_j\), then \(\mu_1=s_j\), violating SO Theorem 1’s nondegeneracy assumption; TS also formulates Proposition 4.4 under a simple teacher spectrum. The displayed Hessian eigenvalues remain correct, but the proof of the order classification does not cover the zero-rate case. **Fix:** assume both bond dimensions are at least two and \(s_r\ne s_j\) for every unlearned mode, or separately prove the equal-singular-value case. Part (iii) should also invoke the exact aligned-mode decoupling of TS Proposition 3.10 before applying TS Corollary 4.3 to a dormant channel.

4. **Major — A5–A7 and “locality before weight,” R7(a), §7.2** ([lines 99–106](/Users/guime/projects/tensor_surrogate/agent_folder/saddle_to_saddle.tex:99)). The assumptions do not justify the claimed ordering. SO shows that excluding a negative Hessian is insufficient: its Example 1 has \(\mu_1<s_j\) but \(\mu_2>s_j\), giving order \(4<k^\star=5\). The required condition is
   \[
   \mu_\ell<s_j\qquad\text{for every }\ell<k^\star/2,
   \]
   together with the nondegeneracy and edge-capacity conditions of SO. Moreover, “initialization scale \(\sigma\)” must mean \(b_r(0)=c_r\sigma\) with fixed \(0<c_r<\infty\); otherwise a more local channel initialized at \(\sigma^M\) can lose to a less local one. Finally, \(\chi_e\ge2\) is sufficient for one channel, not for several overlapping dormant channels. **Fix:** state these hypotheses explicitly and restrict the conclusion to isolated channels at a fixed one-component saddle. No corresponding theorem currently exists after several components have been learned.

5. **Major — Proposition 6.2(i) and Appendix A.4** ([lines 415–429](/Users/guime/projects/tensor_surrogate/agent_folder/saddle_to_saddle.tex:415)). The centre-manifold argument is not a proof of the exact transition statement. For \(k\ge3\), the \(b\)-direction is nonhyperbolic, and the generic centre-manifold theorem neither gives uniqueness nor follows from “the flow on it moves away.” Tangency of the nullcline to a centre manifold also does not by itself prove convergence of passage times on fixed \(\omega_r\)-sections. For \(k=2\), this is instead an ordinary unstable-manifold problem. Moreover, the planar system is exact only when \(p_v=1\); after a non-aligned escape it is merely asymptotically planar. **Fix:** analyze the singular invariant-graph equation
   \[
   h'(b)=\frac{h^{k-1}(s_j-a^{K-k}h^k)}
                 {b^{k-1}(s_r-a^{K-k}b^k)},\qquad a^2=h^2+b^2,
   \]
   prove uniqueness of the orbit approaching \((x_s,0)\) backward, and establish convergence on a transverse section. Treat \(k=2\) separately and state explicitly whether the proposition assumes exact alignment or takes \(\delta\to0\).

6. **Major — Corollary 5.4** ([line 373](/Users/guime/projects/tensor_surrogate/agent_folder/saddle_to_saddle.tex:373)). Its “\(j\) is learned first” hypothesis is undefined for \(k=2\):
   \[
   b_0^{-(k-2)}I_{k-2}>\frac{x_0^{-(k-2)}}{(k-2)s_j}
   \]
   contains both \(I_0\) and division by zero. **Fix:** split the hypothesis:
   \[
   b_0^{-(k-2)}I_{k-2}>
   \frac{x_0^{-(k-2)}}{(k-2)s_j}\quad(k\ge3),
   \]
   and
   \[
   \frac{\log(1/b_0)+\Lambda}{s_r}>
   \frac{\log(1/x_0)}{s_j}+O(1)\quad(k=2).
   \]
   For fixed \(b_0/x_0\), the leading \(k=2\) condition is \(s_r<s_j\), with constants deciding the equal-weight boundary case.

7. **Major — numerical claims in §8 and Table 2** ([line 511](/Users/guime/projects/tensor_surrogate/agent_folder/saddle_to_saddle.tex:511); [script](/Users/guime/projects/tensor_surrogate/agent_folder/checks/s2s_checks.py:508)). Several quoted accuracies are not what the scripts assert:

   - The budget test uses `< 0.02`, i.e. \(2\%\), while the note claims \(0.02\%\), a factor \(100\) tighter.
   - The escape-clock assertion permits \(0.3\%\), versus \(0.1\%\) in Table 2.
   - The plateau-clock assertion permits \(0.5\%\), versus \(0.2\%\); physical-time tests permit \(3\%\).
   - The equal-order random result and the \(\chi=3\) capacity stalls are printed but never passed to `check`, so “82 checks, all passing” does not validate them.
   - The “random initialization” used for the capacity claim is biased toward \(j\) by explicitly adding \(\sigma\) to its channel entries.
   - Neither script tests the allowed \(W'\), transverse perturbations away from the invariant manifold, nor the claimed \(O(1)\) remainders.

   **Fix:** make every Table 2 number an asserted diagnostic with the advertised tolerance; report raw residuals and fitted rates; test the \(W'\) condition and off-manifold perturbations; and describe the five capacity runs as a biased, fixed-seed pilot experiment. Five runs do not support the general claim that one spare direction removes stalling.

8. **Major — R6–R7 use of Proposition 6.3** ([lines 433–449](/Users/guime/projects/tensor_surrogate/agent_folder/saddle_to_saddle.tex:433)). The formula itself is correct:
   \[
   R=\left(\frac{a_*}{s_j^{1/K}}\right)^{n_{\rm out}}
      \left(\frac{x_*}{s_j^{1/K}}\right)^{n_S}
    =f^{\,n_{\rm out}-n_S(K-k)/k}.
   \]
   But it describes one two-channel acquisition from the one-component balanced saddle, with the later channel still attached to \(j\). It is not an update rule after arbitrary histories: with several learned channels, balance is matrix-valued and involves sums of several channel Gram matrices, so there need not be one scalar \(f\) or one \(x_*\). Rerouting also changes which entries define the clock. **Fix:** restrict R6 and Proposition 6.3 to the first acquisition after \(j\), or derive a genuinely multi-channel Gram-matrix rebalancing law.

9. **Major — A3–A4 and §7.2(c), “the first phase”** ([lines 479–481](/Users/guime/projects/tensor_surrogate/agent_folder/saddle_to_saddle.tex:479)). Balance does not generally imply that named coordinate channels carry equal per-core amplitudes. It gives equality of bond Gram matrices; channelwise equalities follow only after a simultaneous singular-frame decomposition, or on the sparse two-channel ansatz. Similarly, a generic small full-bond-rank initialization is not governed by one tensor power flow. TS explicitly notes that several dormant modes interact; its exact power/Oja reduction is rank one. **Fix:** qualify all channelwise balance statements by the diagonal two-channel ansatz and restrict the tensor-power claim to bond rank one. For generic random initialization, this should be an open problem, not a stated consequence.

10. **Minor — Proposition 6.1 and Appendix A.4** ([lines 401–407](/Users/guime/projects/tensor_surrogate/agent_folder/saddle_to_saddle.tex:401)). The equilibrium formulas are correct, but the proof skips a necessary boundary argument. Proposition 5.2 gives convergence as \(\tau\to\infty\); it does not itself prove that \(\tau(t)\to\infty\). Before invoking it, one must exclude \(x\to0\), \(b\to0\), and finite total clock. This is repairable using \(P(0)>0\), which gives \(\rho(\tau)\ge\rho(0)>0\), followed by lower bounds near the boundary. Also, if Proposition 4.2’s allowed \(W'\neq0\) remains in scope, the limit satisfies
   \(W=s_jU_j+s_rU_r\), not \(W=W^\star\). **Fix:** assume \(W'=0\) explicitly and add the clock-divergence/boundary lemma.

11. **Minor — Proposition 4.2 and its “exact” scope** ([lines 208–277](/Users/guime/projects/tensor_surrogate/agent_folder/saddle_to_saddle.tex:208)). Two endpoint cases are mishandled. The definition permits \(b_v=0\), while the reduced equations divide by \(b_v\) and the polar direction \(w_v\) is then undefined. The proof itself only establishes the parametrized flow while amplitudes are positive. Also, when \(S=T\), no \(a_v\) exists and balance does not imply \(a^2=x^2+b^2\); \(a\) is merely an auxiliary variable whose exponent \(K-k\) is zero. **Fix:** formulate invariance first in the linear variables \(V_v=b_vw_v\), restrict the polar equations to \(b_v>0\), and treat \(k=K\) separately or explicitly declare \(a=\sqrt{x^2+b^2}\) to be bookkeeping only.

12. **Minor — Proposition 5.2(ii)–(iv) and Proposition 6.1** ([lines 323–345](/Users/guime/projects/tensor_surrogate/agent_folder/saddle_to_saddle.tex:323)). The sign convention is not formally consistent. If \(p_1(0)=p_2(0)=-1/2\), then \(P(0)>0\), but the actual overlaps tend to \(-1\), and
   \(\log(2/(p_1+p_2))\) is undefined. Pairwise sign flips leave the represented tensor unchanged, but the proposition must explicitly adopt that gauge before redefining \(p_v\). **Fix:** either assume \(p_v(0)>0\) after a stated gauge choice or write
   \[
   \Lambda=\log\frac{2}{|p_1(0)|+|p_2(0)|},
   \]
   and state convergence to a sign pattern with product \(+1\).

13. **Minor — Corollaries 5.3–5.4 and Appendix A.3** ([lines 580–590](/Users/guime/projects/tensor_surrogate/agent_folder/saddle_to_saddle.tex:580)). The claimed error orders are plausible and have the right scaling, but are asserted rather than proved uniformly. Once one proves
   \(a^{K-k}=s_j^{(K-k)/K}(1+O(b^2))\), the physical-time correction is
   \[
   O\!\left(\int b^2\,dC\right)
   =\begin{cases}
   O(1),&k=3,\\
   O(\log(1/\delta)),&k=4,\\
   O(\delta^{4-k}),&k\ge5,
   \end{cases}
   \]
   versus \(C\asymp\delta^{2-k}\), giving the stated relative
   \(O(\delta)\), \(O(\delta^2\log(1/\delta))\), and \(O(\delta^2)\).
   What is missing is a uniform estimate proving the slaving
   \(\omega_j=s_j+O(b^2)\), including the initial relaxation and non-aligned search. The scripts test only scaled convergence, which is compatible with errors much larger than \(O(1)\). **Fix:** add this lemma and test bounded quantities such as
   \(C-\delta^{-(k-2)}I_{k-2}\), not only \(\delta^{k-2}C\).

14. **Minor — description of TS in the abstract, §1 and §3** ([lines 49–61](/Users/guime/projects/tensor_surrogate/agent_folder/saddle_to_saddle.tex:49)). TS does not “solve rank-one channels on any tree without alignment.” TS §4.7 explicitly says that for \(m\ge3\) it does not know the orientation flow in closed form; it derives the triangular reduction and treats equal overlaps for a rank-one teacher. The present note’s unequal-overlap solution is therefore more genuinely new than the attribution suggests. **Fix:** replace “solves” by “reduces to an autonomous orientation flow and quadrature,” distinguishing the exact \(m=2\) solution from the \(m\ge3\) asymptotics.

15. **Minor — equation (7.2), the crossover** ([lines 471–475](/Users/guime/projects/tensor_surrogate/agent_folder/saddle_to_saddle.tex:471)). The equation need not have a unique solution. Writing \(q=k_2-2\), it is
   \[
   \sigma_c^q\log(1/\sigma_c)=B/A.
   \]
   The left side has maximum \(1/(eq)\) at \(\sigma=e^{-1/q}\), so there may be zero, one, or two roots. Only the smaller root is relevant to the small-\(\sigma\) asymptotics. **Fix:** state the existence condition \(B/A\le1/(eq)\), define the smaller root, and warn that the larger root is outside the reliable asymptotic regime.

16. **Major — §9 research questions** ([lines 541–555](/Users/guime/projects/tensor_surrogate/agent_folder/saddle_to_saddle.tex:541)). Q5 and Q6 are not presently well posed, and the central missing question is absent. For generic rotated product atoms,
   \[
   \nabla_{G_v}\mathcal L(\theta_S)\supset
   -s_r\,t_v\prod_{u\ne v}\langle q_u,t_u\rangle,
   \]
   which is generically nonzero; the purported one-component point is therefore not a saddle, and there is no discrete set of cores on which two atoms “differ.” For tied polynomial networks, independent edge gauges, balance invariants, and independent Steiner perturbations are destroyed by weight tying, while repeated inputs change the loss geometry; equation (7.2) is not yet a quantitative prediction there. **Fix:** formulate Q5 in terms of vanishing tangent projections/contact order, and Q6 as asking whether an analogous mechanism survives tying. Add as the highest-priority question the transverse stability and persistence of \(\mathcal M_{j,r}\) under generic \(O(\sigma)\) off-manifold perturbations, \(O(\sigma^2)\) imbalance, and multiple full-rank dormant channels.

17. **Minor — notation throughout.** \(S\) denotes a learned-mode set in TS, the subscript of \(\theta_S\), and later a Steiner subtree; \(C\) is both the central core of the three-core model and the clock; \(a\) is a real shared-core amplitude except when \(k=K\), when it is artificial; and \(\omega_r\) is the coefficient of \(D\), whereas the actually acquired target coefficient is \(P\omega_r\). The \(d\) in Remark 5.3 should be the physical group dimension \(D_v\) (or \(d_g\)), not necessarily the site dimension. **Fix:** rename the Steiner subtree \(\mathcal S_r\), the clock \(\mathcal C_r\), and consistently distinguish \(\omega_r\) from \(\widehat\omega_r=\langle U_r,W\rangle=P\omega_r\).

Overall, the exact isolated two-component calculation is largely sound: the mixed-core invariance argument, reduced flow, clock identity, \(I_\kappa\), \(\Lambda\), hypergeometric expression, rebalancing formula, and conditional clock factor \(R\) check out. The headline multi-component saddle-to-saddle theory does not follow from it and is contradicted by generic residual competition and by the note’s own rerouting example. The note requires major revision: restrict the proved results to the two-channel setting, repair Propositions 3.1, 6.2 and 7.3, and recast the general budget/sequence claims as conjectures with substantially more targeted evidence.
---

## Response to the review (changes made in `saddle_to_saddle.tex`)

Reviewer: gpt-5.6-sol at reasoning effort "max" (Codex CLI, read-only). The requested gpt-6-astra is not available to this account. One review round.

1. **Budget rule (critical).** Accepted. Rule 7.1 is now a conjecture, stated at a one-component saddle for *residual-isolated* dormant channels (every other unlearned component differs from U_j on at least two cores outside the subtree, the analogue of the W' condition), under SO's conditions (A5(ii)) and with b_r(0) = c_r sigma. New paragraph "Channels that see several components": such a channel follows the orientation flow of the whole visible residual, acquires the strongest visible component, and its budget is computed along that flow. New check (section 16): in the three-core model, a channel with overlap 0.9 with the 0.6 component learns the 0.8 one, with escape time equal to the full-Z budget to 1e-4. Rerouting: budgets are now said to be recomputed at each saddle along the shortest available path. Abstract and R7 rewritten accordingly.
2. **Proposition 7.3.** Accepted. Hypotheses are now: nonnegative entries, a learned channel that is a path of nonzero entries, and a fresh index with zero entries in all cores. The proof uses them.
3. **Proposition 3.1.** Accepted. Added bond dimensions >= 2 and s_r != s_j. The proof of (iii) now invokes the decoupling of TS Prop. 3.10.
4. **Assumptions for locality before weight.** Accepted. A5 now lists mu_l < s_j for every l < k*/2 together with SO's nondegeneracy and capacity conditions. A6 has b_r(0) = c_r sigma with fixed c_r. A7 asks for capacity for all dormant channels. The conclusion is restricted to a one-component saddle.
5. **Proposition 6.2(i).** Accepted. There is a new proof via the invariant-graph equation h'(b) = Phi(h,b)/b^{k-1}. Uniqueness comes from the divergence of int db/b^{k-1}; convergence comes from contraction on sections plus continuous dependence. It is stated for aligned channels, with a remark for non-aligned ones.
6. **Corollary 5.4, k = 2.** Accepted. The hypothesis is split by k.
7. **Numerical claims.** Accepted. Asserted tolerances now match the accuracies quoted in Table 2. Results that are printed but not asserted are identified as such. The random initialization is described as biased and as a five-seed pilot. New checks: the W' condition, with an invariant case and a breaking case (section 13); off-manifold perturbations, where the response is linear (section 14); bounded remainders, with the predicted log growth of the physical-time remainder for k = 4 (section 15). 92 asserted checks, all passing.
8. **Proposition 6.3 scope.** Accepted. It is restricted to the first acquisition after j, with the later channel attached to j and residual-isolated. A sentence explains why several acquisitions need a Gram-matrix law.
9. **Channelwise balance, first phase.** Accepted. A3 says balance equates Gram matrices and fixes channel amplitudes only on the two-channel ansatz. The first-phase tensor-power claim is restricted to bond rank one; full-rank initialization is listed as an open question.
10. **Proposition 6.1.** Accepted. Added the boundary and clock-divergence argument (rho >= rho(0) > 0; b and x bounded below; tau -> infinity). The limit is W = s_j U_j + s_r U_r (= W* when W' = 0).
11. **b_v = 0; k = K.** Accepted. The manifold is defined in linear variables V_v. The polar equations are stated for b_v > 0, and a is bookkeeping only when k = K.
12. **Sign convention.** Accepted. The sign gauge is fixed explicitly, and Lambda = log(2/(|p1|+|p2|)).
13. **Error orders.** Partly accepted. Added Lemma A.1 (slaving: omega_j = s_j + O(b^k) after an O(1) relaxation, and a^{K-k} = s_j^{(K-k)/K}(1 + O(b^2))). This makes the error orders follow from the integral of b^2 dC. The text still calls the orders "formal", and numerical tests of the bounded remainders were added.
14. **Description of TS.** Accepted. TS now "reduces a rank-one channel of any tree to an orientation flow and quadratures", solves m = 2, and treats equal overlaps for m >= 3. The closed form for unequal overlaps is presented as new.
15. **Crossover equation.** Accepted. Added the analysis of roots (none, or two roots with the smaller one relevant).
16. **Research questions.** Accepted. New Q1 on persistence and transverse stability of the reduction. Q6 (beyond basis targets) is reformulated: there the one-component point is generically not critical, and the replacement for the Steiner subtree is posed in terms of first-order coupling of new channels. Q7 asks whether the mechanism survives weight tying.
17. **Notation.** Accepted. The clock is now \mathcal C (the central core of the three-core model is C). The one-component saddle is now theta_j (theta_S in SO). We define hat omega_r = <U_r, W> = P omega_r. The physical-leg dimension D replaces d in Remark 5.3.
