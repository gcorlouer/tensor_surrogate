<!-- Critique of Section 4 (Balanced dynamics without alignment), first draft, by gpt-5.6-sol (reasoning effort xhigh) via Codex CLI, 2026-10-13. gpt-6-astra was requested but is not available to this ChatGPT account. The section was revised to address these points. -->

## Overall assessment

The core of the section is strong: the spectral equations and exact rank-one reduction are essentially correct, and both supplied check scripts pass. The main weaknesses begin when the exact rank-one analysis is extrapolated to random initialization, higher-rank plateaus, non-whitened dynamics, and arbitrary trees. Several such statements require additional hypotheses; a few tree/alignment claims are incorrect. The Hessian-spectrum proposition appears plausible and matches the numerical tests, but its advertised exhaustiveness is not proved.

## Issues, ordered by severity

1. **Location:** discussion after \(\eqref{eq:na-tree}\), especially “so that coarser cuts align last.”  
   **Severity:** error.  
   **Problem:** The conclusion does not follow from the product formula. For a cut \(A\mid A^c\), the two factor overlaps are
   \[
   \prod_{g\in A}p_{g,q},\qquad \prod_{g\notin A}p_{g,q},
   \]
   while their product—the overlap of the full rank-one tensor with the teacher component—is \(\prod_g p_{g,q}\), independent of the cut. Even if all \(p_g=p\), whether one cut “aligns later” depends on whether alignment means the subtree factor, the environment factor, their minimum, or the full tensor overlap. Under some definitions a leaf cut can be slower because its environment contains more groups.  
   **Concrete fix:** Define a cut-specific alignment observable explicitly. State only that the two side-overlaps are the corresponding products. Any ordering across cuts then needs a separate proposition under assumptions on group sizes and the chosen observable; otherwise delete “coarser cuts align last.”

2. **Location:** tensor-power paragraph after \(\eqref{eq:na-tree}\), comparison with \(m=2\).  
   **Severity:** error.  
   **Problem:** For a rank-one teacher and equal left/right overlaps \(p\),
   \[
   p'=\sigma^\star p(1-p^2).
   \]
   Starting from \(p(0)\asymp d^{-1/2}\), reaching fixed overlap takes
   \[
   \tau_{\rm align}=\frac{\log d}{2\sigma^\star}+O(1),
   \]
   not \(\log d/\sigma^\star\). The numerical check obtains the latter by evolving \(P=p^2\) with an equation that is not the equation induced by the displayed orientation flow. This also conflicts with \(\eqref{eq:na-escape-random}\), whose coefficient is \(1/2\).  
   **Concrete fix:** Replace the \(m=2\) comparison by \((2\sigma^\star)^{-1}\log d+O(1)\), and correct the numerical check accordingly.

3. **Location:** Proposition “Linear rates at the critical points,” especially “All other eigenvalues vanish.”  
   **Severity:** gap.  
   **Problem:** The proof sketch establishes some invariant \(1\times1\) and \(2\times2\) blocks, but not that they exhaust the full core-parameter Hessian. “We have checked numerically, for several \(S\)” cannot prove an exhaustive spectral proposition for arbitrary dimensions and bond redundancies. It also does not count the kernel or rigorously distinguish gauge, imbalance, and dormant-channel directions.  
   **Concrete fix:** Decompose \((\delta A,\delta B,\delta C)\) in completed teacher singular bases and bond frames. Write the Hessian as
   \[
   J^\top J+\langle M_S-M^\star,D^2M[\cdot,\cdot]\rangle,
   \]
   diagonalize all resulting blocks, and verify that the listed multiplicities plus the kernel dimension equal the full parameter dimension. Until then, weaken “Proposition” to a conjectural or numerically verified statement and remove “all.”

4. **Location:** paragraph preceding Proposition “Linear rates at the critical points.”  
   **Severity:** gap.  
   **Problem:** The partial-sum classification is correct only after an omitted argument. From stationarity one gets
   \[
   (M^\star-M)V=0,\qquad (M^\star-M)^\top U=0,
   \]
   so each active \((u_i,v_i,\sigma_i)\) must be a singular triplet of \(M^\star\). That implication, and the exclusion of additional teacher-null components, should be stated. The classification is also restricted to the exactly balanced stratum and a simple positive teacher spectrum; repeated singular values produce continuous families.  
   **Concrete fix:** Supply the singular-triplet argument and explicitly delimit the claim to balanced cores, positive active singular values, and simple teacher spectrum. Add the repeated-spectrum version in terms of invariant subspaces.

5. **Location:** “Scale invariance of the early phase.”  
   **Severity:** gap.  
   **Problem:** Homogeneity proves exact \(1/s(0)\) scaling for the *truncated* signal-only equation. It does not by itself prove
   \[
   t_1=\widehat I/s(0)+O(1)
   \]
   for the full equation. For \(k\ge2\), one must show that the rescaled signal-only trajectory has a suitable finite-time blow-up/escape, that it is nondegenerate, and that the degree-five self-interaction changes physical time by only \(O(1)\). Exceptional initial shapes may not escape at all. The single rank-two experiment is not a proof.  
   **Concrete fix:** Either prove a singular-perturbation/matched-asymptotic result under explicit nondegeneracy assumptions, or weaken the statement to the heuristic
   \[
   t_1=s(0)^{-1}\big(\widehat I+o(1)\big)
   \]
   for escaping initial shapes, with the \(O(1)\) remainder retained only for the proved rank-one case.

6. **Location:** “Deflated normal form,” especially the inference from \(\eqref{eq:na-rotation}\) to \(I(Y_r)/s_{r+1}\).  
   **Severity:** gap.  
   **Problem:** The rotation expansion itself is reasonable: with \(\delta=\sigma_j/\sigma_i\),
   \[
   (\Omega_U)_{ij}=-s_i\Theta_{ji}
   +O\!\left(s_i\delta^{2/3}\|\Theta\|\right),
   \]
   and similarly for \(V\). But no controlled error is given for replacing the coupled dormant dynamics by a fixed rank-one flow. The reduction also requires more than amplitude separation: active-mode alignment errors must be small, \(Y_r\) must vary slowly, the deflated leading singular value must be simple, and other dormant channels must be negligible or spectrally separated. Dormant channels initialized at comparable scales can interact.  
   **Concrete fix:** State these hypotheses and give an error estimate over the plateau interval. Otherwise label \(I(Y_r)/s_{r+1}\) explicitly as an empirical normal-form approximation. The “within \(0.1\%\)” claim should include the full example parameters or be moved to a numerical section.

7. **Location:** random-initialization paragraph and \(\eqref{eq:na-escape-random}\).  
   **Severity:** overclaim.  
   **Problem:** The correct probabilistic conclusion is roughly
   \[
   \sigma^\star_1 I-\tfrac12\log(1/\alpha_1^2)=O_{\mathbb P}(1),
   \]
   under fixed or sufficiently low teacher rank, bounded aspect ratio, a fixed spectral gap, and a non-negligible kernel component. It is not a deterministic uniform estimate: \(\alpha_1\) has substantial lower-tail probability, and atypically small \(\alpha_1/\alpha_2\) produces a strict-saddle detour. If the teacher rank is comparable to dimension, \(\gamma^2\approx1\) is false and the kernel-based argument fails. The quoted range \([-0.7,0.7]\) comes from one seeded sample at each of four dimensions with one spectrum; it is not a meaningful numerical range for random initialization generally.  
   **Concrete fix:** Formulate the result in probability, state the rank/aspect-ratio/gap regime, and report distributions or quantiles over many trials. Identify the spectrum, seed, rank, and number of samples if retaining the numerical statement.

8. **Location:** “Non-whitened inputs.”  
   **Severity:** gap / overclaim.  
   **Problem:** The leading orientation equation is indeed driven by
   \[
   H=\Sigma_{12}M^\star\Sigma_{34},
   \]
   with an \(O(\sigma)\) correction in rescaled time. Selection of its leading singular pair nevertheless requires a simple leading singular value and nonzero positive-dilation overlap. The \(I(H)/s(0)+O(1)\) conclusion needs the same escape and boundary-layer argument as above. Moreover, “the moment matrix sets the order and the times of acquisition” is not established for several interacting channels. “Relaxes towards the data-weighted optimum” also requires conditions such as nonsingular Gram matrices or a careful treatment of nonuniqueness.  
   **Concrete fix:** Restrict the claim to an infinitesimal rank-one channel with a simple leading singular pair and nonzero overlap. For higher rank, describe \(H\) as the initial driving matrix, not as a proved acquisition-order rule. Define the later rank-one weighted optimum explicitly through the SVD of \(\Sigma_{12}^{1/2}M^\star\Sigma_{34}^{1/2}\).

9. **Location:** “Near the aligned manifold,” \(\eqref{eq:na-near-aligned}\).  
   **Severity:** clarity / gap.  
   **Problem:** Writing \(U=U^\star e^X\), \(V=V^\star e^Y\) only treats rotations *within* the teacher singular subspaces. When \(r^\star<d_{12}\) or \(r^\star<d_{34}\), it is not a general neighborhood of the aligned manifold and omits out-of-span errors. The “second-order” remainder is also not uniform when \(|\sigma_i-\sigma_j|\) is small. Finally, linear growth of \(\xi^+\) suggests a swap but does not prove the nonlinear statement “the modes swap.”  
   **Concrete fix:** Say explicitly that this is the fixed-subspace, simple-student-spectrum linearization with \(|\sigma_i-\sigma_j|\ge c>0\). Add the out-of-span linear equations separately. Replace “the modes swap” by “the current pairing is linearly unstable; a nonlinear swap is expected and observed.”

10. **Location:** “Near the optimum.”  
    **Severity:** gap.  
    **Problem:** The linear operator \(-\mathcal K_{M^\star}\) is correct, including for rank-deficient \(M^\star\), but the justification is missing because the fractional powers are not Lipschitz at zero singular values. One cannot silently invoke an ordinary second-order Taylor expansion. The valid argument is
    \[
    F(M^\star+\Delta)
      =-\mathcal K_{M^\star+\Delta}[\Delta]
      =-\mathcal K_{M^\star}[\Delta]+o(\|\Delta\|),
    \]
    using continuity of \(\mathcal K_M\). The remainder generally has a fractional-power order rather than \(O(\|\Delta\|^2)\). Zero eigenvalues in the null–null block do not yield linear convergence and require separate nonlinear analysis.  
    **Concrete fix:** Include the continuity argument, state the norm, avoid claiming a quadratic remainder, and separate transverse exponential modes from the algebraic null–null modes. Clarify whether the student has rank \(r^\star\) or extra active channels.

11. **Location:** “Alignment and amplification,” including \(\eqref{eq:na-align-amp}\).  
    **Severity:** gap / overclaim.  
    **Problem:** The formula is plausible for fixed \(\eta,\varepsilon\), nonzero \(\alpha_1\), and \(s_{\rm align}\to0\), but these dependencies are not stated. The displayed scale condition is a sufficient asymptotic condition because
    \[
    s_{\rm align}\sim s(0)\sqrt{Z(\tau_\varepsilon)};
    \]
    failure of the strong inequality does not by itself prove the opposite ordering. The exact ordering should be defined by comparing \(t(\tau_\varepsilon)\) and \(t_\eta\).  
    **Concrete fix:** State the limit and parameter dependence precisely, call the scale inequality sufficient rather than an iff criterion, and use the exact quadrature to define the crossover regime.

12. **Location:** strict-saddle detour paragraph after \(\eqref{eq:na-align-amp}\).  
    **Severity:** clarity / overclaim.  
    **Problem:** For fixed initial direction, \(\tau_\times\) is fixed while \(\tau_{\rm esc}\sim\log(1/s(0))\). Hence for sufficiently small \(s(0)\), the channel aligns before escape and never spends time near the weaker-mode saddle. The expression
    \[
    (\tau_\times-\tau_{\rm esc})/\sigma_2^{\star1/3}
    \]
    can become negative. Also, local instability does not by itself prove that a higher-rank trajectory “replaces” one mode by another.  
    **Concrete fix:** Use the positive part
    \[
    (\tau_\times-\tau_{\rm esc})_+/\sigma_2^{\star1/3},
    \]
    and state the additional requirement that the trajectory has already approached the weaker-mode saddle. Restrict the replacement claim to rank one or present it as a nonlinear scenario rather than a theorem.

13. **Location:** Corollary “Escape from the origin,” \(\eqref{eq:na-escape}\)–\(\eqref{eq:na-escape-bounds}\).  
    **Severity:** gap / clarity.  
    **Problem:** The main asymptotic and equality case are correct, but the proof only sketches the \(O(1)\) remainder. Its constant depends on fixed \(\eta\), the direction, spectrum, and especially \(\alpha_1\); it is not uniform as \(\alpha_1\to0\) or \(\eta\uparrow1\). The middle bound is undefined at \(\gamma=0\), although its continuous limit exists. Subsequent prose sometimes calls \(1/(\sigma^\star_1s(0))\) “the escape time” rather than its leading asymptotic.  
    **Concrete fix:** Give a short late-time rescaling proof of the bounded remainder, state its nonuniformity, interpret the \(\gamma=0\) expression by continuity, and consistently say “leading escape time.” The left equality characterization is otherwise correct.

14. **Location:** Proposition “Exact rank-one dynamics” and its following interpretation.  
    **Severity:** clarity.  
    **Problem:** The exact formulas are correct, but “selects the leading teacher mode that it overlaps” should mean the largest \(q\) with \(\alpha_q\neq0\), not merely nonzero left/right overlaps. A purely antisymmetric overlap \(\beta_q\) lies in the negative eigenspace of the dilation and decays. If all \(\alpha_q=0\), escape need not occur.  
    **Concrete fix:** State the selection criterion in terms of the positive-dilation coefficients \(\alpha_q\), and discuss the no-positive-overlap case.

15. **Location:** “The exact solution extends to any TTN” and \(\eqref{eq:na-tree}\).  
    **Severity:** overclaim / gap.  
    **Problem:** What extends is the rank-one triangular reduction, not an exact closed-form solution: for \(m\ge3\), the orientation equation remains unsolved. Compatible exact balance at every edge, nonzero common scale, sign conventions, and preservation of the global rank-one stratum should also be stated carefully. The escape-prefactor integral may diverge if the orientation flow does not reach a component with positive limiting \(\rho\).  
    **Concrete fix:** Replace the opening sentence by “The rank-one reduction extends to any TTN.” State the balance and convergence hypotheses and condition the escape asymptotic on finiteness of the integral.

16. **Location:** tensor-power alignment asymptotic for \(m\ge3\).  
    **Severity:** overclaim.  
    **Problem:** The coefficient
    \[
    \frac{p(0)^{-(m-2)}}{(m-2)\sigma^\star_q}
    \]
    follows for a single winning component with equal positive overlaps across groups. A random initialization does not have equal overlaps; signs and unequal magnitudes affect the constant and the basin. With several odeco components, the \(-\rho p_{g,q}\) coupling can suppress losing components. The robust conclusion is the power \(d^{(m-2)/2}\), not the displayed universal constant or guaranteed alignment with an arbitrary \(q\).  
    **Concrete fix:** Present the formula first for a rank-one teacher and symmetric initial condition. For random initialization, state only a conditional/order-in-probability scaling for the winning basin unless a separate basin analysis is supplied.

17. **Location:** Proposition “Spectral equations.”  
    **Severity:** clarity / minor gap.  
    **Problem:** The formulas and invariant-subspace proof are correct, as confirmed by the checks, but the proposition should specify that they hold on intervals of constant rank with simple positive singular values. At crossings, \(\Omega_U,\Omega_V\) are undefined. In the general-input sentence, “replace \(\Theta-S^3\)” is too compressed because that quantity is not visibly present in every displayed equation. The invariant-subspace statement uses the initial rank \(k=\operatorname{rank}M(0)\), not an unspecified time-dependent \(k\).  
    **Concrete fix:** State the interval-of-validity assumption and write the general-input formulas explicitly with \(G=U^\top R_\Delta V\). Separate the invariant-subspace result into a second assertion, since it does not require a simple spectrum.

18. **Location:** Summary Table \(\ref{tab:na-times}\).  
    **Severity:** error / clarity.  
    **Problem:** Several rows overstate their scope:

    - “any initial direction” contradicts the required \(\alpha_1\neq0\);
    - “exact as \(s(0)\to0\)” should be “leading asymptotic”;
    - \(\tau_\varepsilon\) is generally defined implicitly, not “explicit”;
    - the random-start row needs the low-rank/gapped probabilistic regime;
    - the plateau row needs active/dormant separation and slow \(Y_r\);
    - the swap row lists an instantaneous rate under a heading “Time”;
    - convergence omits the algebraic null–null block;
    - the deeper-tree row needs equal-overlap/winning-component hypotheses.

    **Concrete fix:** Revise each setting column to include these hypotheses, replace the swap rate by its inverse times a logarithmic perturbation factor if a time is wanted, and distinguish exact identities, proved leading asymptotics, local linearizations, and empirical approximations.

19. **Location:** section-wide claims of verification and absence of closed forms.  
    **Severity:** overclaim.  
    **Problem:** Both supplied scripts pass—27/27 numerical and 10/10 symbolic checks—but they verify selected finite-dimensional examples and algebraic identities, not general asymptotics or exhaustive Hessian spectra. Conversely, the fractional powers show that a known Riccati solution does not apply; they do not prove that no closed form exists.  
    **Concrete fix:** Say “does not inherit the known Riccati solution” and describe the scripts as consistency checks. Do not use numerical agreement as part of a proposition’s proof.

20. **Location:** Oja-flow, tensor-power, and saddle discussions.  
    **Severity:** missing reference.  
    **Problem:** The section names Oja’s flow and tensor power flow without citing the established literature. The saddle-to-saddle interpretation also deserves a local citation to the deep-linear results already cited in the introduction.  
    **Concrete fix:** Add references to Oja’s original principal-component flow, standard continuous/power-method analyses, and tensor power methods for orthogonally decomposable tensors—for example Oja (1982), Kofidis–Regalia or Kolda–Mayo, and Anandkumar et al. The critical-point/plateau discussion should locally cite the relevant deep-linear saddle-to-saddle work already present in the bibliography.

No files were modified.