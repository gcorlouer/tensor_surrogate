(* ::Package:: *)
(* Exact checks of Section 3 ("Training dynamics of TTNs") of ttns_surrogate.tex.

   Run:  wolframscript -file sec3_gradient_flow.wl

   Labels such as (d2-white-flow) refer to the equation labels in ttns_surrogate.tex.
   Notation follows the paper: s12 = Sigma_12 = E[z z^T], s34 = Sigma_34 = E[w w^T],
   s1234 = Sigma_[4] = E[(z (x) w)(z (x) w)^T], and vec flattens row by row (lexicographic order).
   sec3_gradient_flow.py runs the same checks with SymPy. *)

results = {};
check[name_, ok_] := (AppendTo[results, {name, TrueQ[ok]}];
   Print[If[TrueQ[ok], "  [PASS] ", "  [FAIL] "], name]);
zeroQ[e_] := AllTrue[Flatten[{Expand[e]}], # === 0 &];
section[s_] := Print["\n== ", s];
vecRow[m_] := Flatten[m];                            (* row-by-row vec *)
unvecRow[v_, r_, c_] := Partition[v, c];
frob[p_, q_] := Total[p q, 2];
outer[u_, v_] := Outer[Times, u, v];
blockDiag[ms_List] := Module[{n = Total[Length /@ ms], out, k = 0},
   out = ConstantArray[0, {n, n}];
   Do[out[[k + 1 ;; k + Length[m], k + 1 ;; k + Length[m]]] = m; k += Length[m], {m, ms}];
   out];

SeedRandom[1];
(* Dimensions are distinct where possible, so that transposition errors show up. *)
dims = {2, 3, 2, 2}; {d1, d2, d3, d4} = dims; bondL = 2; bondR = 3;
d12 = d1 d2; d34 = d3 d4; nx = Total[dims];

(* ------------------------------------------------------------------------ *)
(* Section 3.1: model                                                        *)
(* ------------------------------------------------------------------------ *)
X = Table[x[k, i], {k, 4}, {i, dims[[k]]}];
xs = Flatten[X];
z = Flatten[outer[X[[1]], X[[2]]]];                  (* x1 (x) x2, lexicographic *)
w = Flatten[outer[X[[3]], X[[4]]]];
xi4 = Flatten[Outer[Times, X[[1]], X[[2]], X[[3]], X[[4]]]];
A = ArrayReshape[Table[a[al, i1, i2], {i1, d1}, {i2, d2}, {al, bondL}], {d12, bondL}];
B = ArrayReshape[Table[b[be, i3, i4], {i3, d3}, {i4, d4}, {be, bondR}], {d34, bondR}];
Cm = Table[c[al, be], {al, bondL}, {be, bondR}];
params = Join[Flatten[A], Flatten[B], Flatten[Cm]];
symsA = Flatten[Table[a[al, i1, i2], {al, bondL}, {i1, d1}, {i2, d2}]];
symsB = Flatten[Table[b[be, i3, i4], {be, bondR}, {i3, d3}, {i4, d4}]];
symsC = Flatten[Cm];

(* Teacher with smaller bond dimensions D*_L = 1, D*_R = 2. *)
{tA, tC, tB} = {RandomInteger[{-2, 2}, {d12, 1}], RandomInteger[{-2, 2}, {1, 2}], RandomInteger[{-2, 2}, {d34, 2}]};
Mstar = tA.tC.Transpose[tB];
M = A.Cm.Transpose[B];
DeltaTheta = M - Mstar;
Dm = Table[del[i, j], {i, d12}, {j, d34}];           (* generic coefficient error *)
eD = z.Dm.w;

section["3.1 Forward model"];
hL = Table[Sum[a[al, i1, i2] X[[1, i1]] X[[2, i2]], {i1, d1}, {i2, d2}], {al, bondL}];
hR = Table[Sum[b[be, i3, i4] X[[3, i3]] X[[4, i4]], {i3, d3}, {i4, d4}], {be, bondR}];
g = Sum[Cm[[al, be]] hL[[al]] hR[[be]], {al, bondL}, {be, bondR}];
t4 = Table[Sum[a[al, i1, i2] c[al, be] b[be, i3, i4], {al, bondL}, {be, bondR}],
   {i1, d1}, {i2, d2}, {i3, d3}, {i4, d4}];
check["h^L = A^T z, h^R = B^T w   (d2-hL, d2-hR, d2-matrix-forward)",
  zeroQ[hL - Transpose[A].z] && zeroQ[hR - Transpose[B].w]];
check["g = <T, x1 (x) x2 (x) x3 (x) x4>   (d2-output-index, d2-coefficient-tensor)", zeroQ[g - Flatten[t4].xi4]];
check["g = z^T A C B^T w   (d2-matrix-forward)", zeroQ[g - z.A.Cm.Transpose[B].w]];
check["T^<{1,2}> (lexicographic) = A C B^T   (d2-matricization)", zeroQ[ArrayReshape[t4, {d12, d34}] - M]];
check["vec(M) = vec(T) and vec(z w^T) = z (x) w = x1 (x) x2 (x) x3 (x) x4 (row-by-row vec)",
  zeroQ[vecRow[M] - Flatten[t4]] && zeroQ[vecRow[outer[z, w]] - Flatten[KroneckerProduct[z, w]]] &&
   zeroQ[vecRow[outer[z, w]] - xi4]];
check["Lemma cut at the left bond: F_e = A, H_e = B C^T", zeroQ[M - A.Transpose[B.Transpose[Cm]]]];
delta4 = t4 - ArrayReshape[Mstar, dims];             (* (d2-index-error) *)
check["(d2-index-error) is the 4-index form of Delta = A C B^T - M*",
  zeroQ[ArrayReshape[delta4, {d12, d34}] - DeltaTheta]];

(* ------------------------------------------------------------------------ *)
(* Input laws with exact moments                                             *)
(* ------------------------------------------------------------------------ *)
powProd[pt_, k_] := Times @@ MapThread[If[#2 == 0, 1, #1^#2] &, {pt, k}];   (* avoids 0^0 *)
productAtoms[blocks_] := Function[combo, {Times @@ combo[[All, 1]], Join @@ combo[[All, 2]]}] /@ Tuples[blocks];
rademacher[d_] := {1/2^d, #} & /@ Tuples[{-1, 1}, d];
homogenized[d_] := {1/2^(d - 1), Prepend[#, 1]} & /@ Tuples[{-1, 1}, d - 1];  (* E[xi xi^T] = I, E[xi] != 0 *)
randLower[n_] := Table[Which[j == i, 1, j < i, RandomChoice[{-1, 0, 1}], True, 0], {i, n}, {j, n}];

sq = {{{2, 1}, {1, 1}}, {{2, 1, 0}, {1, 2, 1}, {0, 1, 2}}, {{1, 1/2}, {1/2, 1}}, {{3, 1}, {1, 1}}};
lb = {{{1, 0}, {1, 2}}, {{1, 0, 0}, {-1, 1, 0}, {2, 1, 1}}, {{2, 0}, {1, 1}}, {{1, 0}, {-2, 1}}};
l12 = randLower[5]; l12[[3, 1]] = 1;                 (* x1 and x2 correlated *)
l34 = randLower[4]; l34[[3, 1]] = 1;
l7 = randLower[nx]; l7[[6, 1]] = 1; l7[[8, 3]] = 1;  (* correlations across the cut *)

laws = <|
   "L1 indep. N(0,I)" -> <|"type" -> "gauss", "mu" -> ConstantArray[0, nx], "cov" -> IdentityMatrix[nx],
      "white" -> True, "indep" -> True, "cut" -> True|>,
   "L2 indep. Rademacher" -> <|"type" -> "discrete", "atoms" -> productAtoms[rademacher /@ dims],
      "white" -> True, "indep" -> True, "cut" -> True|>,
   "L3 indep. homogenized (1,u), non-centred" -> <|"type" -> "discrete", "atoms" -> productAtoms[homogenized /@ dims],
      "white" -> True, "indep" -> True, "cut" -> True|>,
   "L4 indep. N(0, S_a^2)" -> <|"type" -> "gauss", "mu" -> ConstantArray[0, nx], "cov" -> blockDiag[#.# & /@ sq],
      "white" -> False, "indep" -> True, "cut" -> True|>,
   "L4b indep. Gaussian, non-centred" -> <|"type" -> "gauss", "mu" -> {1, -1, 0, 2, 1, -1, 0, 1, 1},
      "cov" -> blockDiag[#.Transpose[#] & /@ lb], "white" -> False, "indep" -> True, "cut" -> True|>,
   "L5 indep. discrete, singular Sigma_1, Sigma_2" -> <|"type" -> "discrete", "atoms" -> productAtoms[{
        {1/2, {#, #}} & /@ {-1, 1},
        Flatten[Table[{1/4, {u1, u2, u1 + u2}}, {u1, {-1, 1}}, {u2, {-1, 1}}], 1],
        {{1/3, {2, 1}}, {1/3, {-1, 1}}, {1/3, {-1, -2}}},
        {{1/4, {1, 0}}, {1/4, {-1, 0}}, {1/4, {0, 3}}, {1/4, {0, -3}}}}],
      "white" -> False, "indep" -> True, "cut" -> True|>,
   "L6 Gaussian, (X1,X2) indep. of (X3,X4), X1~X2 correlated" -> <|"type" -> "gauss", "mu" -> ConstantArray[0, nx],
      "cov" -> blockDiag[{l12.Transpose[l12], l34.Transpose[l34]}], "white" -> False, "indep" -> False, "cut" -> True|>,
   "L7 Gaussian, dependent across the cut, non-centred" -> <|"type" -> "gauss", "mu" -> {1, 0, -1, 0, 1, 0, 0, 1, -1},
      "cov" -> l7.Transpose[l7], "white" -> False, "indep" -> False, "cut" -> False|>
   |>;

ClearAll[mom];
mom[name_, k_] := mom[name, k] = With[{law = laws[name]},
    If[law["type"] === "gauss",
     Moment[MultinormalDistribution[law["mu"], law["cov"]], k],
     Total[Function[atom, atom[[1]] powProd[atom[[2]], k]] /@ law["atoms"]]]];
expect[name_, expr_] := Expand[Total[(Last[#] mom[name, First[#]]) & /@ CoefficientRules[Expand[expr], xs]]];

idxGrads[dt_] := {
   Flatten[Table[Sum[dt[[i1, i2, i3, i4]] c[al, be] b[be, i3, i4], {be, bondR}, {i3, d3}, {i4, d4}],
     {al, bondL}, {i1, d1}, {i2, d2}]],
   Flatten[Table[Sum[dt[[i1, i2, i3, i4]] a[al, i1, i2] c[al, be], {al, bondL}, {i1, d1}, {i2, d2}],
     {be, bondR}, {i3, d3}, {i4, d4}]],
   Flatten[Table[Sum[dt[[i1, i2, i3, i4]] a[al, i1, i2] b[be, i3, i4], {i1, d1}, {i2, d2}, {i3, d3}, {i4, d4}],
     {al, bondL}, {be, bondR}]]};

(* ------------------------------------------------------------------------ *)
(* Sections 3.2-3.4: loss, residual moment and gradients, law by law          *)
(* ------------------------------------------------------------------------ *)
store = <||>;
Do[
  Module[{law = laws[name], lossD, resid, s12, s34, sig, mean, phi, s1234, vD, sub, lossTheta, residTheta,
     grads, formula, gradsOf, gA, gB, gC, eS, dS, time0 = AbsoluteTime[]},
   section[name];
   lossD = 1/2 expect[name, eD^2];
   resid = Table[expect[name, eD z[[i]] w[[j]]], {i, d12}, {j, d34}];
   s12 = Table[expect[name, z[[i]] z[[j]]], {i, d12}, {j, d12}];
   s34 = Table[expect[name, w[[i]] w[[j]]], {i, d34}, {j, d34}];
   sig = Table[expect[name, X[[k, i]] X[[k, j]]], {k, 4}, {i, dims[[k]]}, {j, dims[[k]]}];
   mean = expect[name, #] & /@ xs;
   phi = vecRow[outer[z, w]];
   s1234 = Table[expect[name, phi[[p]] phi[[q]]], {p, d12 d34}, {q, d12 d34}];
   vD = vecRow[Dm];
   Print["  (moments in ", Round[AbsoluteTime[] - time0, 0.1], "s; E[X] = ", mean, ")"];

   check["grad_M L = R_Delta := E[e z w^T]   (d2-residual-moment, d2-general-coeff-gradient)",
    zeroQ[Table[D[lossD, del[i, j]], {i, d12}, {j, d34}] - resid]];
   check["R_Delta is the {1,2}-matricization of E[e x1 (x) x2 (x) x3 (x) x4]",
    zeroQ[unvecRow[expect[name, eD #] & /@ xi4, d12, d34] - resid]];
   check["vec(R) = Sigma_[4] vec(Delta) and L = 1/2 vec^T Sigma_[4] vec   (d2-vectorized-loss)",
    zeroQ[vecRow[resid] - s1234.vD] && zeroQ[lossD - 1/2 vD.s1234.vD]];

   sub = Thread[Flatten[Dm] -> Flatten[DeltaTheta]];
   lossTheta = lossD /. sub;
   residTheta = resid /. sub;
   grads = D[lossTheta, #] & /@ params;
   formula = Join[Flatten[residTheta.B.Transpose[Cm]], Flatten[Transpose[residTheta].A.Cm],
     Flatten[Transpose[A].residTheta.B]];
   check["end-to-end: d/d(core) L(A C B^T - M*) = (R B C^T, R^T A C, A^T R B)   (d2-general-gradients)",
    zeroQ[grads - formula]];
   gradsOf[syms_] := D[lossTheta, #] & /@ syms;

   If[law["white"],
    check["E[z z^T] = I, E[w w^T] = I   (d2-z-white, d2-w-white)",
     zeroQ[s12 - IdentityMatrix[d12]] && zeroQ[s34 - IdentityMatrix[d34]]];
    check["L = 1/2 ||Delta||_F^2   (d2-white-loss)", zeroQ[lossD - 1/2 frob[Dm, Dm]]];
    check["E[e z w^T] = Delta   (d2-residual-white)", zeroQ[resid - Dm]];
    check["Sigma_[4] = I", zeroQ[s1234 - IdentityMatrix[d12 d34]]];
    check["gradient flow = (d2-white-flow)",
     zeroQ[-grads - Join[Flatten[(Mstar - M).B.Transpose[Cm]], Flatten[Transpose[Mstar - M].A.Cm],
         Flatten[Transpose[A].(Mstar - M).B]]]];
    {gA, gB, gC} = idxGrads[delta4];
    check["index-form gradients   (d2-index-grad-A, -B, -C)",
     zeroQ[gradsOf[symsA] - gA] && zeroQ[gradsOf[symsB] - gB] && zeroQ[gradsOf[symsC] - gC]];
    {gA, gB, gC} = idxGrads[-delta4];
    check["index-form flow with E := T* - T(theta)   (d2-index-flow)",
     zeroQ[-gradsOf[symsA] - gA] && zeroQ[-gradsOf[symsB] - gB] && zeroQ[-gradsOf[symsC] - gC]];
    If[AnyTrue[mean, # =!= 0 &], Print["  note: this law is NOT centred; only E[X X^T] = I is used."]];
   ];

   If[law["indep"],
    check["Sigma_12 = Sigma_1 (x) Sigma_2, Sigma_34 = Sigma_3 (x) Sigma_4   (d2-gram-LR)",
     zeroQ[s12 - KroneckerProduct[sig[[1]], sig[[2]]]] && zeroQ[s34 - KroneckerProduct[sig[[3]], sig[[4]]]]];
    check["L = 1/2 tr(Delta^T Sigma_12 Delta Sigma_34)   (d2-nonwhite-loss)",
     zeroQ[lossD - 1/2 Tr[Transpose[Dm].s12.Dm.s34]]];
    check["grad_M L = E[e z w^T] = Sigma_12 Delta Sigma_34   (d2-coeff-gradient-nonwhite, d2-residual-nonwhite)",
     zeroQ[resid - s12.Dm.s34]];
    eS = s12.(Mstar - M).s34;
    check["gradient flow = (d2-nonwhite-flow), and E_Sigma = -R_Delta",
     zeroQ[-grads - Join[Flatten[eS.B.Transpose[Cm]], Flatten[Transpose[eS].A.Cm], Flatten[Transpose[A].eS.B]]] &&
      zeroQ[eS + residTheta]];
    check["Sigma_[4] = Sigma_12 (x) Sigma_34 = Sigma_1 (x) Sigma_2 (x) Sigma_3 (x) Sigma_4   (d2-gamma-product)",
     zeroQ[s1234 - KroneckerProduct[s12, s34]] && zeroQ[s1234 - KroneckerProduct @@ sig]];
    dS = Table[Sum[sig[[1, i1, j1]] sig[[2, i2, j2]] sig[[3, i3, j3]] sig[[4, i4, j4]] delta4[[j1, j2, j3, j4]],
       {j1, d1}, {j2, d2}, {j3, d3}, {j4, d4}], {i1, d1}, {i2, d2}, {i3, d3}, {i4, d4}];
    check["(d2-weighted-index-error) matricizes to Sigma_12 Delta Sigma_34",
     zeroQ[ArrayReshape[dS, {d12, d34}] - s12.DeltaTheta.s34]];
    {gA, gB, gC} = idxGrads[dS];
    check["index-form gradients with Delta replaced by Delta_Sigma",
     zeroQ[gradsOf[symsA] - gA] && zeroQ[gradsOf[symsB] - gB] && zeroQ[gradsOf[symsC] - gC]];
    If[AnyTrue[mean, # =!= 0 &],
     Print["  note: this law is NOT centred; Sigma_a := E[X X^T] (second moment), not the covariance."]];
   ];

   If[law["cut"] && ! law["indep"],
    check["blocks dependent within a side: Sigma_12 != Sigma_1 (x) Sigma_2",
     ! zeroQ[s12 - KroneckerProduct[sig[[1]], sig[[2]]]]];
    check["...yet Proposition d2-nonwhite-flow holds with Sigma_12 := E[z z^T], Sigma_34 := E[w w^T]",
     zeroQ[lossD - 1/2 Tr[Transpose[Dm].s12.Dm.s34]] && zeroQ[resid - s12.Dm.s34]];
    check["...and Sigma_[4] = Sigma_12 (x) Sigma_34, which is not Sigma_1 (x) ... (x) Sigma_4",
     zeroQ[s1234 - KroneckerProduct[s12, s34]] && ! zeroQ[s1234 - KroneckerProduct @@ sig]];
   ];

   If[! law["cut"],
    check["dependence across the cut: R_Delta != Sigma_12 Delta Sigma_34", ! zeroQ[resid - s12.Dm.s34]];
    check["dependence across the cut: Sigma_[4] is not Sigma_12 (x) Sigma_34", ! zeroQ[s1234 - KroneckerProduct[s12, s34]]];
   ];

   store[name] = <|"lossD" -> lossD, "resid" -> resid, "s12" -> s12, "s34" -> s34, "s1234" -> s1234,
     "residTheta" -> residTheta, "grads" -> grads|>;
   Print["  (", Round[AbsoluteTime[] - time0, 0.1], "s)"];
  ],
  {name, Keys[laws]}];

(* ------------------------------------------------------------------------ *)
(* Remark: whitened analysis coordinates (law L4, Sigma_a = S_a^2)           *)
(* ------------------------------------------------------------------------ *)
section["Remark: whitened analysis coordinates (L4)"];
With[{st = store["L4 indep. N(0, S_a^2)"]},
  Module[{sqL = KroneckerProduct[sq[[1]], sq[[2]]], sqR = KroneckerProduct[sq[[3]], sq[[4]]],
     tilA, tilB, tilM, residT, gradA, gradB, gradC, gradTA, gradTB, gradTC, pt},
   check["Sigma_12^{1/2} = S_1 (x) S_2, Sigma_34^{1/2} = S_3 (x) S_4 (positive definite square roots)",
    zeroQ[sqL.sqL - st["s12"]] && zeroQ[sqR.sqR - st["s34"]] &&
     PositiveDefiniteMatrixQ[sqL] && PositiveDefiniteMatrixQ[sqR]];
   check["L = 1/2 ||Sigma_12^{1/2} Delta Sigma_34^{1/2}||_F^2   (d2-nonwhite-loss)",
    zeroQ[st["lossD"] - 1/2 frob[sqL.Dm.sqR, sqL.Dm.sqR]]];
   {tilA, tilB, tilM} = {sqL.A, sqR.B, sqL.Mstar.sqR};
   residT = tilA.Cm.Transpose[tilB] - tilM;
   check["Atilde C Btilde^T - Mtilde* = Sigma_12^{1/2} Delta Sigma_34^{1/2}", zeroQ[residT - sqL.DeltaTheta.sqR]];
   gradA = st["residTheta"].B.Transpose[Cm];
   gradB = Transpose[st["residTheta"]].A.Cm;
   gradC = Transpose[A].st["residTheta"].B;
   {gradTA, gradTB, gradTC} = {residT.tilB.Transpose[Cm], Transpose[residT].tilA.Cm, Transpose[tilA].residT.tilB};
   check["GF in (A,B,C): d/dt Atilde = -Sigma_12 grad Ltilde, d/dt Btilde = -Sigma_34 grad Ltilde, d/dt C = -grad_C Ltilde",
    zeroQ[sqL.gradA - st["s12"].gradTA] && zeroQ[sqR.gradB - st["s34"].gradTB] && zeroQ[gradC - gradTC]];
   pt = Thread[params -> RandomInteger[{-4, 4}, Length[params]]/RandomInteger[{1, 3}, Length[params]]];
   check["...which differs from Euclidean gradient flow in (Atilde, Btilde, C)",
    ! zeroQ[(sqL.gradA - gradTA) /. pt]];
  ]];

(* ------------------------------------------------------------------------ *)
(* Section 3.5: differential, adjoint, tangent operator (generic matrices)   *)
(* ------------------------------------------------------------------------ *)
section["3.5 Differential, adjoint and tangent operator (generic symbolic A, B, C, X)"];
dPhi[aa_, bb_, cc_, da_, db_, dc_] := da.cc.Transpose[bb] + aa.cc.Transpose[db] + aa.dc.Transpose[bb];  (* (d2-DPhi) *)
dPhiAdj[aa_, bb_, cc_, xx_] := {xx.bb.Transpose[cc], Transpose[xx].aa.cc, Transpose[aa].xx.bb};       (* (d2-DPhi-adjoint) *)
kOp[aa_, bb_, cc_, xx_] := aa.cc.Transpose[cc].Transpose[aa].xx + aa.Transpose[aa].xx.bb.Transpose[bb] +
   xx.bb.Transpose[cc].cc.Transpose[bb];                                                             (* (d2-K-operator) *)

gA0 = Array[ga, {d12, bondL}]; gB0 = Array[gb, {d34, bondR}]; gC0 = Array[gc, {bondL, bondR}];
gX0 = Array[gx, {d12, d34}];
dA0 = Array[da, {d12, bondL}]; dB0 = Array[db, {d34, bondR}]; dC0 = Array[dc, {bondL, bondR}];
adj0 = dPhiAdj[gA0, gB0, gC0, gX0];
check["d/dt Phi(A+t dA, B+t dB, C+t dC)|_0 = DPhi   (d2-DPhi; dM in d2-differential-proof)",
  zeroQ[(D[(gA0 + t dA0).(gC0 + t dC0).Transpose[gB0 + t dB0], t] /. t -> 0) - dPhi[gA0, gB0, gC0, dA0, dB0, dC0]]];
check["<X, DPhi[dA,dB,dC]> = <X B C^T, dA> + <X^T A C, dB> + <A^T X B, dC>   (d2-DPhi-adjoint)",
  zeroQ[frob[gX0, dPhi[gA0, gB0, gC0, dA0, dB0, dC0]] - frob[adj0[[1]], dA0] - frob[adj0[[2]], dB0] -
     frob[adj0[[3]], dC0]]];
check["DPhi DPhi^* [X] = A C C^T A^T X + A A^T X B B^T + X B C^T C B^T   (d2-M-flow, d2-K-operator)",
  zeroQ[dPhi[gA0, gB0, gC0, Sequence @@ adj0] - kOp[gA0, gB0, gC0, gX0]]];
check["<X, K[X]> = ||X B C^T||^2 + ||X^T A C||^2 + ||A^T X B||^2   (d2-K-positive)",
  zeroQ[frob[gX0, kOp[gA0, gB0, gC0, gX0]] - frob[adj0[[1]], adj0[[1]]] - frob[adj0[[2]], adj0[[2]]] -
     frob[adj0[[3]], adj0[[3]]]]];
jac0 = D[vecRow[gA0.gC0.Transpose[gB0]], {Join[Flatten[gA0], Flatten[gB0], Flatten[gC0]]}];
check["J J^T vec(X) = vec(K[X]) (row-by-row vec)",
  zeroQ[jac0.(Transpose[jac0].vecRow[gX0]) - vecRow[kOp[gA0, gB0, gC0, gX0]]]];

Module[{xi, xj, zi, wi, zj, wj, gi, gj},
  xi = Thread[xs -> RandomInteger[{-3, 3}, nx]]; xj = Thread[xs -> RandomInteger[{-3, 3}, nx]];
  {zi, wi, zj, wj} = {z /. xi, w /. xi, z /. xj, w /. xj};
  gi = zi.M.wi; gj = zj.M.wj;
  check["tangent kernel: <grad g(x_i), grad g(x_j)> = <z_i w_i^T, K[z_j w_j^T]>   (after d2-K-positive)",
   zeroQ[(D[gi, #] & /@ params).(D[gj, #] & /@ params) - frob[outer[zi, wi], kOp[A, B, Cm, outer[zj, wj]]]]];
  check["grad_theta g(x) = DPhi^*[z w^T]",
   zeroQ[(D[gi, #] & /@ params) - Flatten[dPhiAdj[A, B, Cm, outer[zi, wi]]]]];
  ];

section["3.5 Coefficient-space flow and loss decay at an exact random point (law L7)"];
With[{st = store["L7 Gaussian, dependent across the cut, non-centred"]},
  Module[{pt, gradPt, thetaDot, jacPt, mDot, residPt, aP, bP, cP, dLdt},
   pt = Thread[params -> RandomInteger[{-4, 4}, Length[params]]/RandomInteger[{1, 3}, Length[params]]];
   gradPt = st["grads"] /. pt;
   thetaDot = -gradPt;
   jacPt = D[vecRow[M], {params}] /. pt;
   mDot = jacPt.thetaDot;
   residPt = st["residTheta"] /. pt;
   {aP, bP, cP} = {A, B, Cm} /. pt;
   check["dm/dt = -J J^T Sigma_[4] (m - m*)   (d2-vectorized-flow)",
    zeroQ[mDot + jacPt.Transpose[jacPt].st["s1234"].vecRow[DeltaTheta /. pt]]];
   check["dM/dt = -K[R_Delta]   (d2-M-flow)", zeroQ[unvecRow[mDot, d12, d34] + kOp[aP, bP, cP, residPt]]];
   dLdt = gradPt.thetaDot;
   check["dL/dt = -sum ||grad||^2 = -<R, K[R]> <= 0   (d2-loss-decay)",
    zeroQ[dLdt + frob[residPt, kOp[aP, bP, cP, residPt]]] && dLdt <= 0];
  ]];

(* ------------------------------------------------------------------------ *)
(* Proposition (balance invariants)                                          *)
(* ------------------------------------------------------------------------ *)
section["Balance invariants (generic symbolic R)"];
gR0 = Array[gr, {d12, d34}];
{aDot, bDot, cDot} = {-gR0.gB0.Transpose[gC0], -Transpose[gR0].gA0.gC0, -Transpose[gA0].gR0.gB0};
dAtA = Transpose[aDot].gA0 + Transpose[gA0].aDot;
dBtB = Transpose[bDot].gB0 + Transpose[gB0].bDot;
check["d/dt A^T A = -C B^T R^T A - A^T R B C^T = d/dt C C^T",
  zeroQ[dAtA - (-gC0.Transpose[gB0].Transpose[gR0].gA0 - Transpose[gA0].gR0.gB0.Transpose[gC0])] &&
   zeroQ[dAtA - (cDot.Transpose[gC0] + gC0.Transpose[cDot])]];
check["d/dt B^T B = -C^T A^T R B - B^T R^T A C = d/dt C^T C",
  zeroQ[dBtB - (-Transpose[gC0].Transpose[gA0].gR0.gB0 - Transpose[gB0].Transpose[gR0].gA0.gC0)] &&
   zeroQ[dBtB - (Transpose[cDot].gC0 + Transpose[gC0].cDot)]];
Module[{xl, yr},
  xl = RandomInteger[{-2, 2}, {bondL, bondL}]; While[Det[xl] == 0, xl = RandomInteger[{-2, 2}, {bondL, bondL}]];
  yr = RandomInteger[{-2, 2}, {bondR, bondR}]; While[Det[yr] == 0, yr = RandomInteger[{-2, 2}, {bondR, bondR}]];
  check["gauge action (A,C,B) -> (AX, X^{-1} C Y^{-T}, BY) leaves M = A C B^T unchanged",
   zeroQ[(gA0.xl).(Inverse[xl].gC0.Transpose[Inverse[yr]]).Transpose[gB0.yr] - gA0.gC0.Transpose[gB0]]];
  ];

section["Balance invariants and loss decay along numerically integrated flows"];
integrate[name_, tEnd_ : 60, scale_ : 0.3] := Module[{gm, ms, unpack, rhs, lossN, inv, th0, sol, ts, losses, q0, drift},
   gm = N[store[name]["s1234"]]; gm = gm/Max[Eigenvalues[gm]];
   ms = N[Mstar]; ms = ms/Norm[ms, "Frobenius"];
   unpack[th_] := {ArrayReshape[th[[1 ;; d12 bondL]], {d12, bondL}],
     ArrayReshape[th[[d12 bondL + 1 ;; d12 bondL + d34 bondR]], {d34, bondR}],
     ArrayReshape[th[[d12 bondL + d34 bondR + 1 ;;]], {bondL, bondR}]};
   rhs[th_?(VectorQ[#, NumericQ] &)] := Module[{aa, bb, cc, rr},
     {aa, bb, cc} = unpack[th];
     rr = unvecRow[gm.vecRow[aa.cc.Transpose[bb] - ms], d12, d34];
     Join[Flatten[-rr.bb.Transpose[cc]], Flatten[-Transpose[rr].aa.cc], Flatten[-Transpose[aa].rr.bb]]];
   lossN[th_] := Module[{aa, bb, cc, v}, {aa, bb, cc} = unpack[th]; v = vecRow[aa.cc.Transpose[bb] - ms]; v.gm.v/2];
   inv[th_] := Module[{aa, bb, cc}, {aa, bb, cc} = unpack[th];
     Join[Flatten[Transpose[aa].aa - cc.Transpose[cc]], Flatten[Transpose[bb].bb - Transpose[cc].cc]]];
   th0 = scale RandomVariate[NormalDistribution[], Length[params]];
   sol = NDSolveValue[{th'[t] == rhs[th[t]], th[0] == th0}, th, {t, 0, tEnd},
     PrecisionGoal -> 12, AccuracyGoal -> 14, MaxSteps -> Infinity];
   ts = Subdivide[0., tEnd, 2000];
   losses = lossN[sol[#]] & /@ ts;
   q0 = inv[th0];
   drift = Max[Max[Abs[inv[sol[#]] - q0]] & /@ ts];
   Print["  ", name, ": loss ", ScientificForm[First[losses], 3], " -> ", ScientificForm[Last[losses], 3],
    "; max invariant drift ", ScientificForm[drift, 2]];
   check[name <> ": invariants conserved (drift < 1e-8) and loss non-increasing",
    drift < 10^-8 && Max[Differences[losses]] < 10^-12]];
integrate["L4 indep. N(0, S_a^2)"];
integrate["L7 Gaussian, dependent across the cut, non-centred"];

(* ------------------------------------------------------------------------ *)
(* Remarks: empirical loss and tied lower cores                              *)
(* ------------------------------------------------------------------------ *)
section["Remark: empirical loss"];
Module[{n = 5, samples, rules, es, lossN, residHat, sub},
  samples = RandomInteger[{-3, 3}, {n, nx}];
  rules = Thread[xs -> #] & /@ samples;
  es = eD /. # & /@ rules;
  lossN = 1/(2 n) Total[es^2];
  residHat = 1/n Sum[es[[s]] outer[z /. rules[[s]], w /. rules[[s]]], {s, n}];
  check["L_n := (1/2n) sum e_s^2 has grad_M L_n = Rhat",
   zeroQ[Table[D[lossN, del[i, j]], {i, d12}, {j, d34}] - residHat]];
  sub = Thread[Flatten[Dm] -> Flatten[DeltaTheta]];
  check["parameter gradients of L_n = (Rhat B C^T, Rhat^T A C, A^T Rhat B)",
   zeroQ[(D[lossN /. sub, #] & /@ params) -
     Join[Flatten[(residHat /. sub).B.Transpose[Cm]], Flatten[Transpose[residHat /. sub].A.Cm],
      Flatten[Transpose[A].(residHat /. sub).B]]]];
  ];

section["Remark: tied lower cores (whitened loss, d_1 = d_3, d_2 = d_4, D_L = D_R)"];
Module[{sS = Array[ss, {4, 2}], kK = Array[kk, {2, 2}], mT = RandomInteger[{-2, 2}, {4, 4}], dl, lt},
  dl = sS.kK.Transpose[sS] - mT;
  lt = 1/2 frob[dl, dl];
  check["grad of the shared core = grad_A L + grad_B L at A = B = S",
   zeroQ[Table[D[lt, ss[i, j]], {i, 4}, {j, 2}] - (dl.sS.Transpose[kK] + Transpose[dl].sS.kK)]];
  ];

(* ------------------------------------------------------------------------ *)
nFail = Count[results[[All, 2]], False];
Print["\n", Length[results] - nFail, "/", Length[results], " checks passed."];
Scan[Print["  FAILED: ", #[[1]]] &, Select[results, ! #[[2]] &]];
Exit[If[nFail > 0, 1, 0]];
