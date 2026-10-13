(* ::Package:: *)
(* Exact checks of Section 3.6 ("Gradient flow on the balanced manifold") of ttns_surrogate.tex.

   Run:  wolframscript -file sec3_balanced.wl

   Symbolic part of sec3_balanced.py (which also runs the numerical checks): balanced cores,
   closed equation for M, loss decay, spectral form of the tangent operator, aligned modes.
   Frames are exact rational orthogonal matrices (Cayley transforms). *)

results = {};
check[name_, ok_] := (AppendTo[results, {name, TrueQ[ok]}];
   Print[If[TrueQ[ok], "  [PASS] ", "  [FAIL] "], name]);
zeroQ[e_] := AllTrue[Flatten[{Expand[e]}], # === 0 &];
frob[p_, q_] := Total[p q, 2];
SeedRandom[2];
cayley[n_] := Module[{k = ConstantArray[0, {n, n}], v, q},
   Do[v = RandomInteger[{-3, 3}]/RandomInteger[{1, 3}]; k[[i, j]] = v; k[[j, i]] = -v, {i, n}, {j, i + 1, n}];
   q = (IdentityMatrix[n] - k).Inverse[IdentityMatrix[n] + k];
   If[! zeroQ[Transpose[q].q - IdentityMatrix[n]], Print["cayley failed"]]; q];

{d12, d34, bondL, bondR, k} = {6, 4, 2, 3, 2};
{uF, vF, wlF, wrF} = {cayley[d12], cayley[d34], cayley[bondL], cayley[bondR]};
{u, v, wl, wr} = {uF[[All, 1 ;; k]], vF[[All, 1 ;; k]], wlF[[All, 1 ;; k]], wrF[[All, 1 ;; k]]};
sv = Array[s, k]; S = DiagonalMatrix[sv];
A = Expand[u.S.Transpose[wl]]; Cm = Expand[wl.S.Transpose[wr]]; B = Expand[v.S.Transpose[wr]];
M = Expand[A.Cm.Transpose[B]];
pw[mat_, p_] := Expand[mat.DiagonalMatrix[sv^p].Transpose[mat]];   (* frame . S^p . frame^T *)
{l1, r1, l2, r2, lh, rh} = {pw[u, 2], pw[v, 2], pw[u, 4], pw[v, 4], pw[u, 1], pw[v, 1]};

Print["\n== Balanced cores"];
check["A^T A = C C^T and B^T B = C^T C", zeroQ[Transpose[A].A - Cm.Transpose[Cm]] && zeroQ[Transpose[B].B - Transpose[Cm].Cm]];
check["M = U S^3 V^T", zeroQ[M - u.DiagonalMatrix[sv^3].Transpose[v]]];
check["A A^T = U S^2 U^T, (U S^2 U^T)^3 = M M^T; B B^T = V S^2 V^T, (V S^2 V^T)^3 = M^T M",
  zeroQ[A.Transpose[A] - l1] && zeroQ[MatrixPower[l1, 3] - M.Transpose[M]] &&
   zeroQ[B.Transpose[B] - r1] && zeroQ[MatrixPower[r1, 3] - Transpose[M].M]];

Print["\n== Closed equation for M and loss decay"];
rg = Array[rr, {d12, d34}];
{aDot, bDot, cDot} = {Expand[-rg.Expand[B.Transpose[Cm]]], Expand[-Transpose[rg].Expand[A.Cm]], Expand[-Transpose[A].rg.B]};
mDot = Expand[aDot.Cm.Transpose[B] + A.cDot.Transpose[B] + A.Cm.Transpose[bDot]];
check["Mdot = -(MM^T)^{2/3} R - (MM^T)^{1/3} R (M^TM)^{1/3} - R (M^TM)^{2/3}",
  zeroQ[mDot - (-l2.rg - l1.rg.r1 - rg.r2)]];
check["dL/dt = -||(MM^T)^{1/3}R||^2 - ||(MM^T)^{1/6} R (M^TM)^{1/6}||^2 - ||R (M^TM)^{1/3}||^2",
  zeroQ[-(frob[aDot, aDot] + frob[bDot, bDot] + frob[cDot, cDot]) +
     frob[l1.rg, l1.rg] + frob[lh.rg.rh, lh.rg.rh] + frob[rg.r1, rg.r1]]];

kOp[x_] := A.Cm.Transpose[Cm].Transpose[A].x + A.Transpose[A].x.B.Transpose[B] + x.B.Transpose[Cm].Cm.Transpose[B];
left = Table[If[i <= k, sv[[i]]^2, 0], {i, d12}];    (* sigma_i^{2/3} *)
right = Table[If[j <= k, sv[[j]]^2, 0], {j, d34}];
check["K[u_i v_j^T] = kappa_ij u_i v_j^T in complete singular bases",
  AllTrue[Flatten[Table[
     zeroQ[kOp[Outer[Times, uF[[All, i]], vF[[All, j]]]] -
       (left[[i]]^2 + left[[i]] right[[j]] + right[[j]]^2) Outer[Times, uF[[All, i]], vF[[All, j]]]],
     {i, d12}, {j, d34}]], TrueQ]];

Print["\n== Aligned modes"];
lL = Join[Array[gl, k], Range[2, d12 - k + 1]];      (* eigenvalues of Sigma_12 on u_i; numbers off the modes *)
lR = Join[Array[gr, k], Range[3, d34 - k + 2]];
s12 = Expand[uF.DiagonalMatrix[lL].Transpose[uF]]; s34 = Expand[vF.DiagonalMatrix[lR].Transpose[vF]];
mStar = Expand[u.DiagonalMatrix[Array[ts, k]].Transpose[v]];
ral = Expand[Expand[s12.(M - mStar)].s34];
sDot = DiagonalMatrix[Table[gl[i] gr[i] s[i]^2 (ts[i] - s[i]^3), {i, k}]];
check["Adot = U Sdot W_L^T, Cdot = W_L Sdot W_R^T, Bdot = V Sdot W_R^T",
  zeroQ[-ral.Expand[B.Transpose[Cm]] - u.sDot.Transpose[wl]] &&
   zeroQ[-Transpose[A].Expand[ral.B] - wl.sDot.Transpose[wr]] &&
   zeroQ[-Transpose[ral].Expand[A.Cm] - v.sDot.Transpose[wr]]];

nFail = Count[results[[All, 2]], False];
Print["\n", Length[results] - nFail, "/", Length[results], " checks passed."];
Exit[If[nFail > 0, 1, 0]];
