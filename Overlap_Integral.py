import numpy as np
from itertools import product
from dataclasses import dataclass

class Overlap_Integrals:
    """
    Franck–Condon overlaps in the parallel-mode limit (J = I, equal frequencies).

    Inputs (choose ONE constructor):
      - from_Delta(Delta_dimless, valid_mask=None)
          Δ_i are dimensionless displacements (Δ_i = sqrt(ω_i) * K_i)
      - from_K(K_shift, evals_int_au, valid_mask=None)
          K_i are mass-weighted normal-coordinate shifts; λ_i in a.u.

    Public API:
      - s00(): analytic 0–0 overlap
      - per_mode_factors(): S00_i per mode, for diagnostics
      - one_mode_table(i, vmax_m=0, vmax_n): 1D table O_i[m,n]
      - overlaps(vmax_per_mode, initial_quanta=None, return_intensity=True)
          → dict with S00, overlaps {n⃗: S}, and (optionally) intensities {n⃗: |S|²}
      - top_transitions(vmax_per_mode, k=20, initial_quanta=None) → sorted list
    """

    # ------------------------ construction ------------------------
    @classmethod
    def from_Delta(cls, Delta_dimless, valid_mask=None):
        Delta = np.asarray(Delta_dimless, float)
        if valid_mask is not None:
            Delta = Delta[np.asarray(valid_mask, bool)]
        return cls(Delta=Delta)

    @classmethod
    def from_K(cls, K_shift, evals_int_au, valid_mask=None):
        K = np.asarray(K_shift, float)
        lam = np.asarray(evals_int_au, float)
        vm = (lam > 0.0) if valid_mask is None else np.asarray(valid_mask, bool)
        omega = np.zeros_like(lam); omega[vm] = np.sqrt(lam[vm])
        Delta = np.zeros_like(lam); Delta[vm] = np.sqrt(omega[vm]) * K[vm]
        return cls(Delta=Delta[vm])

    # ------------------------ core state -------------------------
    def __init__(self, Delta):
        self.Delta = np.asarray(Delta, float)             # (Nvib,)
        self.alpha = self.Delta / np.sqrt(2.0)            # (Nvib,)
        self.Nv = self.alpha.size
        # cache for 1D tables per mode: dict[i] -> (mmax, nmax, table)
        self._tables = {}
        # precompute S00
        self._S00 = float(np.exp(-0.25 * np.dot(self.Delta, self.Delta)))

    # ------------------------ analytics --------------------------
    def s00(self) -> float:
        """Analytic 0–0 overlap: S00 = exp(-sum Δ_i^2 / 4)."""
        return self._S00

    def per_mode_factors(self):
        """Return S00_i per mode and their logs (contributions)."""
        S00_i = np.exp(-0.25 * self.Delta**2)
        return S00_i, np.log(S00_i)

    # ------------------------ 1D recursion -----------------------
    @staticmethod
    def _one_mode_table(alpha, vmax_m, vmax_n):
        """
        O[m,n] for <m | n(shift)> with recursion:
          sqrt(m+1) O[m+1,n] = α O[m,n] + sqrt(n) O[m,n-1]
        Bases:
          O[0,0] = exp(-α^2/2)
          O[0,n] = e * α^n / sqrt(n!)
          O[m,0] = e * (-α)^m / sqrt(m!)
        """
        mmax, nmax = int(vmax_m), int(vmax_n)
        O = np.zeros((mmax+1, nmax+1), dtype=float)
        e = np.exp(-0.5 * alpha * alpha)
        O[0,0] = e

        # first row (n > 0)
        if nmax >= 1:
            acc = 1.0; inv_sf = 1.0
            for n in range(1, nmax+1):
                acc *= alpha
                inv_sf /= np.sqrt(n)
                O[0, n] = e * acc * inv_sf

        # first col (m > 0)
        if mmax >= 1:
            acc = 1.0; inv_sf = 1.0
            for m in range(1, mmax+1):
                acc *= -alpha
                inv_sf /= np.sqrt(m)
                O[m, 0] = e * acc * inv_sf

        # forward fill
        for m in range(0, mmax):
            for n in range(0, nmax+1):
                term = alpha * O[m, n]
                if n >= 1:
                    term += np.sqrt(n) * O[m, n-1]
                O[m+1, n] = term / np.sqrt(m+1)
        return O

    def one_mode_table(self, i, vmax_m=0, vmax_n=10):
        """Get or build the 1D table for mode i up to (mmax, nmax)."""
        i = int(i)
        prev = self._tables.get(i, None)
        need_build = True
        if prev is not None:
            mmax_old, nmax_old, Oold = prev
            if vmax_m <= mmax_old and vmax_n <= nmax_old:
                return Oold[:vmax_m+1, :vmax_n+1]
        # (re)build to the new larger size
        Onew = self._one_mode_table(self.alpha[i], vmax_m, vmax_n)
        self._tables[i] = (vmax_m, vmax_n, Onew)
        return Onew

    # ------------------------ multimode product ------------------
    def overlaps(self, vmax_per_mode, initial_quanta=None, return_intensity=True):
        """
        Compute <m⃗ | n⃗ (shift)> for all 0<=n_i<=vmax_i with m⃗ default 0⃗.
        Returns: {'S00': float, 'overlaps': {n_tuple: S}, ['intensities': {n: |S|^2}]}
        """
        if isinstance(vmax_per_mode, int):
            vmax = np.full(self.Nv, vmax_per_mode, dtype=int)
        else:
            vmax = np.asarray(vmax_per_mode, dtype=int)
            if vmax.shape != (self.Nv,):
                raise ValueError("vmax_per_mode must be int or shape (Nvib,)")

        if initial_quanta is None:
            mvec = np.zeros(self.Nv, dtype=int)
        else:
            mvec = np.asarray(initial_quanta, dtype=int)
            if mvec.shape != (self.Nv,):
                raise ValueError("initial_quanta must have shape (Nvib,)")

        # build all needed 1D tables
        tables = [self.one_mode_table(i, vmax_m=int(mvec[i]), vmax_n=int(vmax[i]))
                  for i in range(self.Nv)]

        # multiply per mode
        overlaps = {}
        ranges = [range(vmax[i] + 1) for i in range(self.Nv)]
        for n_tuple in product(*ranges):
            val = 1.0
            for i, ni in enumerate(n_tuple):
                val *= tables[i][mvec[i], ni]
            overlaps[n_tuple] = val

        out = {"S00": self._S00, "overlaps": overlaps}
        if return_intensity:
            out["intensities"] = {k: v*v for k, v in overlaps.items()}
        return out

    def top_transitions(self, vmax_per_mode, k=20, initial_quanta=None):
        """Return top-k (n_tuple, S, I) sorted by intensity."""
        res = self.overlaps(vmax_per_mode, initial_quanta=initial_quanta, return_intensity=True)
        items = sorted(res["intensities"].items(), key=lambda kv: kv[1], reverse=True)[:k]
        tops = []
        for n, I in items:
            tops.append((n, res["overlaps"][n], I))
        return res["S00"], tops


