#!/usr/bin/env python3
# TDDFT_HT_mpi.py — TDDFT & Herzberg–Teller derivatives with optional MPI parallelism

import numpy as np
from pyscf import dft, tdscf

# ---- Optional MPI (rank-aware parallel finite differences) ----
try:
    from mpi4py import MPI  # pip install mpi4py
    _COMM = MPI.COMM_WORLD
    _RANK = _COMM.Get_rank()
    _SIZE = _COMM.Get_size()
except Exception:
    _COMM = None
    _RANK = 0
    _SIZE = 1

DEBYE_PER_AU = 2.541746473
EV_PER_AU    = 27.211386245988
BOHR2ANG     = 0.529177210903

class TDDFTHT_Parallel:
    """
    One-stop TDDFT toolkit:
      • compute reference-state TDDFT properties (energies, TDMs, f-osc)
      • compute Herzberg–Teller (HT) derivatives dμ/dQ along vib. normal modes
    """

    # -------------------- Convenience builders --------------------
    @classmethod
    def from_freq(cls, mf, fr, *, istate=0, nstates=5, dq_Q=0.03,
                  use_richardson=True, tracker_sigma_e_eV=0.3, tracker_w_cos=0.85):
        tdht = cls(mf,
                   nstates=nstates,
                   tracker_sigma_e_eV=tracker_sigma_e_eV,
                   tracker_w_cos=tracker_w_cos)
        tdht.geom_ref = fr.atom_list
        tdht.l_MWC = np.asarray(fr.l_MWC, float)
        tdht.evals_int_au = np.asarray(fr.evals_int_au, float)
        tdht.mass_vec_3N = np.asarray(fr.mass_vec, float)
        tdht.compute_reference(istate=istate)
        tdht.compute_ht_derivatives(istate=istate, dq_Q=dq_Q, use_richardson=use_richardson)
        return tdht

    @classmethod
    def from_minimal(cls, mf, *, geom_ref_atomlist, l_MWC, evals_int_au, mass_vec_3N,
                     istate=0, nstates=5, dq_Q=0.03, use_richardson=True,
                     tracker_sigma_e_eV=0.3, tracker_w_cos=0.85):
        tdht = cls(mf,
                   nstates=nstates,
                   tracker_sigma_e_eV=tracker_sigma_e_eV,
                   tracker_w_cos=tracker_w_cos)
        tdht.geom_ref = geom_ref_atomlist
        tdht.l_MWC = np.asarray(l_MWC, float)
        tdht.evals_int_au = np.asarray(evals_int_au, float)
        tdht.mass_vec_3N = np.asarray(mass_vec_3N, float)
        tdht.compute_reference(istate=istate)
        tdht.compute_ht_derivatives(istate=istate, dq_Q=dq_Q, use_richardson=use_richardson)
        return tdht

    # --------------------------- Core ------------------------------
    def __init__(self, mf, *, nstates=5, grid_level=None, conv_tol=None,
                 tracker_sigma_e_eV=0.3, tracker_w_cos=0.85):
        if not hasattr(mf, 'mol'):
            raise TypeError("mf must be a PySCF mean-field object with .mol")
        self.mf = mf
        self.mol = mf.mol
        try:
            self.unit = 'Angstrom' if str(self.mol.unit).lower().startswith('ang') else 'Bohr'
        except Exception:
            self.unit = 'Angstrom'
        self.nstates = int(nstates)
        self.grid_level = grid_level if grid_level is not None else getattr(mf.grids, 'level', 3)
        self.conv_tol = conv_tol if conv_tol is not None else getattr(mf, 'conv_tol', 1e-9)
        self.tracker_sigma_e_eV = float(tracker_sigma_e_eV)
        self.tracker_w_cos = float(tracker_w_cos)

        # Will be populated by convenience builders or manual setters:
        self.geom_ref = None
        self.l_MWC = None
        self.evals_int_au = None
        self.mass_vec_3N = None

        # Outputs:
        self.ref = None
        self.M0_au = None; self.M0_D = None
        self.M0_mag_au = None; self.M0_mag_D = None
        self.omega_elec_au = None; self.omega_elec_eV = None
        self.f_osc = None
        self.ht = None
        self.dmu_dQ_au = None
        self.dmu_dDelta_au = None
        self.valid_mask = None
        self.omega_vib_au = None

    # -------------------- Public: Reference TDDFT --------------------
    def compute_reference(self, istate=0):
        td = tdscf.TDDFT(self.mf)
        td.nstates = max(1, int(max(self.nstates, istate+1)))
        td.kernel()
        if td.e is None or len(td.e) < (istate+1):
            raise RuntimeError("TDDFT failed or returned too few states")
        e_au = np.array(td.e, dtype=float)
        M_au_all = np.array(td.transition_dipole(), dtype=float)
        f_osc = self._get_oscillator_strengths(td, e_au, M_au_all)

        M = M_au_all[istate]
        self.ref = {
            'e_au_all': e_au,
            'M_au_all': M_au_all,
            'f_osc_all': f_osc,
            'omega_au': float(e_au[istate]),
            'omega_eV': float(e_au[istate] * EV_PER_AU),
            'M_au': M,
            'M_D': M * DEBYE_PER_AU,
            'M_mag_au': float(np.linalg.norm(M)),
            'M_mag_D': float(np.linalg.norm(M) * DEBYE_PER_AU),
            'f_osc': float(f_osc[istate]),
            'istate': int(istate),
        }
        self.M0_au = self.ref['M_au']; self.M0_D = self.ref['M_D']
        self.M0_mag_au = self.ref['M_mag_au']; self.M0_mag_D = self.ref['M_mag_D']
        self.omega_elec_au = self.ref['omega_au']; self.omega_elec_eV = self.ref['omega_eV']
        self.f_osc = self.ref['f_osc']
        return self.ref

    # -------------------- Public: HT derivatives (MPI-parallel) --------------------
    def compute_ht_derivatives(self, istate=0, dq_Q=0.03, use_richardson=True):
        """
        Finite-difference HT derivatives dμ/dQ_k at reference geometry.
        Parallelizes across k over MPI ranks if available.
        """
        if self.geom_ref is None or self.l_MWC is None or self.evals_int_au is None or self.mass_vec_3N is None:
            raise ValueError("Provide geom_ref, l_MWC, evals_int_au, mass_vec_3N (use TDDFTHT.from_freq(...) for simplicity).")

        # Ensure reference (M0, E0) is available
        if self.ref is None or self.ref.get('istate', None) != int(istate):
            self.compute_reference(istate=istate)

        mu0 = self.M0_au
        e0  = self.omega_elec_au

        l_CART = self._l_cart_from_l_mwc(self.l_MWC, self.mass_vec_3N)  # (3N, Nvib)
        lam = np.asarray(self.evals_int_au, float)
        omega_vib = np.zeros_like(lam)
        pos = lam > 0.0
        omega_vib[pos] = np.sqrt(lam[pos])

        Nvib = l_CART.shape[1]

        # -------- Serial path (no MPI) --------
        if _SIZE == 1:
            dmu_dQ = np.zeros((Nvib, 3), dtype=float)
            for k in range(Nvib):
                if not pos[k]:
                    continue
                dmu_dQ[k, :] = self._finite_diff_for_mode(k, l_CART[:, k], dq_Q, e0, mu0, use_richardson)
        else:
            # -------- Parallel path (MPI): strided assignment of modes --------
            local_work = [k for k in range(_RANK, Nvib, _SIZE) if pos[k]]
            local_pairs = []
            for k in local_work:
                dmu = self._finite_diff_for_mode(k, l_CART[:, k], dq_Q, e0, mu0, use_richardson)
                local_pairs.append((k, dmu.astype(float, copy=False)))

            gathered = _COMM.gather(local_pairs, root=0)
            if _RANK == 0:
                dmu_dQ = np.zeros((Nvib, 3), dtype=float)
                for chunk in gathered:
                    for k, dmu in chunk:
                        dmu_dQ[k, :] = dmu
            else:
                dmu_dQ = None
            dmu_dQ = _COMM.bcast(dmu_dQ, root=0)

        dmu_dDelta = np.zeros_like(dmu_dQ)
        sel = omega_vib > 0
        dmu_dDelta[sel, :] = dmu_dQ[sel, :] / np.sqrt(omega_vib[sel])[:, None]

        self.ht = {
            'mu0_au': mu0,
            'mu0_D': mu0 * DEBYE_PER_AU,
            'omega_elec_au': e0,
            'omega_elec_eV': e0 * EV_PER_AU,
            'dmu_dQ_au': dmu_dQ,
            'dmu_dDelta_au': dmu_dDelta,
            'valid_mask': pos,
            'omega_vib_au': omega_vib,
            'meta': {
                'dq_Q': dq_Q,
                'use_richardson': bool(use_richardson),
                'unit': self.unit,
                'nstates': self.nstates,
                'grid_level': self.grid_level,
                'conv_tol': self.conv_tol,
                'tracker_sigma_e_eV': self.tracker_sigma_e_eV,
                'tracker_w_cos': self.tracker_w_cos,
                'istate_tracked': int(istate),
                'mpi_size': int(_SIZE),
                'mpi_rank': int(_RANK),
            }
        }
        self.dmu_dQ_au = dmu_dQ
        self.dmu_dDelta_au = dmu_dDelta
        self.valid_mask = pos
        self.omega_vib_au = omega_vib
        return self.ht

    # ---- one-mode finite difference (used by both serial and MPI paths) ----
    def _finite_diff_for_mode(self, k, lcart_col, dq_Q, e0, mu0, use_richardson):
        # central difference at ±dq
        R_plus  = self._displace_in_Q(self.geom_ref, lcart_col, +dq_Q, unit=self.unit)
        R_minus = self._displace_in_Q(self.geom_ref, lcart_col, -dq_Q, unit=self.unit)
        mu_p, _ = self._td_tdm_match(R_plus,  e0, mu0)
        mu_m, _ = self._td_tdm_match(R_minus, e0, mu0)
        D1 = (mu_p - mu_m) / (2.0 * dq_Q)
        if not use_richardson:
            return D1
        # Richardson refinement using half step
        h2 = dq_Q * 0.5
        R_p2 = self._displace_in_Q(self.geom_ref, lcart_col, +h2, unit=self.unit)
        R_m2 = self._displace_in_Q(self.geom_ref, lcart_col, -h2, unit=self.unit)
        mu_p2, _ = self._td_tdm_match(R_p2, e0, mu0)
        mu_m2, _ = self._td_tdm_match(R_m2, e0, mu0)
        D2 = (mu_p2 - mu_m2) / (2.0 * h2)
        return (4.0 * D2 - D1) / 3.0

    # ------------------- Handy getters (nice names) -------------------
    @property
    def M_HT_Q(self):
        return self.dmu_dQ_au

    @property
    def M_HT_Delta(self):
        return self.dmu_dDelta_au

    # --------------------------- Internals ----------------------------
    @staticmethod
    def _get_oscillator_strengths(td, e_au, M_au):
        f_osc = None
        if hasattr(td, 'oscillator_strength'):
            try:
                foscs = td.oscillator_strength(gauge='length')
                foscs = np.asarray(foscs)
                if foscs.ndim == 2 and foscs.shape[1] == 3:
                    f_osc = foscs.sum(axis=1)
                else:
                    f_osc = foscs
            except Exception:
                f_osc = None
        if f_osc is None:
            mu2 = np.sum(M_au**2, axis=1)
            f_osc = (2.0/3.0) * e_au * mu2
        return np.array(f_osc, dtype=float)

    def _td_tdm_match(self, geom_atomlist_or_array, ref_energy_au, ref_mu_au):
        Rarr = self._geom_to_array(geom_atomlist_or_array)
        mol1 = self.mol.copy()
        mol1.set_geom_(Rarr, unit=('Angstrom' if self.unit.lower().startswith('ang') else 'Bohr'))
        mf1 = dft.RKS(mol1)
        
        if getattr(self.mf, 'with_df', None) is not None:
            aux = getattr(self.mf.with_df, 'auxbasis', None)
            mf1 = mf1.density_fit(auxbasis=aux)
        
        mf1.xc = getattr(self.mf, 'xc', 'cam-b3lyp')
        mf1.grids.level = self.grid_level
        try:
            mf1.grids.prune = self.mf.grids.prune
        except Exception:
            pass
        mf1.conv_tol = self.conv_tol
        mf1.kernel()
        if not mf1.converged:
            raise RuntimeError("RKS did not converge at displaced geometry")
        td1 = tdscf.TDDFT(mf1)
        td1.nstates = max(1, int(self.nstates))
        td1.kernel()
        if td1.e is None or len(td1.e) < td1.nstates:
            raise RuntimeError("TDDFT failed or returned too few states at displaced geometry")
        e_au = np.array(td1.e, dtype=float)
        M_au = np.array(td1.transition_dipole(), dtype=float)
        idx = self._match_state_by_energy_and_dipole(ref_energy_au, ref_mu_au, e_au, M_au,
                                                     sigma_e_eV=self.tracker_sigma_e_eV,
                                                     w_cos=self.tracker_w_cos)
        return M_au[idx], e_au[idx]

    @staticmethod
    def _match_state_by_energy_and_dipole(ref_energy_au, ref_mu_au, cand_energies_au, cand_mus_au,
                                          sigma_e_eV=0.3, w_cos=0.85):
        ref_norm = np.linalg.norm(ref_mu_au) + 1e-16
        cand_norm = np.linalg.norm(cand_mus_au, axis=1) + 1e-16
        cos = (cand_mus_au @ ref_mu_au) / (cand_norm * ref_norm)
        cos2 = np.clip(cos, -1.0, 1.0)**2
        dE_eV = np.abs((cand_energies_au - ref_energy_au) * EV_PER_AU)
        gauss = np.exp(-(dE_eV / sigma_e_eV)**2)
        score = w_cos * cos2 + (1.0 - w_cos) * gauss
        return int(np.argmax(score))

    @staticmethod
    def _l_cart_from_l_mwc(l_MWC, mass_vec_3N):
        return np.asarray(l_MWC, float) / np.sqrt(np.asarray(mass_vec_3N, float))[:, None]

    @staticmethod
    def _geom_to_array(geom):
        if isinstance(geom, (list, tuple)) and len(geom) > 0 and isinstance(geom[0], (list, tuple)) and len(geom[0]) == 2:
            return np.array([p for _, p in geom], dtype=float)  # Å
        arr = np.asarray(geom, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("Geometry must be [('El',(x,y,z)), ...] or an (N,3) array")
        return arr

    def _displace_in_Q(self, geom0, l_cart_col, dq_Q, unit='Angstrom'):
        R0 = self._geom_to_array(geom0)
        dR_flat_bohr = dq_Q * np.asarray(l_cart_col, float)
        dR_flat = dR_flat_bohr * BOHR2ANG if str(unit).lower().startswith('ang') else dR_flat_bohr
        N = R0.shape[0]
        return (R0 + dR_flat.reshape(N, 3))
