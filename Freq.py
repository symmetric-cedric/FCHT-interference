# Freq.py  — numerical-Hessian-only vibrational analysis (external mf version)

import numpy as np
from pyscf import dft, grad
import scipy.linalg

# ---- Physical constants (CODATA 2018/2019) ----
HARTREE2J = 4.3597447222071e-18      # J
BOHR2M    = 5.29177210903e-11        # m
AMU2KG    = 1.66053906660e-27        # kg
CLIGHT    = 299_792_458.0            # m/s
BOHR2ANG  = 0.529177210903           # Å / Bohr


class Freq:
    """
    Vibrational analysis (numerical Hessian only) assuming you already ran SCF/DFT.

    You create the molecule and run RKS **outside** this class, then pass the converged
    PySCF mean-field object `mf` here. This does not change the math/logic; we reuse `mf`
    to evaluate gradients at displaced geometries for the finite-difference Hessian.

    Parameters
    ----------
    mf : PySCF SCF object
        Converged mean-field (e.g., dft.RKS(mol) after mf.kernel()).
    atom_list : list[tuple[str, (float,float,float)]], optional
        Coordinates for formatting/inertia in the chosen `unit`. If not given, we
        reconstruct from `mf.mol` using `_atom_list_from_mol`.
    unit : str, optional
        "Angstrom" or "Bohr" label for I/O/printing only. Defaults to mol.unit.
    fd_step_angstrom : float
        Central-difference step size in Å (converted to Bohr internally if needed).
    compute_force_constants : bool
        If True, compute SI force constants (N/m) from frequencies and reduced masses.
    """

    def __init__(self, mf):
        # External inputs
        self.mf = mf
        self.mol = mf.mol
        self.unit = 'Angstrom'
        self.atom_list = self._atom_list_from_mol(self.mol, self.unit)

        # User knobs
        self.fd_step_angstrom = 0.005
        self.compute_force_constants = False

        # Placeholders
        self.masses_amu = None
        self.mass_vec = None
        self.H_cart = None
        self.H_mw = None
        self.H_int = None
        self.eigvals_mw = None
        self.eigvecs_mw = None
        self.freqs_cm1_mw = None
        self.COM = None
        self.I_tensor = None
        self.I_principal = None
        self.X_axes = None
        self.rotor_type = None
        self.linear_axis_index = None
        self.D_trans_list = None
        self.D_rot_list = None
        self.Q_TR = None
        self.D_inter = None
        self.evals_int_au = None
        self.L_internal = None
        self.freqs_cm1 = None
        self.l_MWC = None
        self.l_CART = None
        self.l_CART_norm = None
        self.reduced_masses_amu = None
        self.force_constants_N_per_m = None
        self.cartesian_displacements_modes = None

    # -------------------- public driver --------------------
    def run(self):
        # Numerical Hessian using provided mf
        step_bohr = self._angstrom_to_bohr(self.fd_step_angstrom) if str(self.unit).lower().startswith('ang') else self.fd_step_angstrom
        self.H_cart = self._hessian_cart_numerical(self.mf, step_bohr=step_bohr)

        # Mass-weight and preliminary diagonalization
        self.H_mw, self.mass_vec = self._mass_weight_hessian(self.H_cart, self.mol)
        self.eigvals_mw, self.eigvecs_mw, self.freqs_cm1_mw = self._diagonalize_hessian_prelim(self.H_mw)

        # Inertia / principal axes (Part 2)
        self.masses_amu = np.array(self.mol.atom_mass_list())
        part2 = self._part2_inertia_pipeline(self.atom_list, self.masses_amu,
                                             unit_label=("Å" if str(self.unit).lower().startswith('ang') else "Bohr"),
                                             verbose=False)
        self.COM = part2["COM"]
        self.I_tensor = part2["I_tensor"]
        self.I_principal = part2["I_principal"]
        self.X_axes = part2["X_axes"]
        self.rotor_type = part2["rotor_type"]
        self.linear_axis_index = part2["linear_axis_index"]

        # TR and internal bases (Part 3)
        part3 = self._part3_build_TR_and_internal(part2, self.masses_amu)
        self.D_trans_list = part3["D_trans_list"]
        self.D_rot_list = part3["D_rot_list"]
        self.Q_TR = part3["Q_TR"]
        self.D_inter = part3["D_inter"]

        # Internal projection & modes (Part 4)
        part4 = self._part4_internal_modes(self.H_mw, self.D_inter, self.mass_vec,
                                           compute_force_constants=self.compute_force_constants,
                                           sort_by_frequency=True)
        self.H_int = part4["H_int"]
        self.evals_int_au = part4["evals_int_au"]
        self.L_internal = part4["L_internal"]
        self.freqs_cm1 = part4["freqs_cm1"]
        self.l_MWC = part4["l_MWC"]
        self.l_CART = part4["l_CART"]
        self.l_CART_norm = part4["l_CART_norm"]
        self.reduced_masses_amu = part4["reduced_masses_amu"]
        self.force_constants_N_per_m = part4["force_constants_N_per_m"]

        # Per-mode Cartesian displacements in input format
        self.cartesian_displacements_modes = self._format_displacements(self.l_CART)
        return self

    # -------------------- convenience --------------------
    def _format_displacements(self, l_CART):
        if l_CART is None:
            return None
        N = len(self.atom_list)
        disp_per_mode = []
        for j in range(l_CART.shape[1]):
            mode = []
            for i, (sym, _) in enumerate(self.atom_list):
                dx, dy, dz = l_CART[3*i:3*i+3, j]
                mode.append((sym, (float(dx), float(dy), float(dz))))
            disp_per_mode.append(mode)
        return disp_per_mode

    # -------------------- building blocks --------------------
    @staticmethod
    def _atom_list_from_mol(mol, unit_label):
        coords = mol.atom_coords(unit=('Angstrom' if str(unit_label).lower().startswith('ang') else 'Bohr'))
        atoms = []
        for i in range(mol.natm):
            atoms.append((mol.atom_symbol(i), (float(coords[i,0]), float(coords[i,1]), float(coords[i,2]))))
        return atoms

    @staticmethod
    def _grad_at_coords(mf, coords_bohr):
        mol0 = mf.mol
        mol1 = mol0.copy()
        mol1.set_geom_(coords_bohr, unit='Bohr')
        mf1 = dft.RKS(mol1)
        mf1.xc = mf.xc
        mf1.grids.level = mf.grids.level
        mf1.grids.prune = mf.grids.prune
        mf1.conv_tol = mf.conv_tol
        mf1.kernel()
        g = grad.rks.Gradients(mf1).kernel()  # (natm,3) Eh/Bohr
        return g

    def _hessian_cart_numerical(self, mf, step_bohr=0.01, logger_obj=None):
        mol = mf.mol
        natm = mol.natm
        n3 = 3 * natm
        coords = mol.atom_coords(unit='Bohr').copy()
        H = np.zeros((n3, n3))

        def idx_to_atom_comp(i):
            return i // 3, i % 3

        for i in range(n3):
            A, comp = idx_to_atom_comp(i)
            coords_p = coords.copy()
            coords_p[A, comp] += step_bohr
            g_p = self._grad_at_coords(mf, coords_p).reshape(-1)
            coords_m = coords.copy()
            coords_m[A, comp] -= step_bohr
            g_m = self._grad_at_coords(mf, coords_m).reshape(-1)
            H[:, i] = (g_p - g_m) / (2.0 * step_bohr)
        return 0.5 * (H + H.T)

    @staticmethod
    def _mass_weight_hessian(h_cart, mol):
        masses_amu = np.array(mol.atom_mass_list())  # (N,)
        mass_vec = np.repeat(masses_amu, 3)          # (3N,)
        M_sqrt = np.sqrt(np.outer(mass_vec, mass_vec))
        H_mw = h_cart / M_sqrt
        return 0.5 * (H_mw + H_mw.T), mass_vec

    @staticmethod
    def _eigvals_to_wavenumbers(eigvals):
        au2hz = np.sqrt(HARTREE2J / (AMU2KG * BOHR2M**2)) / (2.0 * np.pi)
        return np.sqrt(np.abs(eigvals)) * (au2hz / CLIGHT) * 1e-2  # cm^-1

    def _diagonalize_hessian_prelim(self, H_mw, zero_thresh=1e-8, imag_thresh=0.0):
        eigvals, eigvecs = scipy.linalg.eigh(H_mw)
        freqs = self._eigvals_to_wavenumbers(eigvals)
        signs = np.ones_like(freqs)
        signs[eigvals < -max(imag_thresh, 0.0)] = -1.0
        freqs_signed = freqs * signs
        freqs_signed[np.abs(eigvals) < zero_thresh] = 0.0
        return eigvals, eigvecs, freqs_signed

    # ---------- inertia & principal axes ----------
    @staticmethod
    def _center_of_mass(positions, masses_amu):
        return (masses_amu[:, None] * positions).sum(axis=0) / masses_amu.sum()

    def _shift_to_com(self, atom_list, masses_amu):
        positions = np.array([pos for _, pos in atom_list], dtype=float)
        com = self._center_of_mass(positions, masses_amu)
        pos_com = positions - com
        shifted_atom_list = [(sym, tuple(p)) for (sym, _), p in zip(atom_list, pos_com)]
        return com, pos_com, shifted_atom_list

    @staticmethod
    def _inertia_tensor(positions_com, masses_amu):
        r2 = np.einsum('ij,ij->i', positions_com, positions_com)
        s  = np.dot(masses_amu, r2)
        rrT = np.einsum('ni,nj->nij', positions_com, positions_com)
        I = s * np.eye(3) - np.einsum('n,nij->ij', masses_amu, rrT)
        return 0.5 * (I + I.T)

    @staticmethod
    def _principal_axes(I, linear_tol=1e-8):
        vals, vecs = np.linalg.eigh(I)
        order = np.argsort(vals)
        Ivals = vals[order]
        X = vecs[:, order]
        Imax = max(Ivals.max(), 1.0)
        small = Ivals < (linear_tol * Imax)
        nz = np.count_nonzero(small)
        if np.allclose(I, 0, atol=linear_tol * Imax):
            rotor_type, linear_axis = 'atom', None
        elif nz >= 1:
            rotor_type, linear_axis = 'linear', int(np.argmin(Ivals))
        else:
            rotor_type, linear_axis = 'nonlinear', None
        return Ivals, X, rotor_type, linear_axis

    def _part2_inertia_pipeline(self, atom_list, masses_amu, *, verbose=False, unit_label="Å", linear_tol=1e-8):
        com, pos_com, shifted_atoms = self._shift_to_com(atom_list, masses_amu)
        I = self._inertia_tensor(pos_com, masses_amu)
        Ivals, X, rotor_type, linear_axis = self._principal_axes(I, linear_tol=linear_tol)
        pos_in_principal = pos_com @ X
        return {
            "COM": com,
            "positions_COM": pos_com,
            "shifted_atom_list": shifted_atoms,
            "I_tensor": I,
            "I_principal": Ivals,
            "X_axes": X,
            "rotor_type": rotor_type,
            "linear_axis_index": linear_axis,
            "positions_in_principal": pos_in_principal,
        }

    # ---------- TR and internal bases ----------
    @staticmethod
    def _build_D_trans(masses_amu):
        N = len(masses_amu)
        s = np.sqrt(masses_amu)
        Tx = np.zeros(3*N); Ty = np.zeros(3*N); Tz = np.zeros(3*N)
        for i in range(N):
            Tx[3*i+0] = s[i]
            Ty[3*i+1] = s[i]
            Tz[3*i+2] = s[i]
        return [Tx, Ty, Tz]

    @staticmethod
    def _build_D_rot(X_axes, positions_COM, masses_amu, linear_axis_index=None, drop_tol=1e-12):
        N = positions_COM.shape[0]
        s = np.sqrt(masses_amu)
        D_list = []
        for k in range(3):
            if linear_axis_index is not None and k == linear_axis_index:
                continue
            omega = X_axes[:, k]
            Rk = np.zeros(3*N)
            for i in range(N):
                cross = np.cross(omega, positions_COM[i])
                Rk[3*i:3*i+3] = s[i] * cross
            if np.linalg.norm(Rk) > drop_tol:
                D_list.append(Rk)
        return D_list

    @staticmethod
    def _orthonormalize_TR(D_trans_list, D_rot_list, lin_indep_tol=1e-10):
        Dcols = [*D_trans_list, *D_rot_list]
        D = np.column_stack(Dcols)
        keep = [j for j in range(D.shape[1]) if np.linalg.norm(D[:, j]) > lin_indep_tol]
        D = D[:, keep] if len(keep) < D.shape[1] else D
        Q, R = np.linalg.qr(D, mode='reduced')
        diag = np.abs(np.diag(R)) if R.ndim == 2 else np.array([])
        if diag.size:
            good = diag > lin_indep_tol
            Q = Q[:, good]
        return Q

    @staticmethod
    def _build_internal_basis(Q_TR, n3):
        U, S, Vt = np.linalg.svd(Q_TR.T, full_matrices=True)
        tol = max(Q_TR.shape) * np.finfo(float).eps * (S[0] if S.size else 1.0)
        r = np.sum(S > tol) if S.size else 0
        D_inter = Vt[r:].T if r < Vt.shape[0] else np.zeros((n3, 0))
        if D_inter.size:
            Q_int, _ = np.linalg.qr(D_inter)
            return Q_int
        else:
            return D_inter

    def _part3_build_TR_and_internal(self, part2, masses_amu):
        posCOM = part2["positions_COM"]
        X = part2["X_axes"]
        rotor_type = part2["rotor_type"]
        lin_axis = part2["linear_axis_index"]
        N = posCOM.shape[0]
        n3 = 3 * N

        D_trans = self._build_D_trans(masses_amu)
        D_rot = self._build_D_rot(X, posCOM, masses_amu, linear_axis_index=lin_axis)
        Q_TR = self._orthonormalize_TR(D_trans, D_rot)

        n_tr_expected = 3 if rotor_type == 'atom' else (5 if rotor_type == 'linear' else 6)
        if Q_TR.shape[1] != n_tr_expected:
            Q, R, piv = scipy.linalg.qr(np.column_stack([*D_trans, *D_rot]), pivoting=True, mode='economic')
            diag = np.abs(np.diag(R))
            keep = diag > (1e-10)
            Q_TR = Q[:, keep]
            if Q_TR.shape[1] != n_tr_expected:
                raise RuntimeError(
                    f"External subspace has {Q_TR.shape[1]} vectors; expected {n_tr_expected}. "
                    "Geometry may be near-linear or masses/coordinates inconsistent."
                )
        D_inter = self._build_internal_basis(Q_TR, n3)
        return {"D_trans_list": D_trans, "D_rot_list": D_rot, "Q_TR": Q_TR, "D_inter": D_inter}

    # ---------- internal projection / modes ----------
    @staticmethod
    def _project_hessian_to_internal(H_mw, D_inter):
        return D_inter.T @ H_mw @ D_inter

    def _diagonalize_internal(self, H_int, imag_thresh=0.0, zero_thresh=1e-10):
        evals, L = scipy.linalg.eigh(H_int)
        freqs = self._eigvals_to_wavenumbers(evals)
        signs = np.ones_like(freqs)
        signs[evals < -max(imag_thresh, 0.0)] = -1.0
        freqs_signed = freqs * signs
        freqs_signed[np.abs(evals) < zero_thresh] = 0.0
        return evals, L, freqs_signed

    @staticmethod
    def _backtransform_modes(D_inter, L, mass_vec):
        l_MWC = D_inter @ L
        inv_sqrt_m = 1.0 / np.sqrt(mass_vec)
        l_CART = l_MWC * inv_sqrt_m[:, None]
        return l_MWC, l_CART

    @staticmethod
    def _normalize_cartesian_modes_and_reduced_masses(l_CART):
        ss = np.sum(l_CART**2, axis=0)
        eps = np.finfo(float).eps
        ss = np.maximum(ss, eps)
        red_masses = 1.0 / ss
        N = 1.0 / np.sqrt(ss)
        l_CART_norm = l_CART * N
        return l_CART_norm, red_masses

    @staticmethod
    def _force_constants_SI(freqs_cm1, red_masses_amu):
        mu_kg = red_masses_amu * AMU2KG
        nu_m_inv = freqs_cm1 * 100.0
        omega = 2.0 * np.pi * CLIGHT * nu_m_inv
        return mu_kg * omega**2

    def _part4_internal_modes(self, H_mw, D_inter, mass_vec, *, compute_force_constants=False, sort_by_frequency=True):
        H_int = self._project_hessian_to_internal(H_mw, D_inter)
        evals, L, freqs_cm1 = self._diagonalize_internal(H_int)
        l_MWC, l_CART = self._backtransform_modes(D_inter, L, mass_vec)
        l_CART_norm, red_masses_amu = self._normalize_cartesian_modes_and_reduced_masses(l_CART)
        k_SI = self._force_constants_SI(freqs_cm1, red_masses_amu) if compute_force_constants else None

        if sort_by_frequency:
            order = np.argsort(freqs_cm1)
            evals = evals[order]
            freqs_cm1 = freqs_cm1[order]
            L = L[:, order]
            l_MWC = l_MWC[:, order]
            l_CART = l_CART[:, order]
            l_CART_norm = l_CART_norm[:, order]
            red_masses_amu = red_masses_amu[order]
            if k_SI is not None:
                k_SI = k_SI[order]

        return {
            "H_int": H_int,
            "evals_int_au": evals,
            "freqs_cm1": freqs_cm1,
            "L_internal": L,
            "l_MWC": l_MWC,
            "l_CART": l_CART,
            "l_CART_norm": l_CART_norm,
            "reduced_masses_amu": red_masses_amu,
            "force_constants_N_per_m": k_SI,
        }

    # -------------------- tiny helpers --------------------
    @staticmethod
    def _angstrom_to_bohr(x_ang):
        return x_ang / BOHR2ANG

