import numpy as np

class ShiftVectorCalculator:
    """
    Gaussian-style normal-coordinate shift vector (K) between a fixed ground-state
    geometry and variable excited-state geometries.

    Initialization:
      - Preferred: from a Freq object (fr) via ShiftVectorCalculator.from_freq(fr)
      - Or: provide R_ground (atom list), masses_amu, mass_vec_3N, l_MWC, evals_int_au

    Embedded conventions:
      - unit: 'Angstrom'
      - enforce_TR_projection: True
      - lmw_orthonormalize: True
      - linear_tol: 1e-10
    """

    # ---- embedded constants ----
    BOHR2ANG = 0.529177210903      # Å per Bohr
    ENFORCE_TR = True
    ORTHONORMALIZE_LMW = True
    LINEAR_TOL = 1e-10

    # ---------- convenience constructors ----------
    @classmethod
    def from_freq(cls, fr):
        """
        Build from your Part 1–4 Freq object.
        Expects:
          fr.atom_list             -> GS geometry as atom list
          fr.masses_amu            -> (N,)
          fr.mass_vec_3N           -> (3N,)
          fr.l_MWC                 -> (3N, Nvib)
          fr.evals_int_au          -> (Nvib,)
        """
        return cls(
            R_ground_atomlist=fr.atom_list,
            masses_amu=fr.masses_amu,
            mass_vec_3N=fr.mass_vec,
            l_MWC=fr.l_MWC,
            evals_int_au=fr.evals_int_au
        )

    # ---------- helpers for atom-list <-> array ----------
    @staticmethod
    def _atomlist_to_array(atom_list):
        symbols = [a for a, _ in atom_list]
        R = np.array([p for _, p in atom_list], dtype=float)
        return symbols, R

    @staticmethod
    def _array_to_atomlist(symbols, R):
        R = np.asarray(R, float)
        return [(sym, (float(x), float(y), float(z))) for sym, (x, y, z) in zip(symbols, R)]

    # ---------- core init ----------
    def __init__(self,
                 R_ground_atomlist=None,
                 masses_amu=None,
                 mass_vec_3N=None,
                 l_MWC=None,
                 evals_int_au=None):
        if any(x is None for x in (R_ground_atomlist, masses_amu, mass_vec_3N, l_MWC, evals_int_au)):
            raise ValueError("Provide all inputs or use ShiftVectorCalculator.from_freq(fr).")

        self.gs_symbols, self.Rg = self._atomlist_to_array(R_ground_atomlist)  # Å
        self.masses_amu = np.asarray(masses_amu, float)
        self.mass_vec_3N = np.asarray(mass_vec_3N, float)
        self.l_MWC_raw = np.asarray(l_MWC, float)
        self.evals_int_au = np.asarray(evals_int_au, float)

        # Precompute TR projector (GS-only) and orthonormalized modes
        self.Q_TR = self._build_TR_basis_massweighted(self.Rg, self.masses_amu, self.LINEAR_TOL) \
                    if self.ENFORCE_TR else None
        self.l_MWC = self._orthonormalize_modes_if_needed(self.l_MWC_raw) \
                    if self.ORTHONORMALIZE_LMW else self.l_MWC_raw

        lam = self.evals_int_au
        self.valid_mask = lam > 0.0
        self.omega = np.zeros_like(lam)
        self.omega[self.valid_mask] = np.sqrt(lam[self.valid_mask])

    # ---- public API ----
    def compute(self, R_excited_atomlist):
        """
        Compute shift outputs for a new ES geometry (same atom order as GS).
        Input/Output geometry format is atom list.
        """
        es_symbols, Re_in = self._atomlist_to_array(R_excited_atomlist)
        # sanity: ensure atom ordering & symbols match
        if es_symbols != self.gs_symbols:
            raise ValueError("Atom symbols/order differ between GS and ES geometries.")

        # 1) MW Kabsch (Eckart) ES -> GS
        Re_aligned_arr, U, com_g, com_e = self._mass_weighted_kabsch_align(self.Rg, Re_in, self.masses_amu)

        # 2) Cartesian internal difference (Å) -> flatten (3N,)
        dR = (Re_aligned_arr - self.Rg).reshape(-1)

        # 3) Convert to Bohr (modes/Hessian in a.u.)
        dR_bohr = dR / self.BOHR2ANG  # since external unit is Å

        # 4) Mass-weight to MW Cartesian
        q_cart = np.sqrt(self.mass_vec_3N) * dR_bohr

        # 5) Project out TR (robust Eckart)
        q_int = self._project_out_TR(q_cart, self.Q_TR) if self.Q_TR is not None else q_cart

        # 6) Project onto internal modes
        K = self.l_MWC.T @ q_int  # (Nvib,)

        # 7) Dimensionless Δ and α
        Delta = np.zeros_like(self.evals_int_au)
        Delta[self.valid_mask] = np.sqrt(self.omega[self.valid_mask]) * K[self.valid_mask]
        alpha = Delta / np.sqrt(2.0)

        # build aligned ES atom list for convenience
        Re_aligned_atomlist = self._array_to_atomlist(self.gs_symbols, Re_aligned_arr)

        return {
            'R_excited_aligned_atomlist': Re_aligned_atomlist,
            'rotation_U': U,
            'com_ground': com_g,
            'com_excited': com_e,
            'delta_R_cart_ang_flat': dR,   # Å flattened
            'q_massweighted': q_cart,
            'q_internal': q_int,
            'K_shift': K,
            'Delta_dimless': Delta,
            'alpha': alpha,
            'valid_mask': self.valid_mask.copy(),
        }

    # ---- utilities ----
    @staticmethod
    def _center_of_mass(R, masses_amu):
        m = np.asarray(masses_amu, float)
        return (m[:, None] * R).sum(axis=0) / m.sum()

    @classmethod
    def _mass_weighted_kabsch_align(cls, R_ref, R_mov, masses_amu):
        m = np.asarray(masses_amu, float)
        X = np.asarray(R_ref, float); Y = np.asarray(R_mov, float)
        comX = cls._center_of_mass(X, m); comY = cls._center_of_mass(Y, m)
        Xc = X - comX; Yc = Y - comY
        w = np.sqrt(m)[:, None]
        C = (Yc * w).T @ (Xc * w)
        V, S, Wt = np.linalg.svd(C)
        d = np.sign(np.linalg.det(V @ Wt))
        D = np.diag([1.0, 1.0, d])
        U = V @ D @ Wt
        Y_aligned = (Yc @ U) + comX
        return Y_aligned, U, comX, comY

    @classmethod
    def _build_TR_basis_massweighted(cls, R_ref, masses_amu, lin_tol=1e-10):
        R = np.asarray(R_ref, float); m = np.asarray(masses_amu, float)
        N = R.shape[0]
        r_com = cls._center_of_mass(R, m)
        r = R - r_com
        s = np.sqrt(m)
        # translations
        Tx = np.zeros(3*N); Ty = np.zeros(3*N); Tz = np.zeros(3*N)
        for i in range(N):
            Tx[3*i+0] = s[i]; Ty[3*i+1] = s[i]; Tz[3*i+2] = s[i]
        vecs = [Tx, Ty, Tz]
        # rotations about x,y,z
        axes = np.eye(3)
        for k in range(3):
            Rk = np.zeros(3*N)
            for i in range(N):
                Rk[3*i:3*i+3] = s[i] * np.cross(axes[k], r[i])
            if np.linalg.norm(Rk) > lin_tol:
                vecs.append(Rk)
        D = np.column_stack(vecs)              # (3N, 6) or (3N, 5) if linear
        Q, Rmat = np.linalg.qr(D, mode='reduced')
        good = np.abs(np.diag(Rmat)) > lin_tol
        return Q[:, good]

    @staticmethod
    def _project_out_TR(q_massweighted, Q_TR):
        return q_massweighted - Q_TR @ (Q_TR.T @ q_massweighted)

    @staticmethod
    def _orthonormalize_modes_if_needed(l_MWC, ortho_tol=1e-8):
        L = np.asarray(l_MWC, float)
        G = L.T @ L
        if np.linalg.norm(G - np.eye(G.shape[0])) < ortho_tol:
            return L
        Q, _ = np.linalg.qr(L)
        return Q
