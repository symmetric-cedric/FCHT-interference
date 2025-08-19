#!/usr/bin/env python3
import os, sys, time, inspect
import numpy as np
from pathlib import Path

# -------- threads from scheduler (optional; your .sh can set these too) --------
if "PBS_NCPUS" in os.environ and "OMP_NUM_THREADS" not in os.environ:
    t = os.environ["PBS_NCPUS"]
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, t)

# -------- try MPI (for parallel HT and rank-0 printing) --------
try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except Exception:
    COMM = None
    RANK = 0
    SIZE = 1

# -------- your repo path --------
sys.path.append("/rds/general/user/lcp21/home/FCHT_Interference/Python_Codes/Classes")

from Ground_State import SCFRunner

# Prefer your parallel class if present; otherwise fall back to MPI/serial Freq
FreqClass = None
mode_hint = None
try:
    from Freq_Parallel import FreqParallel as FreqClass
    mode_hint = "workers"
except Exception:
    try:
        from Freq_Parallel import Freq as FreqClass  # if your file exposes Freq
        mode_hint = "mpi/serial"
    except Exception:
        from freq_mpi import Freq as FreqClass       # from the script I sent earlier
        mode_hint = "mpi/serial"

# ------------------ test geometry (NH3) ------------------
GS_atom_list = [
    ('N', (0.0, 0.0, -0.10000001)),
    ('H', (0.0, -0.942809, 0.23333324)),
    ('H', (-0.81649655, 0.4714045, 0.23333324)),
    ('H', (0.81649655, 0.4714045, 0.23333324)),
]

# ------------------ SCF ------------------
# (We run it on every rank for simplicity; PySCF objects aren't trivially broadcastable.)
scf_t0 = time.time()
Ammonia_GS = SCFRunner(GS_atom_list)
scf_t1 = time.time()

# ------------------ instantiate Freq class safely ------------------
def build_and_run_freq(mf):
    params = inspect.signature(FreqClass.__init__).parameters
    # derive worker count from env, else 1
    default_workers = int(os.environ.get("MAX_WORKERS",
                             os.environ.get("PBS_NCPUS",
                             os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
    kwargs = {}
    used_key = None
    for key in ("max_workers", "n_workers", "workers", "num_workers"):
        if key in params:
            kwargs[key] = default_workers
            used_key = key
            break
    start = time.time()
    fr_obj = FreqClass(mf, **kwargs).run()
    end = time.time()
    return fr_obj, end - start, used_key

fr, wall_freq, used_key = build_and_run_freq(Ammonia_GS.mf)

# ------------------ Displace Geometry (utility; not essential to HT) ------------------
def displace_geometry(Frequency, mode_index, magnitude):
    """Displace geometry along a single normal mode."""
    atom_list = Frequency.atom_list
    cart_modes = Frequency.cartesian_displacements_modes
    mode_disp = cart_modes[mode_index]  # list[(sym,(dx,dy,dz))]
    displaced = []
    for (sym, (x, y, z)), (_, (dx, dy, dz)) in zip(atom_list, mode_disp):
        displaced.append((sym, (x + magnitude * dx,
                                y + magnitude * dy,
                                z + magnitude * dz)))
    return displaced

# Example: displace along mode 3 with magnitude 0.5 (not used later, but kept for your testing)
geom_disp = displace_geometry(fr, mode_index=0, magnitude=0)

Ammonia_ES = SCFRunner(geom_disp)


# ------------------ Electronic Dipole Moments  ------------------
from Electronic_Dipole_Moments_Parallel import TDDFTHT_Parallel  # ensure this is the MPI-enabled version

ht_t0 = time.time()
tdht = TDDFTHT_Parallel.from_freq(Ammonia_ES.mf, fr, istate=0, nstates=6, dq_Q=0.03, use_richardson=True)
ht_t1 = time.time()

# ------------------ Shift Vector  ------------------

from Shift_Vector import ShiftVectorCalculator

  
svc = ShiftVectorCalculator.from_freq(fr)

R_excited_atomlist = geom_disp

res = svc.compute(R_excited_atomlist)

# ------------------ Overlap  ------------------

from Overlap_Integral import Overlap_Integrals

Delta = res["Delta_dimless"]
mask  = res["valid_mask"]

fc = Overlap_Integrals.from_Delta(Delta, valid_mask=mask)

S00, top = fc.top_transitions(vmax_per_mode=10, k=12)

# ------------------ Combine Result  ------------------

# Helper: per-mode FC overlaps <0|n_i> and q-overlaps <0|Δ_i|n_i>
def get_S_and_q_per_mode(fc_obj, n_tuple):
    """
    Returns:
      S_i : (Nv,) array with <0|n_i(shift)>
      q_i : (Nv,) array with <0|Δ_i|n_i(shift)> = (1/sqrt(2)) * O[1,n]
    """
    n_tuple = tuple(int(x) for x in n_tuple)
    Nv = fc_obj.Nv
    S_i = np.zeros(Nv, dtype=float)
    q_i = np.zeros(Nv, dtype=float)
    for i, ni in enumerate(n_tuple):
        O = fc_obj.one_mode_table(i, vmax_m=1, vmax_n=int(ni))  # shape (2, ni+1)
        S_i[i] = O[0, ni]
        q_i[i] = (1.0 / np.sqrt(2.0)) * O[1, ni]
    return S_i, q_i

def prod_except_index(vals):
    """prod of all entries except index i, returned as array."""
    vals = np.asarray(vals, float)
    Nv = vals.size
    left = np.ones(Nv, float)
    right = np.ones(Nv, float)
    for i in range(1, Nv):
        left[i] = left[i-1] * vals[i-1]
        right[-i-1] = right[-i] * vals[-i]
    return left * right

# Make sure tdht vectors line up with fc modes (fc removed invalid modes)
valid_mask = np.asarray(res["valid_mask"], bool)  # from ShiftVectorCalculator
M_FC = np.asarray(tdht.M0_au, float)              # (3,)
M_HT_all = np.asarray(tdht.dmu_dDelta_au, float)  # (Nv_total,3)
M_HT = M_HT_all[valid_mask, :]                    # (Nv,3) to match fc.Nv

# Build records for top-K FC transitions, then compute FC/HT/total
records = []
for n_tuple, S_fc_prod, _I_fc in top:
    # per-mode overlaps
    S_i, q_i = get_S_and_q_per_mode(fc, n_tuple)      # each shape (Nv,)

    # FC piece
    S_overall = float(np.prod(S_i))
    TDM_FC = M_FC * S_overall                          # (3,)

    # HT piece: sum_i M_HT[i] * q_i * prod_{j≠i} S_j
    P_except = prod_except_index(S_i)                  # (Nv,)
    TDM_HT = np.sum(M_HT * (q_i * P_except)[:, None], axis=0)

    # Intensities
    I_FC = float(np.dot(TDM_FC, TDM_FC))
    I_HT = float(np.dot(TDM_HT, TDM_HT))
    TDM_tot = TDM_FC + TDM_HT
    I_tot = float(np.dot(TDM_tot, TDM_tot))

    records.append({
        "n_tuple": tuple(int(x) for x in n_tuple),
        "I_tot": I_tot,
        "I_FC": I_FC,
        "I_HT": I_HT,
        "M_FC": M_FC.copy(),             # 3-vector (M0)
        "overall_S_FC": S_overall,       # scalar
        "individual_S_FC": S_i.copy(),   # list/array of scalars
        "M_HT": M_HT.copy(),             # (Nv,3) list of 3-vectors
        "q_overlaps": q_i.copy(),        # (Nv,)
        "TDM_FC_vec": TDM_FC.copy(),
        "TDM_HT_vec": TDM_HT.copy(),
        "TDM_tot_vec": TDM_tot.copy(),
    })

# Optionally: re-sort by total intensity (not just FC)
records.sort(key=lambda r: r["I_tot"], reverse=True)

# ------------------ write results (rank 0 only) -----------------
if RANK == 0:
    def vec_str(v, prec=6):
        return np.array2string(np.asarray(v, float), precision=prec, suppress_small=True)

    def prod_str(vals, prec=6):
        vals = np.asarray(vals, float)
        if vals.size == 0:
            return "1.0"
        parts = [f"{x:.6e}" for x in vals]
        return " * ".join(parts)

    out_path = Path("M1S0.txt").resolve()
    with open(out_path, "w") as f:
        # --- header & timings ---
        f.write("MPI ranks: {}\n".format(SIZE))
        f.write("Parallel mode (Freq): {}\n".format(mode_hint))
        if used_key:
            workers = os.environ.get(
                "MAX_WORKERS",
                os.environ.get("PBS_NCPUS", os.environ.get("SLURM_CPUS_PER_TASK", "1"))
            )
            f.write("Workers kwarg used: {}={}\n".format(used_key, workers))
        else:
            f.write("Workers kwarg used: none (class takes none)\n")
        f.write("SCF time: {:.2f} s\n".format(scf_t1 - scf_t0))
        f.write("Freq time: {:.2f} s\n".format(wall_freq))
        f.write("TDDFT+HT time: {:.2f} s\n\n".format(ht_t1 - ht_t0))

        # --- system info ---
        f.write("Number of atoms: {}\n".format(len(GS_atom_list)))
        f.write("Number of vibrational modes (valid): {}\n\n".format(fc.Nv))

        # --- frequencies ---
        f.write("Frequencies (cm^-1):\n")
        f.write(" ".join("{:.4f}".format(val) for val in fr.freqs_cm1) + "\n\n")

        # --- shift vectors (Delta) ---
        f.write("Shift Vectors (dimensionless Delta):\n")
        f.write(vec_str(Delta) + "\n\n")

        # --- M0 ---
        f.write("M0 (electronic dipole, a.u.):\n")
        f.write(vec_str(M_FC) + "\n\n")

        # --- individual global arrays (from the first record, purely for listing) ---
        if len(records) > 0:
            f.write("Individual FC integral along each normal coordinate (S_i):\n")
            f.write(vec_str(records[0]["individual_S_FC"]) + "\n\n")

            f.write("Individual dmu/dDelta along each normal coordinate (a.u.):\n")
            f.write(vec_str(M_HT) + "\n\n")

            f.write("Individual HT integral along each normal coordinate (q_i):\n")
            f.write(vec_str(records[0]["q_overlaps"]) + "\n\n")

            S_i_first = np.asarray(records[0]["individual_S_FC"], float)
            S_over_first = float(np.prod(S_i_first)) if S_i_first.size else 1.0
            f.write("Overall FC_Integral = Product(S_i) = {} = {:.6e}\n\n".format(
                prod_str(S_i_first), S_over_first
            ))
        else:
            f.write("No transitions available.\n\n")

        # --- Top transitions with numeric equations ---
        f.write("Top transitions by total intensity (FC+HT):\n")
        for rec in records[:12]:
            n_tuple = rec["n_tuple"]
            S_i = np.asarray(rec["individual_S_FC"], float)    # per-mode <0|n_i>
            q_i = np.asarray(rec["q_overlaps"], float)         # per-mode <0|Q_i|n_i>
            M0 = np.asarray(rec["M_FC"], float)                # 3-vector
            MHT = np.asarray(rec["M_HT"], float)               # (Nv,3)
            S_over = float(rec["overall_S_FC"])

            # Precompute product over all modes except i
            P_except = prod_except_index(S_i)                  # (Nv,)

            # FC vector and intensity
            TDM_FC = M0 * S_over
            I_FC = float(np.dot(TDM_FC, TDM_FC))

            f.write("n = {}\n".format(n_tuple))

            # Overall FC product for this transition
            f.write("  Overall FC_Integral = Product(S_i) = {} = {:.6e}\n".format(
                prod_str(S_i), S_over
            ))

            # TDM_FC numeric build
            f.write("  TDM_FC = M0 * Product(S_i) = {} * {:.6e} = {}\n".format(
                vec_str(M0), S_over, vec_str(TDM_FC)
            ))
            f.write("  I_FC = norm(TDM_FC)^2 = norm({})^2 = {:.6e}\n\n".format(
                vec_str(TDM_FC), I_FC
            ))

            # HT contributions per mode
            f.write("  Individual HT contributions per mode:\n")
            ht_terms = []
            Nv_local = len(S_i)
            for i in range(Nv_local):
                scalar_i = float(q_i[i] * P_except[i])
                term_vec = MHT[i] * scalar_i
                ht_terms.append(term_vec)

                # Show numeric equation for this mode:
                #   scalar_i = q_i * Product(S_j, j!=i)
                #   contrib_i = scalar_i * dM/dQ_i
                prod_wo_i = np.delete(S_i, i)
                f.write("    Mode {}:\n".format(i))
                f.write("      Product(S_j, j!=i) = {}\n".format(prod_str(prod_wo_i)))
                f.write("      scalar_i = q_i * Product(S_j, j!=i) = {:.6e} * {} = {:.6e}\n".format(
                    q_i[i], prod_str(prod_wo_i), scalar_i
                ))
                f.write("      dM/dQ_i = {}\n".format(vec_str(MHT[i])))
                f.write("      contrib_i = scalar_i * dM/dQ_i = {:.6e} * {} = {}\n".format(
                    scalar_i, vec_str(MHT[i]), vec_str(term_vec)
                ))

            # Sum of HT terms
            if ht_terms:
                TDM_HT = np.sum(ht_terms, axis=0)
            else:
                TDM_HT = np.zeros(3, float)
            I_HT = float(np.dot(TDM_HT, TDM_HT))

            f.write("\n")
            f.write("  TDM_HT (sum over i) = " + " + ".join(vec_str(v) for v in ht_terms) +
                    " = {}\n".format(vec_str(TDM_HT)))
            f.write("  I_HT = norm(TDM_HT)^2 = norm({})^2 = {:.6e}\n\n".format(
                vec_str(TDM_HT), I_HT
            ))

            # Total
            TDM_tot = TDM_FC + TDM_HT
            I_tot = float(np.dot(TDM_tot, TDM_tot))

            f.write("  TDM_tot = TDM_FC + TDM_HT = {} + {} = {}\n".format(
                vec_str(TDM_FC), vec_str(TDM_HT), vec_str(TDM_tot)
            ))
            f.write("  I_FCHT = norm(TDM_tot)^2 = norm({})^2 = {:.6e}\n\n".format(
                vec_str(TDM_tot), I_tot
            ))

    print("Results written to {}".format(out_path))
