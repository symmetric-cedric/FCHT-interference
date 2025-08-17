from pyscf import gto, dft

class SCFRunner:
    """
    Extremely simple class: takes an atom list and outputs mol and mf.
    Basis, unit, xc, etc. are hard-coded.
    """

    def __init__(self, atom_list):
        self.atom_list = atom_list
        self.mol = self.build_mol()   # call the method via self
        self.mf  = self.run_rks()

    def build_mol(self):
        mol = gto.Mole()
        mol.atom = self.atom_list
        mol.unit = 'Angstrom'
        mol.basis = 'aug-cc-pVDZ'
        mol.charge = 0
        mol.spin = 0
        mol.build()
        self.mol = mol
        return mol

    def run_rks(self):
        if self.mol is None:
            self.build_mol()
        mf = dft.RKS(self.mol)
        mf.xc = 'cam-b3lyp'
        mf.grids.level = 4
        mf.grids.prune = None
        mf.conv_tol = 1e-10
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("RKS did not converge")
        self.mf = mf
        return mf

