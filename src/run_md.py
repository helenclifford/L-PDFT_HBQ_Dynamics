import os
import h5py
import numpy as np

from pyscf import gto, mcscf, mcpdft, M
from pyscf.csf_fci import csf_solver
from pyscf.lib import chkfile

from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Ha, Bohr, Debye, fs

seed = 93217

xyz = "enol.xyz"
chk_file = "hbq_44_ts.chk"

nactorbs = 4
nelect = 4
nstates = 2

# NVE parameters
n_steps = 1000
timestep_fs = 0.5


class pyscf_lpdft(Calculator):

    implemented_properties = ['energy', 'forces']

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='pyscf_lpdft', atoms=None, directory='.', **kwargs):

        super().__init__(restart=restart, ignore_bad_restart_file=ignore_bad_restart_file,
                         label=label, atoms=atoms, **kwargs)

        self.mol = gto.M(
                atom=atoms_from_ase(self.atoms),
                basis="6-31g**",
                verbose=5,
                output="pyscf.log",
                spin=0,
                unit="Angstrom",
                symmetry=False,
                charge=0)

        hf = self.mol.RHF().density_fit()   #Density fiting used here
        hf.run()

        mc = mcpdft.CASSCF(hf, "tPBE", nactorbs, nelect, grids_level=4)
        mc.fcisolver = csf_solver(self.mol, smult=1)
        mc = mc.multi_state([1.0/nstates,]*nstates, 'lin')
        mc.max_cycle = 2000
        mc.conv_tol = 1e-7
        mc.conv_tol_grad = 1e-4
        self.add_chkfile(mc)
        mc.kernel()
        self.scanner = mc.nuc_grad_method(state=1).as_scanner()    #S1 state gradients
        self.scanner.max_cycle = 2000


    def set_atoms(self, atoms):
        if self.atoms != atoms:
            self.atoms = atoms.copy()
            self.results = {}

    def add_chkfile(self, mc):
        mc.chkfile = "mcscf.chk"
        if os.path.isfile(chk_file):
            print("Loading in old_mcscf.chk")
            mo = chkfile.load(chk_file, "mcscf/mo_coeff")
            mo = mcscf.project_init_guess(mc, mo, prev_mol=None)
            mc.mo_coeff = mo
            
    def calculate(
            self,
            atoms=None,
            properties=['energy', 'forces'],
            system_changes=all_changes):

        super().calculate(atoms, properties, system_changes)

        self.set_atoms(atoms)
        self.mol.set_geom_(atoms_from_ase(self.atoms), unit="Angstrom")

        etot, grad = self.scanner(self.mol)

        if not self.scanner.converged:
            raise RuntimeError('Gradients did not converge')

        self.results['energy'] = etot * Ha
        forces = -1. * grad * (Ha / Bohr)
        totalforces = []
        totalforces.extend(forces)
        totalforces = np.array(totalforces)
        self.results['forces'] = totalforces
        

def main():
    print("Starting...")

    atoms = read(xyz)

    atoms.calc = pyscf_lpdft(atoms=atoms)

    MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=np.random.default_rng(seed))
    atoms.set_momenta(atoms.get_momenta())

    # run md
    dyn = VelocityVerlet(atoms, timestep=timestep_fs * fs,
                         trajectory='md.traj', logfile='md.log')
    dyn.run(n_steps)

    print("Finished...")


if __name__ == "__main__":
    main()
