#enol form, optimize S0 geometry 

import numpy as np
from pyscf import gto, scf, mcscf, mcpdft, lib
from pyscf.tools import molden
from pyscf.geomopt.geometric_solver import optimize
from mrh.my_pyscf.fci import csf_solver

mol = gto.Mole()
mol.atom = '''
  C      -3.162035      0.173145      0.184902
  C      -3.257035     -1.220309      0.125175
  C      -2.129891     -2.013658      0.025310
  C      -1.905898      0.781462      0.143234
  C      -0.733137     -0.021058      0.040266
  C      -0.853673     -1.422737     -0.018372
  C       0.336679     -2.226721     -0.122411
  C       1.570977     -1.666726     -0.165425
  C       1.731961     -0.242455     -0.108141
  C       0.586264      0.560503     -0.006488
  N       0.706064      1.935409      0.050874
  C       1.892597      2.533777      0.012156
  C       2.990310      0.399495     -0.148882
  C       3.072350      1.773552     -0.089514
  O      -1.860785      2.134352      0.203904
  H      -4.075756      0.773800      0.263873
  H      -4.246090     -1.691652      0.158288
  H       1.873813      3.629853      0.064284
  H       4.029408      2.298463     -0.118849
  H       3.897746     -0.204757     -0.227978
  H       2.473702     -2.281214     -0.244574
  H       0.218172     -3.316160     -0.166502
  H      -2.223803     -3.104890     -0.020337
  H      -0.931939      2.418526      0.165206
'''

mol.basis = "6-31g**"

mol.verbose = 4
mol.build()

nstates = 2
nelect = 4
nactorbs = 4

mf = scf.RHF(mol).density_fit()
mf.max_cycle = 1
mf.run()

mc = mcpdft.CASSCF(mf, 'tPBE', nactorbs, nelect, grids_level = 4)
mc.fcisolver = csf_solver(mol, smult=1)
mc = mc.multi_state([1.0/nstates,]*nstates, 'lin')

mo = lib.chkfile.load('hbq_natorb.chk', 'mcscf/mo_coeff')       # load guess for active space orbitals
mc.mo_coeff = mo
mc.conv_tol = 1e-8
mc.conv_tol_grad = 1e-4
mc.max_cycle_macro = 400
mc.max_cycle = 1000

mc.kernel()

#optimize geometry relative to ground state 
mol_eq = mc.Gradients(state=0).optimizer().kernel()    #optimize geometry relative to ground state 
print(mol_eq.tostring())
print(mol_eq.atom_coords(unit='Ang'))
