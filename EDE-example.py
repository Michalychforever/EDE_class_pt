import matplotlib
from cycler import cycler
matplotlib.rcParams['axes.prop_cycle'] = cycler(color=['#2424f0','#df6f0e','#3cc03c','#d62728','#b467bd','#ac866b','#e397d9','#9f9f9f','#ecdd72','#77becf'])
matplotlib.use('pdf')
matplotlib.rc('font', family='serif', serif='cm10')
matplotlib.rc('text', usetex=True)
fontProperties = {'family':'sans-serif',
                  'weight' : 'normal', 'size' : 16}
import matplotlib.pyplot as plt
import numpy as np
from classy import Class
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from scipy.integrate import odeint
import math


font = {'size'   : 16, 'family':'STIXGeneral'}
axislabelfontsize='large'
matplotlib.rc('font', **font)
matplotlib.mathtext.rcParams['legend.fontsize']='medium'
plt.rcParams["figure.figsize"] = [8.0,6.0]



M = Class()
# Table I of 1908.06995, third column, best-fit values
# Note: f and m found by trial-and-error to give the best-fit fEDE=.12, zc=10^3.562=3647.
M.set({'f_scf': 3.98e+26, 'm_scf': 5.31e-28, 'thetai_scf': 2.83, 'A_s': 2.215e-09, 'n_s': 0.9889, '100*theta_s': 1.04152, 'omega_b': 0.02253, 'omega_cdm': 0.1306, 'm_ncdm': 0.06, 'tau_reio': 0.072})


#'non linear':can choose 'halofit' or 'HMCODE'
M.set({'non linear':'HMCODE','N_ncdm':1, 'N_ur':2.0328, 'Omega_Lambda':0.0, 'Omega_fld':0, 'Omega_scf':-1, 'n_scf':3, 'CC_scf':1, 'scf_parameters':'1, 1, 1, 1, 1, 0.0', 'scf_tuning_index':3, 'attractor_ic_scf':'no', 'output':'tCl pCl lCl mPk', 'lensing':'yes', 'l_max_scalars':2508, 'P_k_max_h/Mpc':20,'z_max_pk':4.})
M.compute()

print(M.Omega_m())


baM = M.get_background()

fEDE = M.fEDE()
z_c = M.z_c()

baH = baM['H [1/Mpc]']
baT = baM['conf. time [Mpc]']
baa = 1/(1 + baM['z'])
bV = baM['V_e_scf']
bpp = baM["phi'_scf"]
baCrit = baM['(.)rho_crit']
rho_scf = (bpp*bpp/(2*baa*baa) + bV)/3.


plt.figure(figsize=(10,10))
plt.rcParams.update({'font.size': 18})
plt.plot(baM['z'],rho_scf/baCrit,lw=2,c='k')
plt.axhline(fEDE,c='r',ls='--')
plt.axvline(z_c,c='r',ls='--')
plt.xscale('log')
plt.xlim([10,1e5])
plt.xlabel('z')
plt.ylabel(r'$f_{EDE}$')
plt.show()
#plt.savefig('fEDE-example.pdf')
