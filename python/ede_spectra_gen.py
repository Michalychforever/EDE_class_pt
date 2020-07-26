from classy import Class 
import numpy as np
import matplotlib.pyplot as plt
from numpy import log, exp
import sys
from time import time 

# z = 0.61
# z = 0.38

cosmo = Class ()
cosmo.set({'k_pivot':'0.05',
'ln10^{10}A_s':'3.067',	
# 'A_s':'2.215e-9',
'n_s':'0.9829',
# 'YHe':0.25,
'tau_reio':'0.0552',
# 'T_cmb':'2.726',
'h':'0.71',
'omega_b':'0.02262',
'N_ncdm':'1',
'm_ncdm':'0.06',
'N_ur':'2.0328',
'omega_cdm':'0.12955',
'P_k_max_h/Mpc': '100.',
# 'output':'mPk,tCl',
'lensing':'yes',
'l_max_scalars':'3000',
'output':'mPk,tCl,lCl,pCl',
'z_pk':'0.38,0.61',
'non linear':' SPT ',
'IR resummation':' Yes ',
'Bias tracers':' Yes ',
'RSD':' Yes ',
'AP':'Yes',
'Omega_Lambda':0.0, 
'Omega_fld':0, 
'Omega_scf':-1, 
'fEDE':0.101,
'log10z_c':3.56,
'thetai_scf':2.78,
'attractor_ic_scf':'no',
'scf_parameters': '1, 1, 1, 1, 1, 0.0',
'n_scf':3,
'CC_scf':1,
'scf_tuning_index':3,
'Omfid':0.31,
'SigmaFOG':0.})
t1 = time()
cosmo.compute() 


h = cosmo.h()
# Da =cosmo.angular_distance(z)
# print("Da=",Da)
# fz = cosmo.scale_independent_growth_factor_f(z)
# print("fz=",fz)


kmax = 0.25
kmin = 0.01
d0=np.loadtxt('/Users/michalychforever/Dropbox/SDSS_RSD/mean_spec_ngc_z3.dat')
k0=d0[:,0]
P0=d0[:,1]
P2=d0[:,2]
k_size1 = len(k0)

omit = 0
count = 0
for i in range(k_size1):
    if (k0[i] <= kmax and k0[i] >= kmin):
        count = count + 1
    if k0[i] < kmin:
        omit = omit + 1
k_size = count
print("k_size=",k_size)
print("omit=",omit)

k = np.zeros(k_size)
P2noW = np.zeros(k_size)
P0noW = np.zeros(k_size)
for i in range(k_size):
	k[i] = k0[i+omit]


chunk = [0,1,2,3]
zs = [0.61, 0.61, 0.38, 0.38] 

b1ar = [1.8123395200344528, 1.9711643689791116, 1.7976849068792673, 1.8195391895221746]
b2ar = [-3.389038516264213, 0.710113288319272, -2.0348992182906898, 0.23185932639227247]
bG2ar = [0.7123168452164335,-0.0805024264784932, -0.059485567428447315, -0.07465249324055871]
css0ar = [-21.323675253905517, 24.23120503080546, -31.5739719535399, 27.343551631976293]
css2ar = [-47.06658475569365, 69.21226868799967, 18.148849100604917, 85.34120183788738]
Pshotar = [6709.17682313595, 109.9363346039784, 95.42485522945113, 293.92835441647276]
bGamma3ar = [0., 0., 0., 0.]
b4ar = [256.82199304134315, 78.62142674299282, 374.08451421691973, -146.54664459312085]

norm =1.
# b1 = 1.909433
# b2 = -2.357092
# bG2 = 3.818261e-01
# css0 = -2.911944e+01
# css2 = -1.235181e+01
# Pshot = 2.032084e+03
# bGamma3 = 0.
# b4 = 1.924983e+02

print('S8=',cosmo.sigma8()*(cosmo.Omega_m()/0.3)**0.5)

kmsMpc = 3.33564095198145e-6
rd=cosmo.rs_drag()
print('rd=',rd)

for j in range(len(chunk)):
	z = zs[j]
	b1 = b1ar[j]
	b2 = b2ar[j]
	bG2 = bG2ar[j]
	css0 = css0ar[j]
	css2 = css2ar[j]
	Pshot = Pshotar[j]
	bGamma3 = bGamma3ar[j]
	b4 = b4ar[j]
	fz = cosmo.scale_independent_growth_factor_f(z)
	da=cosmo.angular_distance(z)
	# print('DA=',da)
	hz=cosmo.Hubble(z)/kmsMpc
	# print('Hz=',hz)
	DV = (z*(1+z)**2*da**2/cosmo.Hubble(z))**(1./3.)
	# print('DV=',DV)
	fs8 = cosmo.scale_independent_growth_factor(z)*cosmo.scale_independent_growth_factor_f(z)*cosmo.sigma8()
	# print('fs8=',fs8)
	# print('sigma8=',cosmo.sigma8())
	print('rd/DV=',rd/DV)
	print('rdH=',rd*cosmo.Hubble(z))
	print('rd/DA=',rd/da)


	P2noW = np.zeros(k_size)
	P0noW = np.zeros(k_size)
	for i in range(k_size):
		kinloop1 = k[i] * h
		P2noW[i] = (norm**2.*cosmo.pk(kinloop1, z)[18] +norm**4.*(cosmo.pk(kinloop1, z)[24])+ norm**1.*b1*cosmo.pk(kinloop1, z)[19] +norm**3.*b1*(cosmo.pk(kinloop1, z)[25]) + b1**2.*norm**2.*cosmo.pk(kinloop1, z)[26] +b1*b2*norm**2.*cosmo.pk(kinloop1, z)[34]+ b2*norm**3.*cosmo.pk(kinloop1, z)[35] + b1*bG2*norm**2.*cosmo.pk(kinloop1, z)[36]+ bG2*norm**3.*cosmo.pk(kinloop1, z)[37]  + 2.*(css2)*norm**2.*cosmo.pk(kinloop1, z)[12]/h**2. + (2.*bG2+0.8*bGamma3)*norm**3.*cosmo.pk(kinloop1, z)[9])*h**3.  + fz**2.*b4*k[i]**2.*((norm**2.*fz**2.*70. + 165.*fz*b1*norm+99.*b1**2.)*4./693.)*(35./8.)*cosmo.pk(kinloop1,z)[13]*h
		P0noW[i] = (norm**2.*cosmo.pk(kinloop1, z)[15] +norm**4.*(cosmo.pk(kinloop1, z)[21])+ norm**1.*b1*cosmo.pk(kinloop1, z)[16] +norm**3.*b1*(cosmo.pk(kinloop1, z)[22]) + norm**0.*b1**2.*cosmo.pk(kinloop1, z)[17] +norm**2.*b1**2.*cosmo.pk(kinloop1, z)[23] + 0.25*norm**2.*b2**2.*cosmo.pk(kinloop1, z)[1] +b1*b2*norm**2.*cosmo.pk(kinloop1, z)[30]+ b2*norm**3.*cosmo.pk(kinloop1, z)[31] + b1*bG2*norm**2.*cosmo.pk(kinloop1, z)[32]+ bG2*norm**3.*cosmo.pk(kinloop1, z)[33] + b2*bG2*norm**2.*cosmo.pk(kinloop1, z)[4]+ bG2**2.*norm**2.*cosmo.pk(kinloop1, z)[5] + 2.*css0*norm**2.*cosmo.pk(kinloop1, z)[11]/h**2. + (2.*bG2+0.8*bGamma3)*norm**2.*(b1*cosmo.pk(kinloop1, z)[7]+norm*cosmo.pk(kinloop1, z)[8]))*h**3.+Pshot + fz**2.*b4*k[i]**2.*(norm**2.*fz**2./9. + 2.*fz*b1*norm/7. + b1**2./5)*(35./8.)*cosmo.pk(kinloop1,z)[13]*h
	np.savetxt('pk_mock_ede_chunk_'+str(chunk[j])+'.dat', np.column_stack((k,P0noW,P2noW)))

t2 = time()
print("overall elapsed time=",t2-t1)
### everything is in units Mpc! 

#l = np.array(range(2,2501))
#factor = l*(l+1)/(2*np.pi)
#raw_cl = cosmo.raw_cl(2500)
#lensed_cl = cosmo.lensed_cl(2500)
#raw_cl.viewkeys()

#z = 0.2
#raw_pk = np.zeros(len(k))
#for i in range(len(k)):
#    raw_pk[i] = k[i]*cosmo.pk(k[i],z)

#cosmo.set({'P_k_max_h/Mpc': '100.','output':'mPk','z_pk':'0.2','non linear':' SPT ','IR resummation':' Yes ','Bias tracers':' No '})
#cosmo.compute()

#z = 0.2
#pk_lin = np.zeros(len(k))
#for i in range(len(k)):
#    pk_lin[i] = float(cosmo.pk(k[i],z))

#plt.loglog(k,raw_pk)
#plt.xlabel(r"$k$")

#plt.ylabel(r"$P(k)$")
#plt.tight_layout()
#plt.savefig("misha_test.pdf")

