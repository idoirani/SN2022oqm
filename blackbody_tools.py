import numpy as np
from astropy import table
from astropy.io import ascii
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy import constants 
import os
import sys
UTILS_PATH=os.environ['UTILS_PATH']
sys.path.insert(1,UTILS_PATH)
from extinction_maayane import *
from PhotoUtils import *
from extinction import ccm89,calzetti00, apply
from scipy.optimize  import minimize,curve_fit
sys.path.insert(1,UTILS_PATH+'/Barak')
#from barak.extinction import SMC_Gordon03,LMC_Gordon03
sys.path.append('/home/idoi/Dropbox/Objects/ZTF infant sample/analysis') 
sys.path.insert(1,'./')
from params import * 
import dynesty
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc


from Ni_tools import *
import tqdm
import pandas as pd
## priors 
rad_high=3e15
rad_low=4e12
temp_high=5e5
temp_low=5000

# constants 
c=constants.c.value*1e10
h=constants.h.value*1e23
k_B=constants.k_B.value*1e23
sigma_sb=constants.sigma_sb.cgs.value
c_cgs = constants.c.cgs.value
LAW = 'MW'

eVs = constants.h.to('eV s').value

path_mat = '/home/idoi/Dropbox/Objects/ZTF infant sample/Multigroup simulations/RSG_SED_batch1.mat'
path_key  =  '/home/idoi/Dropbox/Objects/ZTF infant sample/Multigroup simulations/RSG_SED_batch1_key.mat'




def interp_data(time,data,datcol='flux'):
		data_interp=np.interp(time,data['real_time'],data[datcol])
		return data_interp


def plot_error_photo(filter,color,marker):
		filter_photometry=data[data['filter']==filter]
		plt.errorbar(filter_photometry['real_time'],filter_photometry['flux'],filter_photometry['fluxerr'],marker=marker,color=color)


def bb_F(lam,T,r,EBV=0,EBV_mw=0, R_v=3.1,z=0,LAW=LAW):
	A_v=R_v*EBV
	A_v_mw=3.1*EBV_mw
	
	flux=1.191/(lam**(5))/(np.exp((1.9864/1.38064852)/(lam*T))-1)*(np.pi*(r)**2) 
	#flux=apply(ccm89(lam*1e4, A_v, R_v), flux)
	#flux=apply(ccm89(lam*1e4*(1+z), A_v_mw, 3.1), flux)
	flux=apply_extinction(1e4*lam,flux,EBV=EBV,EBV_mw=EBV_mw,R_v=R_v,z=z,LAW=LAW)
	return flux  

def apply_extinction(lam,flam,EBV=0,EBV_mw=0, R_v=3.1,z=0,LAW='MW'):
	if EBV>0:
		if LAW == 'SMC':
			ex = SMC_Gordon03(lam*(1+z))
			ex.set_EBmV(EBV)
			A_v = ex.Av
			Alam  = A_v*ex.AlamAv 
			#import ipdb; ipdb.set_trace()
			flux = flam*10**(-0.4*Alam/0.87)
		elif LAW == 'LMC':
			ex = LMC_Gordon03(lam*(1+z))
			ex.set_EBmV(EBV)
			A_v = ex.Av
			Alam  = A_v*ex.AlamAv 
			flux = flam*10**(-0.4*Alam/1.08)
		elif LAW == 'Cal00':
			A_v=R_v*EBV
			flux=apply(calzetti00(lam, A_v, R_v,unit='aa'), flam)
			
		else:
			A_v=R_v*EBV
			flux=apply(ccm89(lam, A_v, R_v,unit='aa'), flam)

	else:
		flux = flam
	if EBV_mw>0:
		A_v_mw=3.1*EBV_mw
		flux=apply(ccm89(lam*(1+z), A_v_mw, 3.1,unit='aa'), flux)
	return flux  


#def bb_fit(lam,f_lam,lims,f_lam_err=None,include_errors=True,EBV=0,Ebv_MW=0,R_v=3.1,z=0):
#    func= lambda lam,T,r: bb_F(lam,T,r,EBV=EBV,EBV_mw=Ebv_MW, R_v=R_v, z=z)
#
#    if include_errors:  
#        popt, pcov = curve_fit(func, lam, f_lam, sigma=f_lam_err,bounds=lims,method='trf')
#
#    else:
#        popt, pcov = curve_fit(func, lam, f_lam,bounds=lims,method='trf')    
#
#    perr = np.sqrt(np.diag(pcov))
#
#    return popt, perr


def bb_fit(lam,f_lam,lims,f_lam_err=None,include_errors=True,EBV=0,Ebv_MW=0,R_v=3.1,z=0,ret_cov = False):
	func= lambda lam,T,r: bb_F(lam,T,r,EBV=EBV,EBV_mw=Ebv_MW, R_v=R_v, z=z)

	if include_errors:  
		popt, pcov = curve_fit(func, lam, f_lam, sigma=f_lam_err,bounds=lims,method='trf')

	else:
		popt, pcov = curve_fit(func, lam, f_lam,bounds=lims,method='trf')    

	perr = np.sqrt(np.diag(pcov))
	if ret_cov:
		return popt, perr,pcov
	else: 
		return popt, perr





	



def generate_bb_mag(T_array,filt,r = 1e14, d = 3.08e26, Rv=3.1, EBV = 0,EBV_MW = 0, LAW = 'MW'):
	v = 67.4*d/3.08e24
	z = v/300000
	Trans  = filter_transmission[filt]
	lam  = np.linspace(1000,20000,1901)
	
	m_array = []
	for i,T in enumerate(T_array):
		flux = bb_F(lam*1e-4,T*1e-4,(r/d)*1e10,EBV=EBV,EBV_mw = EBV_MW,z = z, R_v = Rv, LAW = LAW)*1e-13                                                                  
		m = SynPhot(lam,flux,Trans)
		m_array.append(m) 
	m_array = np.array(m_array) 
	return m_array 



from numba import jit, njit 

@njit
def fast_trapz(y,x):
	trap = np.trapz(y,x)
	return trap

@njit
def fast_interp(x,xp,yp):
	interp = np.interp(x,xp,yp)
	return interp

def SynPhot_fast_AB(Lambda,Flux,Filter_lam,Filter_T, sys='AB',ret_flux=False):
	c_cgs = constants.c.cgs.value
	cAA  = c_cgs*1e8
	ZPflux=3631
	ZPflux=ZPflux*1e-23
	ZPflux_vec=ZPflux*cAA*(Lambda**(-2))
	T_filt_interp=np.interp(Lambda,Filter_lam,Filter_T,left=0,right=0)
	trans_flux=fast_trapz(Flux*Lambda*T_filt_interp,Lambda)
	norm_flux=fast_trapz(ZPflux_vec*T_filt_interp*Lambda,Lambda)
	mag=-2.5*np.log10(trans_flux/norm_flux)
	return mag  


def SynPhot_fast(Lambda,Flux,Filter, sys='AB',ret_flux=False):
	if sys.lower()=='vega':
		ZPflux_vec=fast_interp(Lambda,vega_spec[:,0] ,vega_spec[:,1] ,left=0,right=0)
	if sys.lower()=='ab':
		ZPflux=3631
		ZPflux=ZPflux*1e-23
		ZPflux_vec=ZPflux*cAA*(Lambda**(-2))

	Filter["col1"] = Filter["col1"].astype(float)
	T_filt_interp=np.interp(Lambda,Filter["col1"],Filter["col2"],left=0,right=0)

	trans_flux=fast_trapz(Flux*Lambda*T_filt_interp,Lambda)
	norm_flux=fast_trapz(ZPflux_vec*T_filt_interp*Lambda,Lambda)
	if norm_flux == 0:
		import ipdb; ipdb.set_trace()
	mag=-2.5*np.log10(trans_flux/norm_flux)
	if sys.lower()=='vega':
		mag=-2.5*np.log10(trans_flux/norm_flux)+0.03
	if ret_flux:
		return trans_flux/norm_flux 
	return mag  


eVs = constants.h.to('eV s').value
c_cgs = constants.c.cgs.value
k_B_eVK = constants.k_B.to('eV/K').value
k_B = constants.k_B.cgs.value
sigma_b  = constants.sigma_sb.cgs.value

cAA  = c_cgs*1e8
h = constants.h.cgs.value
@njit
def B_nu(nu_eV,T_eV):
	nu_hz = nu_eV/eVs
	B_nu = (2*h*(nu_hz)**3/c_cgs**2)*(np.exp(nu_eV/T_eV)-1)**(-1)
	return B_nu




@njit
def L_BB(tday, L_break,t_break_days,t_tr_day):
	t_tilda = tday/t_break_days #same units
	L = (t_tilda**(-4/3)+t_tilda**(-0.172)*f_corr_SW(tday,t_tr_day))
	L = L_break*L
	return L

@njit
def T_color_bb(tday, T_break,t_break_days):
	t_tilda = tday/t_break_days #same units
	T = T_break*np.minimum(0.97*t_tilda**(-1/3), t_tilda**(-0.45))
	return  T


@njit
def f_corr_SW(tday,t_tr_day):
	A = 0.9
	alpha = 0.5
	a = 2
	f_corr = A*np.exp(-(a*tday/t_tr_day)**alpha) 
	return f_corr

@njit
def break_params_bo(R13, beta_bo,rho_bo9,k34,Menv):
	v_bo_9 = beta_bo*29979245800.0/1e9
	t_trans = 0.85*R13/v_bo_9 #hrs
	L_trans = 3.98e42*R13**(2/3)*v_bo_9**(5/3)*rho_bo9**(-1/3)*k34**(-4/3) #erg/s
	T_trans = 8.35*R13**(-1/3)*v_bo_9**(0.463)*rho_bo9**(-0.083)*k34**(-1/3) #eV
	t_trasnp = 18.9*np.sqrt(Menv)*(R13*rho_bo9)**(0.14)*k34**(0.64)*(v_bo_9)**(-0.36) #days
	return t_trans , T_trans, L_trans,t_trasnp

@njit
def break_params_new(R13, v85,fM,Menv,k34):
	t_trans = 0.86*R13**1.26*v85**(-1.13)*(fM*k34)**(-0.13)        #hrs
	L_trans = 3.69e42*R13**(0.78)*v85**(2.11)*fM**0.11*k34**(-0.89) #erg/s
	T_trans = 8.19*R13**(-0.32)*v85**(0.58)*fM**(0.03)*k34**(-0.22) #eV
	t_trasnp =19.5*np.sqrt(Menv*k34/v85) #days
	return t_trans , T_trans, L_trans,t_trasnp

def bo_params_from_phys(fM,vstar85,R13,k034=1):
	beta_bo = 0.033039*fM**0.129356*vstar85**1.12936*k034**0.129356/R13**0.258713
	rho_bo9 = 1.1997*fM**0.322386/R13**1.64477/vstar85**0.677614/k034**0.677614
	return beta_bo,rho_bo9


def phys_params_from_bo(rho_bo9,beta_bo,R13,k034=1,n=3/2,beta1 = 0.1909):
	rho_bo = rho_bo9*1e-9
	kappa = 0.345*k034
	Rstar = R13*1e13
	vbo = beta_bo*29979245800
	vsstar = vbo*(rho_bo*kappa*beta_bo*Rstar/(n+1))**(-beta1*n)
	fM = (4*np.pi/3)*rho_bo*(rho_bo*kappa*beta_bo/(n+1))**n*(Rstar)**(n+3)
	v85 = vsstar/10**8.5
	fM = fM/constants.M_sun.cgs.value
	return v85,fM


@njit
def validity(R13, v85,rho_bo9,k34,Menv):
	t_trans_hrs , T_trans_eV, L_trans_ergs,t_trasnp = break_params_new(R13, v85,rho_bo9,k34,Menv)
	t07eV = 2.67*t_trans_hrs*(T_trans_eV/5)**2.2 #days
	t_up = min(t07eV,t_trasnp) #days
	t_down = 0.2*t_trans_hrs**(-0.1)*(L_trans_ergs/10**41.5)**0.55*(T_trans_eV/5)**(-2.21) #hrs
	t_down = t_down/24 #days
	return t_down,t_up
	
@njit
def validity2(R13, v85,fM,k34,Menv):
	t07eV = 6.86*R13**(0.56)*v85**(0.16)*k34**(-0.61)*fM**(-0.06)  #days
	t_trasnp  = 19.5*np.sqrt(Menv*k34/v85)
	t_up = min(t07eV,t_trasnp/2) #days
	t_down = 17*R13/60 #hrs
	t_down = t_down/24 #days
	return t_down,t_up

@njit
def L_nu_gray(tday, nu_eV,R13,v85,fM,Menv,k34):
	t_trans , T_trans, L_trans,t_tr_day =break_params_new(R13, v85,fM,Menv,k34)
	L_break_41_5erg = L_trans/10**41.5
	T_break_5eV = T_trans/5
	t_break_days = t_trans/24
	L_bb = L_BB(tday, L_break_41_5erg,t_break_days,t_tr_day)*10**(41.5)
	T_col =  T_color_bb(tday, T_break_5eV,t_break_days)*5
	L_nubb = L_bb*(np.pi)*B_nu(nu_eV,T_col)/(sigma_sb*(T_col*eV2K)**4)
	return  L_nubb


@njit
def f_nu_gray(tday, lam_AA,R13,v85,fM,k34,Menv,d = 3.08e26):
	nu_hz = c_cgs/(lam_AA/1e8)
	nu_eV = nu_hz*eVs
	L = L_nu_gray(tday, nu_eV,R13,v85,fM,Menv,k34)
	f_nu = L/(4*np.pi*d**2)
	f_lam = f_nu2f_lam(f_nu,lam_AA)
	return f_lam


def MW_F_gray(tday,lam,R13,v85,fM,k34,Menv,d = 3.08e26, EBV=0,EBV_mw=0, R_v=3.1,z=0,LAW='MW'):
	f_lam = np.array(list(map(lambda l: f_nu_gray(tday,l,R13,v85,fM,k34,Menv,d = d), lam )))
	flux_corr = apply_extinction(lam,f_lam,EBV=EBV,EBV_mw=EBV_mw,R_v=R_v,z=z,LAW=LAW)
	return flux_corr
						
   
def generate_MW_mag(time,R13,v85,fM,k34,Menv,filt_list,z=0, d = 3.08e26, Rv=3.1, EBV = 0,EBV_MW = 0, LAW = 'MW',H0 = 67.4,UV_sup = False,reduced = False):
	func = MW_F_gray
	v = H0*d/3.08e24
	z = v/300000
	lam  = np.linspace(1000,10000,90)
	m_array = {}
	for filt in filt_list:
		mm = []
		Trans  = filter_transmission[filt]
		for t in time:
			flux = func(t,lam,R13,v85,fM,k34,Menv,d = d, EBV=EBV,EBV_mw=EBV_MW, R_v=Rv,z=z,LAW=LAW)
			m = SynPhot(lam,flux,Trans)
			mm.append(m)	
		m_array[filt] = np.array(mm)
	return m_array 

def generate_MW_mag_single(time,R13,v85,fM,k34,Menv,filt,z=0, d = 3.08e26, Rv=3.1, EBV = 0,EBV_MW = 0, LAW = 'MW',H0 = 67.4,UV_sup = False,reduced=True):
	func = MW_F_gray

	
	v = H0*d/3.08e24
	z = v/300000
	Trans  = filter_transmission_fast[filt]
	Trans_l = Trans[:,0]
	Trans_T = Trans[:,1]
	lam  = np.linspace(np.min(Trans_l),np.max(Trans_l),20)
	m_array = {}
	mm = []
	for t in time:
		flux = func(t,lam,R13,v85,fM,k34,Menv,d = d, EBV=EBV,EBV_mw=EBV_MW, R_v=Rv,z=z,LAW=LAW)
		m = SynPhot_fast_AB(lam,flux,Trans_l,Trans_T)
		mm.append(m)
	m_array[filt] = np.array(mm)    	
	return m_array

	
def generate_MW_flux(time,lam,R13,v85,fM,k34,Menv, d = 3.08e26, Rv=3.1, EBV = 0,EBV_MW = 0, LAW = 'MW',H0 = 67.4,UV_sup = False,old_form = False,reduced = False,gray = False):
	func = MW_F_gray
	
	v = H0*d/3.08e24
	z = v/300000
	f_array = {}
	mm = []
	if isinstance(time,(float, int)):
		time = np.array([time])
	
	f_array = np.zeros((len(time),len(lam)))
	for i in range(len(time)):
		flux = func(time[i],lam,R13,v85,fM,k34,Menv,d = d, EBV=EBV,EBV_mw=EBV_MW, R_v=Rv,z=z,LAW=LAW)
		f_array[i,:] = flux
	return f_array 


@njit
def f_nu2f_lam(f_nu,lam_AA):
	c_AA = 2.99792458e+18
	f_lam_AA=(c_AA/lam_AA**2)*f_nu
	return f_lam_AA


eV2K =1/constants.k_B.to('eV / K ')
eV2K = eV2K.value

@njit
def f_nu_bb(lam_AA,T,R,d = 3.08e26):
	lam_cm = (lam_AA/1e8)
	nu_hz = c_cgs/lam_cm
	nu_eV = nu_hz*eVs
	T_eV = T/eV2K
	L_bb = 4*np.pi*R**2
	L_nubb = L_bb*(np.pi)*B_nu(nu_eV,T_eV)
	f_nu = L_nubb/(4*np.pi*d**2)
	f_lam = f_nu2f_lam(f_nu,lam_AA)
	return f_lam,f_nu



class model_cooling(object):
	def __init__(self,T0, R0, alpha,beta,t0,distance = 3.08e26,ebv = 0,Rv = 3.1, LAW = 'MW'):
		self.ebv = ebv
		self.Rv = Rv
		self.distance = distance
		self.LAW  = LAW
		self.T0 = T0
		self.R0 = R0 
		self.alpha = alpha 
		self.beta = beta
		self. t0 = t0
	def mags(self,time,filt_list = FILTERS): 
		m_array={}
		for filt in filt_list:
			R_evo = self.R_evolution(time)
			T_evo = self.T_evolution(time)
			mag = generate_cooling_mag(T_evo, R_evo, filt,d = self.distance, Rv=self.Rv, EBV = self.ebv,EBV_MW = 0, LAW = self.LAW)
			m_array[filt] = mag
		return m_array
	def mags_single(self,time,filt): 
		m_array={}
		R_evo = self.R_evolution(time)
		T_evo = self.T_evolution(time)
		mag = generate_cooling_mag(T_evo, R_evo, filt,d = self.distance, Rv=self.Rv, EBV = self.ebv,EBV_MW = 0, LAW = self.LAW)
		m_array[filt] = mag
		return m_array
	def T_evolution(self,time): 
		time = time.flatten()
		alpha = self.alpha
		T0 = self.T0
		Tevo = T0*time**alpha
		return Tevo
	def R_evolution(self,time): 
		time = time.flatten()
		beta = self.beta
		R0 = self.R0
		Revo = R0*time**beta
		return Revo
	def L_evolution(self, time):
		time = time.flatten()
		R_evo = self.R_evolution(time)
		T_evo = self.T_evolution(time) 
		L_evo = sigma_sb*4*np.pi*(R_evo)**2*(T_evo)**4
		return L_evo 
	def likelihood(self,dat,sys_err = 0.05,nparams = 7):
		t0 = self.t0 
		ebv = self.ebv
		Rv = self.Rv
		LAW = self.LAW
		d = self.distance
		filt_list = np.unique(dat['filter'])
		chi2_dic = {}
		dof = 0
		chi_tot = 0
		N_band = []
		for filt in filt_list:
			dat_filt = dat[dat['filter']==filt]
			time = dat_filt['t_rest']-t0

			#R_evo = self.R_evolution(time)
			#T_evo = self.T_evolution(time)
			#mag = generate_cooling_mag_single(T_evo, R_evo, filt,d = d, Rv=Rv, EBV = ebv,EBV_MW = 0, LAW = LAW)
			mag = self.mags_single(time,filt)[filt]
			mag_obs = dat_filt['absmag']
			mag_err = dat_filt['AB_MAG_ERR'] + sys_err
			c2 = chi_sq(mag_obs,mag_err, mag)
			N = len(c2)
			#N_band.append(N)
			chi2_dic[filt] = c2#*N**2
			dof = dof+N
			chi_tot = chi_tot +np.sum(chi2_dic[filt])
		#N_band = np.array(N_band) 
		chi_tot = chi_tot#/np.sum(N_band**2)
		dof = dof - nparams
		
		rchi2 = chi_tot/dof
			
		if chi_tot == 0:
			logprob = -np.inf
		elif dof <= 0:
			logprob = -np.inf
		else:
			#prob = scipy.stats.chi2.pdf(chi_tot, dof)
			#prob = float(prob)
			logprob = log_chi_sq_pdf(chi_tot, dof)
		#logprob = np.log(prob)
		return logprob, chi_tot, dof

	def likelihood_bb(self,dat_bb,sys_err = 0.05,nparams = 10):
		t0 = self.t0 
		dof = 0
		chi_tot = 0
		
		time = dat_bb['t_rest']-t0
		R_evo = self.R_evolution(time)
		T_evo = self.T_evolution(time)
		err_R = np.sqrt((dat_bb['R_up']-dat_bb['R'])**2 + (sys_err*dat_bb['R'])**2)
		err_T = np.sqrt((dat_bb['T_up']-dat_bb['T'])**2 + (sys_err*dat_bb['T'])**2)

		c2_R = chi_sq(dat_bb['R'],err_R, R_evo)
		c2_T = chi_sq(dat_bb['T'],err_T, T_evo)

		N = len(c2_R) + len(c2_T)
		#N_band.append(N)
		chi_tot = np.sum(c2_R) +np.sum(c2_T) #*N**2
		dof =N
		dof = dof - nparams
		
		rchi2 = chi_tot/dof
			
		#if chi_tot == 0:
		#    logprob = -np.inf
		#elif dof <= 0:
		#    logprob = -np.inf
		#else:
		#    #prob = scipy.stats.chi2.pdf(chi_tot, dof)
		#    #prob = float(prob)
		#    logprob = log_chi_sq_pdf(chi_tot, dof)
		##logprob = np.log(prob)
		logprob = -chi_tot
		return logprob, chi_tot, dof


class model_cooling_broken(object):
	def __init__(self,T0, R0, alpha,beta,t0,t_br,t_br2,alpha2,beta2,distance = 3.08e26,ebv = 0,Rv = 3.1, LAW = 'MW'):
		self.ebv = ebv
		self.Rv = Rv
		self.distance = distance
		self.LAW  = LAW
		self.T0 = T0
		self.R0 = R0 
		self.alpha = alpha 
		self.beta = beta
		self.alpha2 = alpha2
		self.beta2 = beta2
		self. t_br = t_br
		self. t_br2 = t_br2
		self. t0 = t0
		self.T2 =  T0*(t_br)**(alpha)/((t_br)**(alpha2))
		self.R2 =  R0*(t_br2)**(beta)/((t_br2)**(beta2))


	def T_evolution(self,time): 
		time = time.flatten()
		alpha = self.alpha
		alpha2 = self.alpha2

		T0 = self.T0
		T2 = self.T2

		Tevo = T0*time**alpha
		Tevo2 = T2*time**alpha2
		T = np.maximum(Tevo,Tevo2)
		return T
	def R_evolution(self,time): 
		time = time.flatten()
		beta = self.beta
		beta2 = self.beta2
		R0 = self.R0
		R2 = self.R2
		Revo = R0*time**beta
		Revo2 = R2*time**beta2
		R = np.minimum(Revo,Revo2)
		return R
	def L_evolution(self, time):
		time = time.flatten()
		R_evo = self.R_evolution(time)
		T_evo = self.T_evolution(time) 
		L_evo = sigma_sb*4*np.pi*(R_evo)**2*(T_evo)**4
		return L_evo     
	def mags(self,time,filt_list = FILTERS): 
		m_array={}
		for filt in filt_list:
			R_evo = self.R_evolution(time)
			T_evo = self.T_evolution(time)
			mag = generate_cooling_mag(T_evo, R_evo, filt,d = self.distance, Rv=self.Rv, EBV = self.ebv,EBV_MW = 0, LAW = self.LAW)
			m_array[filt] = mag
		return m_array
	
	def likelihood(self,dat,sys_err = 0.05,nparams = 11):
		t0 = self.t0 
		ebv = self.ebv
		Rv = self.Rv
		LAW = self.LAW
		d = self.distance
		filt_list = np.unique(dat['filter'])
		chi2_dic = {}
		dof = 0
		chi_tot = 0
		N_band = []
		for filt in filt_list:
			dat_filt = dat[dat['filter']==filt]
			time = dat_filt['t_rest']-t0
			R_evo = self.R_evolution(time)
			T_evo = self.T_evolution(time)
			mag = generate_cooling_mag(T_evo, R_evo, filt,d = d, Rv=Rv, EBV = ebv,EBV_MW = 0, LAW = LAW)
			
			mag_obs = dat_filt['absmag']
			mag_err = dat_filt['AB_MAG_ERR'] + sys_err
			
			c2 = chi_sq(mag_obs,mag_err, mag)
			N = len(c2)
			#N_band.append(N)

			chi2_dic[filt] = c2#*N**2


			dof = dof+N
			chi_tot = chi_tot +np.sum(chi2_dic[filt])
		#N_band = np.array(N_band) 
		chi_tot = chi_tot#/np.sum(N_band**2)
		dof = dof - nparams
		
		rchi2 = chi_tot/dof
			
		
		if chi_tot == 0:
			logprob = -np.inf
		elif dof <= 0:
			logprob = -np.inf
		else:
			#prob = scipy.stats.chi2.pdf(chi_tot, dof)
			#prob = float(prob)
			logprob = log_chi_sq_pdf(chi_tot, dof)
		#logprob = np.log(prob)
		return logprob, chi_tot, dof


class model_cooling_combined(object):
	def __init__(self,T0, R0, alpha,beta,t0,t_br,alpha2, taum,Mni,t_gamma,distance = 3.08e26,ebv = 0,Rv = 3.1, LAW = 'MW'):
		self.ebv = ebv
		self.Rv = Rv
		self.distance = distance
		self.LAW  = LAW

		self.T0 = T0
		self.R0 = R0 
		self.alpha = alpha 
		self.beta = beta
		self.alpha2 = alpha2
		self.t_br = t_br
		self.T2 =  T0*(t_br)**(alpha)/((t_br)**(alpha2))
		self.t0 = t0

		self.taum = taum
		self.Mni = Mni
		self.t_gamma = t_gamma


	def T_evolution(self,time): 
		time = time.flatten()
		alpha = self.alpha
		alpha2 = self.alpha2

		T0 = self.T0
		T2 = self.T2

		Tevo = T0*time**alpha
		Tevo2 = T2*time**alpha2
		T = np.maximum(Tevo,Tevo2)
		return T
	def L_evolution(self, time):
		time = time.flatten()
		T_evo = self.T_evolution(time) 
		beta = self.beta
		R0 = self.R0
		taum = self.taum
		Mni = self.Mni
		t_gamma = self.t_gamma
		Revo1 = R0*time**beta
		L_evo1 = 4*np.pi*sigma_sb*(Revo1)**2*(T_evo)**4
		L_Ni =LSN_Ni56(time,taum, Mni = Mni,t0 =t_gamma, kappa=0.07) 
		L_evo = L_evo1+L_Ni

		return L_evo  


	def R_evolution(self,time): 
		time = time.flatten()
		Tevo = self.T_evolution(time)
		Levo = self.L_evolution(time)
		R = np.sqrt(Levo/(4*np.pi*sigma_sb*(Tevo)**4))
		return R
   
	def mags(self,time,filt_list = FILTERS): 
		m_array={}
		for filt in filt_list:
			R_evo = self.R_evolution(time)
			T_evo = self.T_evolution(time)
			mag = generate_cooling_mag(T_evo, R_evo, filt,d = self.distance, Rv=self.Rv, EBV = self.ebv,EBV_MW = 0, LAW = self.LAW)
			m_array[filt] = mag
		return m_array
	
	def likelihood(self,dat,sys_err = 0.05,nparams = 7):
		t0 = self.t0 
		ebv = self.ebv
		Rv = self.Rv
		LAW = self.LAW
		d = self.distance
		filt_list = np.unique(dat['filter'])
		chi2_dic = {}
		dof = 0
		chi_tot = 0
		N_band = []
		for filt in filt_list:
			dat_filt = dat[dat['filter']==filt]
			time = dat_filt['t_rest']-t0
			R_evo = self.R_evolution(time)
			T_evo = self.T_evolution(time)
			mag = generate_cooling_mag(T_evo, R_evo, filt,d = d, Rv=Rv, EBV = ebv,EBV_MW = 0, LAW = LAW)
			
			mag_obs = dat_filt['absmag']
			mag_err = dat_filt['AB_MAG_ERR'] + sys_err
			
			c2 = chi_sq(mag_obs,mag_err, mag)
			N = len(c2)
			#N_band.append(N)

			chi2_dic[filt] = c2#*N**2


			dof = dof+N
			chi_tot = chi_tot +np.sum(chi2_dic[filt])
		#N_band = np.array(N_band) 
		chi_tot = chi_tot#/np.sum(N_band**2)
		dof = dof - nparams
		
		rchi2 = chi_tot/dof
			
		
		if chi_tot == 0:
			logprob = -np.inf
		elif dof <= 0:
			logprob = -np.inf
		else:
			#prob = scipy.stats.chi2.pdf(chi_tot, dof)
			#prob = float(prob)
			logprob = log_chi_sq_pdf(chi_tot, dof)
		#logprob = np.log(prob)
		return logprob, chi_tot, dof

	def likelihood_bb(self,dat_bb,sys_err = 0.05,nparams = 10):
		t0 = self.t0 
		dof = 0
		chi_tot = 0
		
		time = dat_bb['t_rest']-t0
		L_evo = self.L_evolution(time)
		T_evo = self.T_evolution(time)
		err_L = np.sqrt((dat_bb['L_up']-dat_bb['L'])**2 + (sys_err*dat_bb['L'])**2)
		err_T = np.sqrt((dat_bb['T_up']-dat_bb['T'])**2 + (sys_err*dat_bb['T'])**2)

		c2_L = chi_sq(dat_bb['L'],err_L, L_evo)
		c2_T = chi_sq(dat_bb['T'],err_T, T_evo)

		N = len(c2_L) + len(c2_T)
		#N_band.append(N)
		chi_tot = np.sum(c2_L) +np.sum(c2_T) #*N**2
		dof =N
		dof = dof - nparams
		
		rchi2 = chi_tot/dof
			
		#if chi_tot == 0:
		#    logprob = -np.inf
		#elif dof <= 0:
		#    logprob = -np.inf
		#else:
		#    #prob = scipy.stats.chi2.pdf(chi_tot, dof)
		#    #prob = float(prob)
		#    logprob = log_chi_sq_pdf(chi_tot, dof)
		##logprob = np.log(prob)
		logprob = -chi_tot
		return logprob, chi_tot, dof




class model_SC_combined(object):
	def __init__(self,R13,v85,fM,Menv,t0,k34, taum,Mni,t_gamma,distance = 3.08e26,ebv = 0,Rv = 3.1, LAW = 'MW'):
		self.ebv = ebv
		self.Rv = Rv
		self.distance = distance
		self.LAW  = LAW
		self.R13  = R13
		self.v85  = v85
		self.fM   = fM
		self.Menv = Menv
		self.k34  = k34
		self.taum = taum
		self.Mni = Mni
		self.t_gamma = t_gamma
		self.t0 = t0

		t_break_hrs,T_break_eV,L_break,t_tr_day = break_params_new(R13, v85,fM,Menv,k34)
		self.L_break       = L_break
		self.t_break_days  = t_break_hrs/24
		self.t_tr_day      = t_tr_day
		self.T_break       = T_break_eV/k_B_eVK
		t_down,t_up = validity2(R13, v85,fM,k34,Menv)
		self.t_down = t_down 
		self.t_up   = t_up  
	def T_evolution(self,time): 
		time = time.flatten()
		T_break = self.T_break
		t_break_days = self.t_break_days
		T = T_color_bb(time, T_break,t_break_days)
		return T


	def L_evolution(self, time):
		time = time.flatten()
		t_break_days = self.t_break_days
		L_break = self.L_break
		t_tr_day = self.t_tr_day
		taum = self.taum
		Mni = self.Mni
		t_gamma = self.t_gamma
		L_evo1 = L_BB(time, L_break,t_break_days,t_tr_day)
		L_Ni =LSN_Ni56(time,taum, Mni = Mni,t0 =t_gamma) 
		L_evo = L_evo1+L_Ni

		return L_evo  


	def R_evolution(self,time): 
		time = time.flatten()
		Tevo = self.T_evolution(time)
		Levo = self.L_evolution(time)
		R = np.sqrt(Levo/(4*np.pi*sigma_sb*(Tevo)**4))
		return R
   
	def mags(self,time,filt_list = FILTERS): 
		m_array={}
		for filt in filt_list:
			R_evo = self.R_evolution(time)
			T_evo = self.T_evolution(time)
			mag = generate_cooling_mag(T_evo, R_evo, filt,d = self.distance, Rv=self.Rv, EBV = self.ebv,EBV_MW = 0, LAW = self.LAW)
			m_array[filt] = mag
		return m_array

	
	def likelihood(self,dat,sys_err = 0.05,nparams = 7):
		t0 = self.t0 
		ebv = self.ebv
		Rv = self.Rv
		LAW = self.LAW
		d = self.distance
		t_down  = self.t_down 
		t_up    = self.t_up   
		dat = dat[(dat['t_rest']>t_down)&(dat['t_rest']<t_up)]
		filt_list = np.unique(dat['filter'])
		chi2_dic = {}
		dof = 0
		chi_tot = 0
		N_band = []
		for filt in filt_list:
			dat_filt = dat[dat['filter']==filt]
			time = dat_filt['t_rest']-t0
			R_evo = self.R_evolution(time)
			T_evo = self.T_evolution(time)
			mag = generate_cooling_mag(T_evo, R_evo, filt,d = d, Rv=Rv, EBV = ebv,EBV_MW = 0, LAW = LAW)
			
			mag_obs = dat_filt['absmag']
			mag_err = dat_filt['AB_MAG_ERR'] + sys_err
			
			c2 = chi_sq(mag_obs,mag_err, mag)
			N = len(c2)
			#N_band.append(N)

			chi2_dic[filt] = c2#*N**2


			dof = dof+N
			chi_tot = chi_tot +np.sum(chi2_dic[filt])
		#N_band = np.array(N_band) 
		chi_tot = chi_tot#/np.sum(N_band**2)
		dof = dof - nparams
		
		rchi2 = chi_tot/dof
			
		
		if chi_tot == 0:
			logprob = -np.inf
		elif dof <= 0:
			logprob = -np.inf
		else:
			#prob = scipy.stats.chi2.pdf(chi_tot, dof)
			#prob = float(prob)
			logprob = log_chi_sq_pdf(chi_tot, dof)
		#logprob = np.log(prob)
		return logprob, chi_tot, dof

	def likelihood_bb(self,dat_bb,sys_err = 0.05,nparams = 10):
		t0 = self.t0 
		dof = 0
		chi_tot = 0
		
		time = dat_bb['t_rest']-t0
		L_evo = self.L_evolution(time)
		T_evo = self.T_evolution(time)
		err_L = np.sqrt((dat_bb['L_up']-dat_bb['L'])**2 + (sys_err*dat_bb['L'])**2)
		err_T = np.sqrt((dat_bb['T_up']-dat_bb['T'])**2 + (sys_err*dat_bb['T'])**2)

		c2_L = chi_sq(dat_bb['L'],err_L, L_evo)
		c2_T = chi_sq(dat_bb['T'],err_T, T_evo)

		N = len(c2_L) + len(c2_T)
		#N_band.append(N)
		chi_tot = np.sum(c2_L) +np.sum(c2_T) #*N**2
		dof =N
		dof = dof - nparams
		
		rchi2 = chi_tot/dof
			
		#if chi_tot == 0:
		#    logprob = -np.inf
		#elif dof <= 0:
		#    logprob = -np.inf
		#else:
		#    #prob = scipy.stats.chi2.pdf(chi_tot, dof)
		#    #prob = float(prob)
		#    logprob = log_chi_sq_pdf(chi_tot, dof)
		##logprob = np.log(prob)
		logprob = -chi_tot
		return logprob, chi_tot, dof





class model_SC(object):
	def __init__(self,R13,v85,fM,Menv,t0,k34,filter_transmission = None,distance = 3.08e19,ebv = 0,Rv = 3.1, LAW = 'MW'):
		self.ebv = ebv
		self.Rv = Rv
		self.distance = distance
		self.LAW  = LAW
		self.R13  = R13
		self.v85  = v85
		self.fM   = fM
		self.Menv = Menv
		self.k34  = k34
		self.t0 = t0
		t_break_hrs,T_break_eV,L_break,t_tr_day =  break_params_new(R13, v85,fM,Menv,k34)
		self.L_break       = L_break
		self.t_break_days  = t_break_hrs/24
		self.t_tr_day      = t_tr_day
		self.T_break       = T_break_eV/k_B_eVK
		t_down,t_up = validity2(R13, v85,fM,k34,Menv)
		self.t_down = t_down
		self.t_up   = t_up
		self.filter_transmission = filter_transmission
	def T_evolution(self,time): 
		time = time.flatten()
		T_break = self.T_break
		t_break_days = self.t_break_days
		T = T_color_bb(time, T_break,t_break_days)
		return T
	def L_evolution(self, time):
		time = time.flatten()
		t_break_days = self.t_break_days
		L_break = self.L_break
		t_tr_day = self.t_tr_day
		L_evo = L_BB(time, L_break,t_break_days,t_tr_day)
		return L_evo  
	def R_evolution(self,time): 
		time = time.flatten()
		Tevo = self.T_evolution(time)
		Levo = self.L_evolution(time)
		R = np.sqrt(Levo/(4*np.pi*sigma_sb*(Tevo)**4))
		return R
	def mags(self,time,filt_list): 
		m_array={}
		for filt in filt_list:
			R_evo = self.R_evolution(time)
			T_evo = self.T_evolution(time)
			mag = generate_cooling_mag_single(T_evo, R_evo, self.filter_transmission[filt],d = self.distance, Rv=self.Rv, EBV = self.ebv,EBV_MW = 0, LAW = self.LAW)
			m_array[filt] = mag
		return m_array
	def mags_single(self,time,filt): 
		#m_array = generate_SC_mag_single(time,self.R13,self.v85,self.fM,self.k34,self.Menv,filt, self.filter_transmission[filt]
		#							, d =  self.distance, Rv=self.Rv, EBV = self.ebv
		#  							,EBV_MW = 0, LAW = self.LAW)
		R_evo = self.R_evolution(time)
		T_evo = self.T_evolution(time)
		m_array = generate_cooling_mag_single(T_evo, R_evo,self.filter_transmission[filt],d = self.distance, Rv=self.Rv, EBV = self.ebv,EBV_MW = 0, LAW = self.LAW)

		return m_array    	
	def flux(self,time,lam): 
		f_array = generate_MW_flux(time,lam,self.R13,self.v85,self.fM,self.k34,self.Menv
									, d =  self.distance, Rv=self.Rv, EBV = self.ebv
									,EBV_MW = 0, LAW = self.LAW,UV_sup = False,reduced = False, old_form = False, gray = True)

		return f_array 
	def likelihood(self,dat,sys_err = 0.05,nparams = 7):
		t0 = self.t0 
		ebv = self.ebv
		Rv = self.Rv
		LAW = self.LAW
		d = self.distance
		t_down  = self.t_down 
		t_up    = self.t_up   
		dat = dat[(dat['t_rest']>t_down)&(dat['t_rest']<t_up)]
		filt_list = np.unique(dat['filter'])
		chi2_dic = {}
		dof = 0
		chi_tot = 0
		N_band = []
		for filt in filt_list:
			dat_filt = dat[dat['filter']==filt]
			time = dat_filt['t_rest']-t0
			m_array = self.mags(time,filt_list = [filt])
			mag = m_array[filt]
			mag_obs = dat_filt['absmag']
			mag_err = dat_filt['AB_MAG_ERR'] + sys_err
			c2 = chi_sq(mag_obs,mag_err, mag)
			N = len(c2)
			chi2_dic[filt] = c2
			dof = dof+N
			chi_tot = chi_tot +np.sum(chi2_dic[filt])
		dof = dof - nparams
		
		rchi2 = chi_tot/dof
			
		if chi_tot == 0:
			logprob = -np.inf
		elif dof <= 0:
			logprob = -np.inf
		else:
			logprob = log_chi_sq_pdf(chi_tot, dof)
		if np.isnan(logprob):
			import ipdb; ipdb.set_trace()
		return logprob, chi_tot, dof
	def likelihood_cov(self,dat,inv_cov,sys_err = 0,nparams = 7):
		t0 = self.t0 
		ebv = self.ebv
		Rv = self.Rv
		LAW = self.LAW
		d = self.distance
		t_down  = self.t_down 
		t_up    = self.t_up   
		cond_valid = (dat['t_rest']-t0>t_down)&(dat['t_rest']-t0<t_up)
		args_valid = np.argwhere(cond_valid).flatten()
		if np.sum(cond_valid) == 0:
			chi_tot = 0
			logprob = -np.inf
			dof = -1
			return logprob, chi_tot, dof
		inds = [np.min(args_valid),np.max(args_valid)+1]

		dat = dat[args_valid]
		#_,dat_res = compute_resid(dat.copy(),self)
		dof = len(dat)
		filt_list = np.unique(dat['filter'])
		delta = np.zeros(dof,)
		for filt in filt_list:
			cond_filt = dat['filter']==filt
			args_filt = np.argwhere(cond_filt).flatten()
			dat_filt = dat[args_filt]
			time = dat_filt['t_rest']-t0
			mag = dat['absmag'][args_filt]
			mags =   self.mags_single(time,filt=filt)
			res = np.array(mag - mags)
			delta[cond_filt] = res
		#cov = cov[inds[0]:inds[1],inds[0]:inds[1]]
		#inv_cov = np.linalg.inv(cov)
		inv_cov = inv_cov[inds[0]:inds[1],inds[0]:inds[1]]
		#prod = np.dot(delta,inv_cov)
		#chi_tot = np.dot(prod,delta)
		chi_tot = delta @ inv_cov @ delta
		dof = dof - nparams
		rchi2 = chi_tot/dof
		if chi_tot == 0:
			logprob = -np.inf
		elif dof <= 0:
			logprob = -np.inf
		else:
			#prob = scipy.stats.chi2.pdf(chi_tot, dof)
			#prob = float(prob)
			logprob = log_chi_sq_pdf(chi_tot, dof)
		if np.isnan(logprob):
			import ipdb; ipdb.set_trace()
		#logprob = np.log(prob)
		return logprob, chi_tot, dof
	def likelihood_bb(self,dat_bb,sys_err = 0.05,nparams = 10):
		t0 = self.t0 
		dof = 0
		chi_tot = 0
		
		time = dat_bb['t_rest']-t0
		L_evo = self.L_evolution(time)
		T_evo = self.T_evolution(time)
		err_L = np.sqrt((dat_bb['L_up']-dat_bb['L'])**2 + (sys_err*dat_bb['L'])**2)
		err_T = np.sqrt((dat_bb['T_up']-dat_bb['T'])**2 + (sys_err*dat_bb['T'])**2)

		c2_L = chi_sq(dat_bb['L'],err_L, L_evo)
		c2_T = chi_sq(dat_bb['T'],err_T, T_evo)

		N = len(c2_L) + len(c2_T)
		chi_tot = np.sum(c2_L) +np.sum(c2_T) 
		dof =N
		dof = dof - nparams
		
		rchi2 = chi_tot/dof

		logprob = -chi_tot
		return logprob, chi_tot, dof








def generate_cooling_mag_single(T_array, R_array ,filter_transmission,z=0, d = 3.08e26, Rv=3.1, EBV = 0,EBV_MW = 0, LAW = 'MW',H0 = 67.4):
	if len(T_array)!=len(R_array):
		raise Exception('R and T arrays have to be the same length') 
	v = H0*d/3.08e24
	z = v/300000
	Trans  = filter_transmission

	lam  = np.linspace(np.min(Trans[:,0]),np.max(Trans[:,0]),90)
	mm = []
	for i in range(len(T_array)):
		flux = f_nu_bb(lam,T_array[i],R_array[i],d = d)[0]
		if EBV>0:
			flux = apply_extinction(lam,flux,EBV=EBV,EBV_mw=EBV_MW, R_v=Rv,z=z,LAW=LAW)
		#m = SynPhot_fast(lam,flux,Trans)
		m = SynPhot_fast_AB(lam,flux,Trans[:,0],Trans[:,1])
		mm.append(m)	
	m_array = np.array(mm)
	return m_array 
	

def generate_cooling_mag(T_array, R_array ,filt,z=0, d = 3.08e26, Rv=3.1, EBV = 0,EBV_MW = 0, LAW = 'MW',H0 = 67.4):
	if len(T_array)!=len(R_array):
		raise Exception('R and T arrays have to be the same length') 
	v = H0*d/3.08e24
	z = v/300000
	lam  = np.linspace(1000,10000,90)
	mm = []
	for i in range(len(T_array)):
		flux = f_nu_bb(lam,T_array[i],R_array[i],d = d)[0]
		if EBV>0:
			flux = apply_extinction(lam,flux,EBV=EBV,EBV_mw=EBV_MW, R_v=Rv,z=z,LAW=LAW)
		try:
			Trans  = filter_transmission[filt]
		except:
			import ipdb; ipdb.set_trace()
		#m = SynPhot_fast(lam,flux,Trans)

		m = SynPhot_fast_AB(lam,flux,Trans['col1'],Trans['col2'])
		mm.append(m)	
	m_array = np.array(mm)
	return m_array 



@njit
def chi_sq(obs,err, mod):
	c2 = (obs-mod)**2/err**2
	return c2
from math import gamma


@njit
def log_chi_sq_pdf(x, k):

	logp = (k/2-1)*np.log(x)-0.5*x-0.5*k*np.log(2)-np.log(gamma(k/2))
	return logp


def compute_resid(Data,obj):
	filt_list = np.unique(Data['filter'])
	cond_dic = {}
	resid = {}
	obj.distance = 3.0856e+19
	Data['resid'] = np.zeros_like(Data['absmag'])
	Data['mod_mag'] = np.zeros_like(Data['absmag'])

	for band in filt_list:
		cond_dic[band] = (Data['filter']==band)&(Data['t']>0)
	# change to model_cooling_broken
	for i,band in enumerate(filt_list):
		t = Data['t_rest'][cond_dic[band]]-obj.t0
		mag = Data['absmag'][cond_dic[band]]
		mags =   obj.mags(t,filt_list=[band])
		res = np.array(mag - mags[band])
		resid[band] = res
		Data['resid'][cond_dic[band]] = res
		Data['mod_mag'][cond_dic[band]] = mags[band]

	return resid,Data





def likelihood_model_cooling(data, T0, R0, alpha,beta,t0,distance = 3.0856e19,ebv = 0,Rv = 3.1 , LAW = 'MW',sys_err = 0.05,nparams = 7):
	obj = model_cooling(T0, R0, alpha,beta,t0, distance =distance, ebv = ebv, Rv = Rv, LAW = LAW)
	logprob, chi_tot, dof = obj.likelihood(data, sys_err = sys_err ,nparams = nparams)
	return logprob, chi_tot, dof 


def likelihood_broken_double_pl(data, T0, R0, alpha,beta,t0, t_br, t_br2, alpha2,beta2,distance = 3.0856e19,ebv = 0,Rv = 3.1 , LAW = 'MW',sys_err = 0.05,nparams = 11):
	obj = model_cooling_broken(T0, R0, alpha,beta,t0, t_br, t_br2, alpha2,beta2, distance =distance, ebv = ebv, Rv = Rv, LAW = LAW)
	logprob, chi_tot, dof = obj.likelihood(data, sys_err = sys_err ,nparams = 11)
	return logprob, chi_tot, dof

def likelihood_broken_power_law(data, T0, R0, alpha,beta,t0, t_br, alpha2,beta2,distance = 3.0856e19,ebv = 0,Rv = 3.1 , LAW = 'MW',sys_err = 0.05,nparams = 10):
	T2 = T0*(t_br)**(alpha)/((t_br)**(alpha2))
	R2 = R0*(t_br)**(beta)/((t_br)**(beta2))

	obj1 = model_cooling(T0, R0, alpha,beta,t0, distance =distance, ebv = ebv, Rv = Rv, LAW = LAW)
	obj2 = model_cooling(T2, R2, alpha2,beta2,t0, distance =distance, ebv = ebv, Rv = Rv, LAW = LAW)



	data1 = data[data['t']<t_br+t0]
	data2 = data[data['t']>=t_br+t0]

	_, chi_tot1, dof1 = obj1.likelihood(data1, sys_err = sys_err ,nparams = 0)
	_, chi_tot2, dof2 = obj2.likelihood(data2, sys_err = sys_err ,nparams = 0)
	chi_tot = chi_tot1 + chi_tot2
	dof_tot = dof1 + dof2 - nparams
	if chi_tot == 0:
		logprob = -np.inf
	elif dof_tot <= 0:
		logprob = -np.inf
	else:
		#prob = scipy.stats.chi2.pdf(chi_tot, dof)
		#prob = float(prob)
		logprob = log_chi_sq_pdf(chi_tot, dof_tot)

	return logprob, chi_tot, dof_tot 

def likelihood_broken_pl_Ni(data, T0, R0, alpha,beta,t0, t_br, alpha2,taum=10,Mni=0.1,t_gamma=100,distance = 3.0856e19,ebv = 0,Rv = 3.1 , LAW = 'MW',sys_err = 0.05,nparams = 8):

	obj = model_cooling_combined(T0, R0, alpha,beta,t0,t_br,alpha2, taum,Mni,t_gamma, distance =distance, ebv = ebv, Rv = Rv, LAW = LAW)
	logprob, chi_tot, dof_tot = obj.likelihood(data, sys_err = sys_err ,nparams = 7)
	return logprob, chi_tot, dof_tot 

def likelihood_powerlaw(data, T0, R0, alpha,beta,t0,data_bb = 'None',distance = 3.0856e19,ebv = 0,Rv = 3.1 , LAW = 'MW',sys_err = 0.05,nparams = 5):

	obj = model_cooling(T0, R0, alpha,beta,t0, distance =distance, ebv = ebv, Rv = Rv, LAW = LAW)


	_, chi_tot, dof = obj.likelihood(data, sys_err = sys_err ,nparams = nparams)
	if data_bb!='None':
		_, chi_tot_bb, dof_bb = obj.likelihood_bb(data_bb, sys_err = sys_err ,nparams = nparams)
		chi_tot = chi_tot+chi_tot_bb
		dof_tot = dof + dof_bb
	else:
		dof_tot = dof
	if chi_tot == 0:
		logprob = -np.inf
	elif dof_tot <= 0:
		logprob = -np.inf
	else:
		#prob = scipy.stats.chi2.pdf(chi_tot, dof)
		#prob = float(prob)
		logprob = log_chi_sq_pdf(chi_tot, dof_tot)

	return logprob, chi_tot, dof_tot 


def likelihood_SC_Ni(data,R13,v85,fM,Menv,t0,k34=1,taum=10,Mni=0.1,t_gamma=100,distance = 3.0856e19,ebv = 0,Rv = 3.1 , LAW = 'MW',sys_err = 0.05,nparams = 8):

	obj = model_SC_combined(R13,v85,fM,Menv,t0,k34, taum,Mni,t_gamma, distance =distance, ebv = ebv, Rv = Rv, LAW = LAW)
	logprob, chi_tot, dof_tot = obj.likelihood(data, sys_err = sys_err ,nparams = 6)
	return logprob, chi_tot, dof_tot 


def likelihood_SC(data,R13,v85,fM,Menv,t0,k34=1,distance = 3.0856e19,ebv = 0,Rv = 3.1 , LAW = 'MW',sys_err = 0.05,nparams = 7):

	obj = model_SC(R13,v85,fM,Menv,t0,k34, distance =distance, ebv = ebv, Rv = Rv, LAW = LAW)
	logprob, chi_tot, dof_tot = obj.likelihood(data, sys_err = sys_err ,nparams = nparams)
	return logprob, chi_tot, dof_tot 
def likelihood_SC_cov(data,inv_cov,R13,v85,fM,Menv,t0,filter_transmission,k34=1,distance = 3.0856e19,ebv = 0,Rv = 3.1 , LAW = 'MW',sys_err = 0.05,nparams = 7,UV_sup = False,reduced=True):
	
	obj = model_SC(R13,v85,fM,Menv,t0,k34, distance =distance, ebv = ebv, Rv = Rv, LAW = LAW,filter_transmission = filter_transmission)
	logprob, chi_tot, dof_tot = obj.likelihood_cov(data,inv_cov, sys_err = sys_err ,nparams = nparams)
	return logprob, chi_tot, dof_tot 


def fit_SC_Ni(data,taum,Mni,t_gamma,k34 = 1, plot_corner = True,sys_err = 0.05, LAW = 'MW',ebv = 0, Rv = 3.1,**kwargs):  
	inputs={'priors':[np.array([0.01,5]),
					  np.array([0.3,5]),
					  np.array([0.1,1000]),
					  np.array([0.1,10]),
					  #np.array([-0.05,0.05])
					  ]
			,'maxiter':100000
			,'maxcall':500000
			,'nlive':250}                            
	inputs.update(kwargs)
	priors=inputs.get('priors')  
	maxiter=inputs.get('maxiter')  
	maxcall=inputs.get('maxcall')  
	nlive=inputs.get('nlive')  

	def prior_transform(u,priors = priors):
		x=uniform_prior_transform(u,priors =priors )
		return x
	def myloglike(x):
		R13,v85,fM,Menv = x   
		loglike = likelihood_SC_Ni(data,R13,v85,fM,Menv,t0=0,k34 = k34,taum=taum,Mni=Mni,t_gamma=t_gamma,ebv = ebv, Rv = Rv, sys_err = sys_err, LAW = LAW)[0]
		return loglike  

	dsampler = dynesty.DynamicNestedSampler(myloglike, prior_transform,  ndim = 5,nlive=nlive,update_interval=600)
	dsampler.run_nested(maxiter=maxiter, maxcall=maxcall)
	dresults = dsampler.results
	if plot_corner:
		# Plot a summary of the run.
		#rfig, raxes = dyplot.runplot(dresults)
		# Plot traces and 1-D marginalized posteriors.
		#tfig, taxes = dyplot.traceplot(dresults)    
		# Plot the 2-D marginalized posteriors.
		cfig, caxes = dyplot.cornerplot(dresults,labels=[r'$R_{13}$',r'$v_{s,8.5}$',r'$f_{\rhp}M$',r'$M_{env}$',r'$t_{0}$']
								,label_kwargs={'fontsize':14}, color = '#0042CF',show_titles = True)
	## add labels!
	# Extract sampling results.
	samples = dresults.samples  # samples
	weights = np.exp(dresults.logwt - dresults.logz[-1])  # normalized weights
	# Compute 10%-90% quantiles.
	quantiles = [dyfunc.quantile(samps, [0.1, 0.9], weights=weights)
				 for samps in samples.T]
	# Compute weighted mean and covariance.
	mean, cov = dyfunc.mean_and_cov(samples, weights)
	# Resample weighted samples.
	samples_equal = dyfunc.resample_equal(samples, weights)
	# Generate a new set of results with statistical+sampling uncertainties.
	results_sim = dyfunc.simulate_run(dresults)

	return mean, quantiles,dresults



	
def fit_SC(data, dic_transmission,k34 = 1, plot_corner = True,sys_err = 0.05,**kwargs):  
	'''
	Function to fit 
	'''
	inputs={'priors':[np.array([0.05,5]),
					  np.array([0.3,5]),
					  np.array([0.05,1000]),
					  np.array([0.1,10]),
					  np.array([-0.5,0.5]),
					  np.array([0,0.3]),
					  np.array([2,5])]
			,'maxiter':100000
			,'maxcall':500000
			,'nlive':250
			,'ebv':'fit'
			,'Rv':3.1
			,'LAW':'MW',
			'UV_sup':False,
			'reduced':True,
			'covariance':True
			,'inv_cov':''
			,'rec_time_lims':[-np.inf,np.inf]}                            
	inputs.update(kwargs)
 
	maxiter=inputs.get('maxiter')  
	maxcall=inputs.get('maxcall')  
	nlive=inputs.get('nlive')  
	ebv = inputs.get('ebv')  
	Rv = inputs.get('Rv')  
	LAW = inputs.get('LAW')  
	priors=inputs.get('priors') 
	UV_sup = inputs.get('UV_sup')
	reduced = inputs.get('reduced')

	covariance = inputs.get('covariance')
	inv_cov = inputs.get('inv_cov')
	rec_time_lims = inputs.get('rec_time_lims')

	if ebv == 'fit':
		if Rv != 'fit':
			priors = priors[0:6]
	else:
		priors = priors[0:5]
	if inv_cov =='':
		 inv_cov=  np.diagflat(1/(np.array(data['AB_MAG_ERR'])**2+ sys_err**2))

	global filter_transmission_fast 
	filter_transmission_fast = dic_transmission
	data = data['t_rest','filter','absmag','AB_MAG_ERR']


	def prior_transform(u,priors = priors):
		x=uniform_prior_transform(u,priors =priors )
		return x

	def myloglike(x):
		R13,v85,fM,Menv,t0,ebv = x   
		t07 = 6.86*R13**(0.56)*v85**(0.16)*k34**(-0.61)*fM**(-0.06) + t0
		if (t07<rec_time_lims[1])&(t07>rec_time_lims[0]):
			if covariance:
				loglike = likelihood_SC_cov(data,inv_cov,R13,v85,fM,Menv,t0=t0,k34 = k34,ebv = ebv, Rv = Rv, sys_err = sys_err, LAW = LAW,nparams = 6,UV_sup=UV_sup,reduced=reduced,filter_transmission = filter_transmission_fast)[0]
			else:
				loglike = likelihood_SC(data,R13,v85,fM,Menv,t0=t0,k34 = k34,ebv = ebv, Rv = Rv, sys_err = sys_err, LAW = LAW,nparams = 6,UV_sup=UV_sup,reduced=reduced,filter_transmission = filter_transmission_fast)[0]
		else: 
			loglike = -np.inf
		return loglike  
	def myloglike2(x):
		R13,v85,fM,Menv,t0,ebv,Rv = x   
		t07 = 6.86*R13**(0.56)*v85**(0.16)*k34**(-0.61)*fM**(-0.06) + t0
		if (t07<rec_time_lims[1])&(t07>rec_time_lims[0]):        
			if covariance:
				loglike = likelihood_SC_cov(data,inv_cov,R13,v85,fM,Menv,t0=t0,k34 = k34,ebv = ebv, Rv = Rv, sys_err = sys_err, LAW = LAW,nparams = 7,UV_sup=UV_sup,reduced=reduced,filter_transmission = filter_transmission_fast)[0]
			else: 
				loglike = likelihood_SC(data,R13,v85,fM,Menv,t0=t0,k34 = k34,ebv = ebv, Rv = Rv, sys_err = sys_err, LAW = LAW,nparams = 7,UV_sup=UV_sup,reduced=reduced,filter_transmission = filter_transmission_fast)[0]
		else: 
			loglike = -np.inf
		return loglike
	def myloglike3(x):
		R13,v85,fM,Menv,t0 = x 
		t07 = 6.86*R13**(0.56)*v85**(0.16)*k34**(-0.61)*fM**(-0.06) + t0
		if (t07<rec_time_lims[1])&(t07>rec_time_lims[0]):   
			if covariance:
				loglike = likelihood_SC_cov(data,inv_cov,R13,v85,fM,Menv,t0=t0,k34 = k34,ebv = ebv, Rv = Rv, sys_err = sys_err, LAW = LAW,nparams = 5,UV_sup=UV_sup,reduced=reduced,filter_transmission = filter_transmission_fast)[0]
			else: 
				loglike = likelihood_SC(data,R13,v85,fM,Menv,t0=t0,k34 = k34,ebv = ebv, Rv = Rv, sys_err = sys_err, LAW = LAW,nparams = 5,UV_sup=UV_sup,reduced=reduced,filter_transmission = filter_transmission_fast)[0]
		else: 
			loglike = -np.inf
		return loglike

	if ebv == 'fit':
		if Rv == 'fit': 
			myloglike_choice = myloglike2
			ndim = 7
			labels = [r'$R_{13}$',r'$v_{s,8.5}$',r'$f_{\rhp}M$',r'$M_{env}$',r'$t_{0}$',r'$E(B-V)$',r'$R_V$']
			labels = labels        
		else:
			ndim = 6
			labels = [r'$R_{13}$',r'$v_{s,8.5}$',r'$f_{\rhp}M$',r'$M_{env}$',r'$t_{0}$',r'$E(B-V)$'] 
			myloglike_choice = myloglike
	else:
		ndim = 5
		labels = [r'$R_{13}$',r'$v_{s,8.5}$',r'$f_{\rhp}M$',r'$M_{env}$',r'$t_{0}$']
		myloglike_choice = myloglike3

	dsampler = dynesty.DynamicNestedSampler(myloglike_choice, prior_transform,  ndim = ndim,nlive=nlive,update_interval=600)

	dsampler.run_nested(maxiter=maxiter, maxcall=maxcall)
	dresults = dsampler.results
	if plot_corner:
		# Plot a summary of the run.
		#rfig, raxes = dyplot.runplot(dresults)
		# Plot traces and 1-D marginalized posteriors.
		#tfig, taxes = dyplot.traceplot(dresults)    
		# Plot the 2-D marginalized posteriors.
		cfig, caxes = dyplot.cornerplot(dresults,labels=labels
								,label_kwargs={'fontsize':14}, color = '#0042CF',show_titles = True)
	## add labels!
	# Extract sampling results.
	samples = dresults.samples  # samples
	weights = np.exp(dresults.logwt - dresults.logz[-1])  # normalized weights
	# Compute 10%-90% quantiles.
	quantiles = [dyfunc.quantile(samps, [0.1, 0.9], weights=weights)
				 for samps in samples.T]
	# Compute weighted mean and covariance.
	mean, _ = dyfunc.mean_and_cov(samples, weights)
	# Resample weighted samples.
	#samples_equal = dyfunc.resample_equal(samples, weights)
	# Generate a new set of results with statistical+sampling uncertainties.
	#results_sim = dyfunc.simulate_run(dresults)

	return mean, quantiles,dresults



def fit_powerlaw(data,data_bb = 'None', plot_corner = True,sys_err = 0.05, LAW = 'MW',**kwargs):  
	inputs={'priors':[np.array([15000,50000]),
					  np.array([10**14,10**15]),
					  np.array([-1.3,0]),
					  np.array([0.5,1]),
					  np.array([-0.5,0.5])]
			,'maxiter':100000
			,'maxcall':500000
			,'nlive':250}                            
	inputs.update(kwargs)
	priors=inputs.get('priors')  
	maxiter=inputs.get('maxiter')  
	maxcall=inputs.get('maxcall')  
	nlive=inputs.get('nlive')  
  
	def prior_transform(u,priors = priors):
		x=uniform_prior_transform(u,priors =priors )
		return x
	def myloglike(x):
		T0,R0,alpha,beta,t0= x   
		loglike = likelihood_powerlaw(data, T0,R0,alpha,beta,t0,data_bb = data_bb,ebv = 0, Rv = 3.1, sys_err = sys_err, LAW = LAW)[0]
		return loglike  
	dsampler = dynesty.DynamicNestedSampler(myloglike, prior_transform, ndim = 5,nlive=250,update_interval=600)
	dsampler.run_nested(maxiter=maxiter, maxcall=maxcall)
	dresults = dsampler.results
	if plot_corner:
		# Plot a summary of the run.
		#rfig, raxes = dyplot.runplot(dresults)
		# Plot traces and 1-D marginalized posteriors.
		#tfig, taxes = dyplot.traceplot(dresults)    
		# Plot the 2-D marginalized posteriors.
		cfig, caxes = dyplot.cornerplot(dresults)
	## add labels!
	# Extract sampling results.
	samples = dresults.samples  # samples
	weights = np.exp(dresults.logwt - dresults.logz[-1])  # normalized weights
	# Compute 10%-90% quantiles.
	quantiles = [dyfunc.quantile(samps, [0.1, 0.9], weights=weights)
				 for samps in samples.T]
	# Compute weighted mean and covariance.
	mean, cov = dyfunc.mean_and_cov(samples, weights)
	# Resample weighted samples.
	samples_equal = dyfunc.resample_equal(samples, weights)
	# Generate a new set of results with statistical+sampling uncertainties.
	results_sim = dyfunc.simulate_run(dresults)

	return mean, quantiles,dresults




def uniform_prior_transform(u,**kwargs):
	inputs={'priors':[np.array([15000,50000]),
					  np.array([10**14,10**15]),
					  np.array([-1.3,0]),
					  np.array([0.5,1]),
					  np.array([-1,0]),
					  np.array([1,5]),
					  np.array([0,-1.3]),
					  np.array([0.5,1])]}                            
	inputs.update(kwargs)
	priors=inputs.get('priors')
	def strech_prior(u,pmin,pmax):
		x = (pmax-pmin) * u + pmin
		return x
	prior_list = []
	for i,prior in enumerate(priors):
		pmin=prior[0]
		pmax=prior[1]
		prior_dist=strech_prior(u[i],pmin,pmax)
		prior_list.append(prior_dist)
	return prior_list


def fit_broken_powerlaw(data,priors, plot_corner = True,sys_err = 0.05, LAW = 'MW'):    
	def prior_transform(u,priors = priors):
		x=uniform_prior_transform(u,priors =priors )
		return x
	def myloglike(x):
		T0,R0,alpha,beta,t0,t_br,alpha2, beta2 = x   
		loglike = likelihood_broken_power_law(data, T0,R0,alpha,beta,t0,t_br,alpha2, beta2,ebv = 0, Rv = 3.1, sys_err = sys_err, LAW = LAW)[0]
		return loglike  
	dsampler = dynesty.DynamicNestedSampler(myloglike, prior_transform, ndim = 8,nlive=250,update_interval=600)
	dsampler.run_nested(maxiter=100000, maxcall=500000)
	dresults = dsampler.results
	if plot_corner:
		# Plot a summary of the run.
		#rfig, raxes = dyplot.runplot(dresults)
		# Plot traces and 1-D marginalized posteriors.
		#tfig, taxes = dyplot.traceplot(dresults)    
		# Plot the 2-D marginalized posteriors.
		cfig, caxes = dyplot.cornerplot(dresults)
	## add labels!
	# Extract sampling results.
	samples = dresults.samples  # samples
	weights = np.exp(dresults.logwt - dresults.logz[-1])  # normalized weights
	# Compute 10%-90% quantiles.
	quantiles = [dyfunc.quantile(samps, [0.1, 0.9], weights=weights)
				 for samps in samples.T]
	# Compute weighted mean and covariance.
	mean, cov = dyfunc.mean_and_cov(samples, weights)
	# Resample weighted samples.
	samples_equal = dyfunc.resample_equal(samples, weights)
	# Generate a new set of results with statistical+sampling uncertainties.
	results_sim = dyfunc.simulate_run(dresults)

	return mean, quantiles,dresults


def fit_broken_powerlaw2(data, plot_corner = True,sys_err = 0.05, LAW = 'MW',**kwargs):  
	inputs={'priors':[np.array([15000,50000]),
					  np.array([10**14,10**15]),
					  np.array([-1.3,0]),
					  np.array([0.5,1]),
					  np.array([-1,0]),
					  np.array([1,5]),
					  np.array([1,5]),
					  np.array([0,-1.3]),
					  np.array([0.5,1])]
			,'maxiter':100000
			,'maxcall':500000
			,'nlive':250}                            
	inputs.update(kwargs)
	priors=inputs.get('priors')  
	maxiter=inputs.get('maxiter')  
	maxcall=inputs.get('maxcall')  
	nlive=inputs.get('nlive')  

	def prior_transform(u,priors = priors):
		x=uniform_prior_transform(u,priors =priors )
		return x
	def myloglike(x):
		T0,R0,alpha,beta,t0,t_br,t_br2,alpha2, beta2 = x   
		loglike = likelihood_broken_double_pl(data, T0,R0,alpha,beta,t0,t_br,t_br2,alpha2, beta2,ebv = 0, Rv = 3.1, sys_err = sys_err, LAW = LAW)[0]
		return loglike  
	dsampler = dynesty.DynamicNestedSampler(myloglike, prior_transform,  ndim = 9,nlive=nlive,update_interval=600)
	dsampler.run_nested(maxiter=maxiter, maxcall=maxcall)
	dresults = dsampler.results
	if plot_corner:
		# Plot a summary of the run.
		#rfig, raxes = dyplot.runplot(dresults)
		# Plot traces and 1-D marginalized posteriors.
		#tfig, taxes = dyplot.traceplot(dresults)    
		# Plot the 2-D marginalized posteriors.
		cfig, caxes = dyplot.cornerplot(dresults,labels=['$T_{0}$','$R_{0}$',r'$\alpha$',r'$\beta$',r'$t_{exp}$',r'$t_{br}$',r'$t_{br,2}$',r'$\alpha_{2}$',r'$\beta_{2}$']
								,label_kwargs={'fontsize':14}, color = '#0042CF',show_titles = True)
	## add labels!
	# Extract sampling results.
	samples = dresults.samples  # samples
	weights = np.exp(dresults.logwt - dresults.logz[-1])  # normalized weights
	# Compute 10%-90% quantiles.
	quantiles = [dyfunc.quantile(samps, [0.1, 0.9], weights=weights)
				 for samps in samples.T]
	# Compute weighted mean and covariance.
	mean, cov = dyfunc.mean_and_cov(samples, weights)
	# Resample weighted samples.
	samples_equal = dyfunc.resample_equal(samples, weights)
	# Generate a new set of results with statistical+sampling uncertainties.
	results_sim = dyfunc.simulate_run(dresults)

	return mean, quantiles,dresults


def fit_broken_powerlaw_Ni(data,taum,Mni,t_gamma, plot_corner = True,sys_err = 0.05, LAW = 'MW',ebv = 0, Rv = 3.1,**kwargs):  
	inputs={'priors':[np.array([15000,50000]),
					  np.array([10**14,10**15]),
					  np.array([-1.3,0]),
					  np.array([0.5,1]),
					  np.array([-1,0]),
					  np.array([1,5]),
					  np.array([0,-1.3])]
			,'maxiter':100000
			,'maxcall':500000
			,'nlive':250}                            
	inputs.update(kwargs)
	priors=inputs.get('priors')  
	maxiter=inputs.get('maxiter')  
	maxcall=inputs.get('maxcall')  
	nlive=inputs.get('nlive')  

	def prior_transform(u,priors = priors):
		x=uniform_prior_transform(u,priors =priors )
		return x
	def myloglike(x):
		T0, R0, alpha,beta,t0, t_br, alpha2 = x   
		loglike = likelihood_broken_pl_Ni(data, T0, R0, alpha,beta,t0, t_br, alpha2,taum=taum,Mni=Mni,t_gamma=t_gamma,ebv = ebv, Rv = Rv, sys_err = sys_err, LAW = LAW)[0]
		return loglike  
	dsampler = dynesty.DynamicNestedSampler(myloglike, prior_transform,  ndim = 7,nlive=nlive,update_interval=600)
	dsampler.run_nested(maxiter=maxiter, maxcall=maxcall)
	dresults = dsampler.results
	if plot_corner:
		# Plot a summary of the run.
		#rfig, raxes = dyplot.runplot(dresults)
		# Plot traces and 1-D marginalized posteriors.
		#tfig, taxes = dyplot.traceplot(dresults)    
		# Plot the 2-D marginalized posteriors.
		cfig, caxes = dyplot.cornerplot(dresults,labels=['$T_{0}$','$R_{0}$',r'$\alpha$',r'$\beta$',r'$t_{exp}$',r'$t_{br}$',r'$\alpha_{2}$']
								,label_kwargs={'fontsize':14}, color = '#0042CF',show_titles = True)
	## add labels!
	# Extract sampling results.
	samples = dresults.samples  # samples
	weights = np.exp(dresults.logwt - dresults.logz[-1])  # normalized weights
	# Compute 10%-90% quantiles.
	quantiles = [dyfunc.quantile(samps, [0.1, 0.9], weights=weights)
				 for samps in samples.T]
	# Compute weighted mean and covariance.
	mean, cov = dyfunc.mean_and_cov(samples, weights)
	# Resample weighted samples.
	samples_equal = dyfunc.resample_equal(samples, weights)
	# Generate a new set of results with statistical+sampling uncertainties.
	results_sim = dyfunc.simulate_run(dresults)

	return mean, quantiles,dresults



def fit_Ni(times, L_bol,L_bol_err,kappa=0.1, N=5000, **kwargs):
	inputs={'priors':[np.array([1,100]),
					  np.array([0.01,5]),
					  np.array([5,500])]
			,'plot_corner':False
			,'maxiter':100000
			,'maxcall':500000
			,'nlive':250}                            
	inputs.update(kwargs)
	priors=inputs.get('priors')  
	maxiter=inputs.get('maxiter')  
	maxcall=inputs.get('maxcall')  
	nlive=inputs.get('nlive')
	plot_corner=inputs.get('plot_corner')  
  
	def prior_transform(u,priors = priors):
		x=uniform_prior_transform(u,priors =priors )
		return x
	def myloglike(x):
		tau, Mni, t0 = x   
		model = LSN_Ni56(times,taum = tau, Mni = Mni,t0 = t0,N = N, kappa=kappa)
		c2 = np.sum(chi_sq(L_bol,L_bol_err, model))
		if np.isnan(c2):
			import ipdb; ipdb.set_trace()
		loglike = -c2#log_chi_sq_pdf(c2, len(times)) 
		return loglike 
	dsampler = dynesty.DynamicNestedSampler(myloglike, prior_transform, ndim = 3,nlive=nlive,update_interval=600)
	

	dsampler.run_nested(maxiter=maxiter, maxcall=maxcall)

	dresults = dsampler.results
	if plot_corner:
		# Plot a summary of the run.
		#rfig, raxes = dyplot.runplot(dresults)
		# Plot traces and 1-D marginalized posteriors.
		#tfig, taxes = dyplot.traceplot(dresults)    
		# Plot the 2-D marginalized posteriors.
		cfig, caxes = dyplot.cornerplot(dresults)
	## add labels!
	# Extract sampling results.
	samples = dresults.samples  # samples
	weights = np.exp(dresults.logwt - dresults.logz[-1])  # normalized weights
	# Compute 10%-90% quantiles.
	quantiles = [dyfunc.quantile(samps, [0.1, 0.9], weights=weights)
				 for samps in samples.T]
	# Compute weighted mean and covariance.
	mean, _ = dyfunc.mean_and_cov(samples, weights)
	
	return mean, quantiles,dresults



def fit_magnetar(times, L_bol,L_bol_err,kappa=0.1, N=5000, **kwargs):
	inputs={'priors':[np.array([1,100]),
					  np.array([0.1,10]),
					  np.array([1,100])]
			,'plot_corner':False
			,'maxiter':100000
			,'maxcall':500000
			,'nlive':250}                            
	inputs.update(kwargs)
	priors=inputs.get('priors')  
	maxiter=inputs.get('maxiter')  
	maxcall=inputs.get('maxcall')  
	nlive=inputs.get('nlive')
	plot_corner=inputs.get('plot_corner')  
  
	def prior_transform(u,priors = priors):
		x=uniform_prior_transform(u,priors =priors )
		return x
	def myloglike(x):
		tau, B14, Pms = x   
		model = LSN_magnetar(times,taum = tau,B14=B14,Pms=Pms,N = N, kappa=kappa)
		c2 = np.sum(chi_sq(L_bol,L_bol_err, model))
		if np.isnan(c2):
			import ipdb; ipdb.set_trace()
		loglike = -c2#log_chi_sq_pdf(c2, len(times)) 
		return loglike 
	dsampler = dynesty.DynamicNestedSampler(myloglike, prior_transform, ndim = 3,nlive=nlive,update_interval=600)
	

	dsampler.run_nested(maxiter=maxiter, maxcall=maxcall)

	dresults = dsampler.results
	if plot_corner:
		# Plot a summary of the run.
		#rfig, raxes = dyplot.runplot(dresults)
		# Plot traces and 1-D marginalized posteriors.
		#tfig, taxes = dyplot.traceplot(dresults)    
		# Plot the 2-D marginalized posteriors.
		cfig, caxes = dyplot.cornerplot(dresults)
	## add labels!
	# Extract sampling results.
	samples = dresults.samples  # samples
	weights = np.exp(dresults.logwt - dresults.logz[-1])  # normalized weights
	# Compute 10%-90% quantiles.
	quantiles = [dyfunc.quantile(samps, [0.1, 0.9], weights=weights)
				 for samps in samples.T]
	# Compute weighted mean and covariance.
	mean, _ = dyfunc.mean_and_cov(samples, weights)
	
	return mean, quantiles,dresults




def fit_Ni_pl(data,kappa=0.1, N=5000, **kwargs):
	inputs={'priors':[np.array([15000,50000]),
					  np.array([10**14,10**15]),
					  np.array([-1.3,0]),
					  np.array([0.5,1]),
					  np.array([-0.5,0.5]),
					  np.array([1,5]),
					  np.array([0,-1.3]),
					  np.array([1,100]),
					  np.array([0.01,5]),
					  np.array([5,500])]
			,'plot_corner':False
			,'maxiter':100000
			,'maxcall':500000
			,'nlive':250
			,'sys_err':0.1
			,'ebv':0
			,'Rv':3.1 
			,'LAW':'MW'}                            
	inputs.update(kwargs)
	priors=inputs.get('priors')  
	maxiter=inputs.get('maxiter')  
	maxcall=inputs.get('maxcall')  
	nlive=inputs.get('nlive')
	plot_corner=inputs.get('plot_corner')  
	sys_err = inputs.get('sys_err')
	ebv = inputs.get('ebv')
	Rv = inputs.get('Rv')
	LAW = inputs.get('LAW')

	def prior_transform(u,priors = priors):
		x=uniform_prior_transform(u,priors =priors )
		return x
	def myloglike(x):
		T0, R0, alpha,beta,t0,t_br,alpha2, taum,Mni,t_gamma = x   
		obj = model_cooling_combined(T0, R0, alpha,beta,t0,t_br,alpha2, taum,Mni,t_gamma, distance  = 3.0856e19, ebv = ebv, Rv = Rv, LAW = LAW)
		logprob, _, _ = obj.likelihood_bb(data, sys_err = sys_err ,nparams = 10)
		return logprob 
	dsampler = dynesty.DynamicNestedSampler(myloglike, prior_transform, ndim = 10,nlive=nlive,update_interval=600)
	

	dsampler.run_nested(maxiter=maxiter, maxcall=maxcall)

	dresults = dsampler.results
	if plot_corner:
		# Plot a summary of the run.
		#rfig, raxes = dyplot.runplot(dresults)
		# Plot traces and 1-D marginalized posteriors.
		#tfig, taxes = dyplot.traceplot(dresults)    
		# Plot the 2-D marginalized posteriors.
		cfig, caxes = dyplot.cornerplot(dresults)
	## add labels!
	# Extract sampling results.
	samples = dresults.samples  # samples
	weights = np.exp(dresults.logwt - dresults.logz[-1])  # normalized weights
	# Compute 10%-90% quantiles.
	quantiles = [dyfunc.quantile(samps, [0.1, 0.9], weights=weights)
				 for samps in samples.T]
	# Compute weighted mean and covariance.
	mean, _ = dyfunc.mean_and_cov(samples, weights)
	
	return mean, quantiles,dresults



def plot_lc_SC_Ni(dat,params,t0, taum,Mni,t_gamma,k34, c_band, lab_band, offset,distance =3.0856e19, ebv = 0, Rv = 3.1,logtmin = -2):
	R13,v85,fM,Menv = params
	obj = model_SC_combined(R13,v85,fM,Menv,t0,k34, taum,Mni,t_gamma, distance =distance, ebv = ebv, Rv = Rv, LAW = LAW)
	logtmax = np.log10(obj.t_tr_day/2)
	time_2 = np.logspace(logtmin,logtmax,30)
	filt_list = np.unique(dat['filter'])
	cond_dic = {}
	for band in filt_list:
		cond_dic[band] = (dat['filter']==band)&(dat['t']>0)
	mags =   obj.mags(time_2,filt_list=filt_list)

	plt.figure(figsize=(6,15))
	for i,band in enumerate(filt_list):
		plt.plot(time_2,mags[band]-offset[band],color =c_band[band] ,alpha = 0.5,ls = '-.',linewidth = 3)

		plt.errorbar(dat['t_rest'][cond_dic[band]]-t0, dat['absmag'][cond_dic[band]]-offset[band],dat['AB_MAG_ERR'][cond_dic[band]],marker = '*', ls = '',color = c_band[band], markersize = 10)
		if np.sign(-offset[band]) == 1:
			string = lab_band[band]+' +{0}'.format(-offset[band])
		elif np.sign(-offset[band]) == -1:
			string = lab_band[band]+' -{0}'.format(-offset[band])
		elif np.sign(-offset[band]) == 0:
			string = lab_band[band]
		if lab_band[band]!='':
			plt.text(-1,np.mean(dat['absmag'][cond_dic[band]]-offset[band]),string, color =c_band[band] ) 
	plt.xlim((-2,7))

	plt.gca().invert_yaxis()
	plt.xlabel('Rest-frame days since estimated explosion',fontsize = 14)
	plt.ylabel('M (AB mag)',fontsize = 14)
	pass



def plot_bb_SC_Ni(Temps,params,t0, taum,Mni,t_gamma,k34, c_band, lab_band, offset,distance =3.0856e19, ebv = 0, Rv = 3.1):
	R13,v85,fM,Menv = params

	obj = model_SC_combined(R13,v85,fM,Menv,t0,k34, taum,Mni,t_gamma, distance =distance, ebv = ebv, Rv = Rv, LAW = LAW)
	logmax = np.max(np.log10(Temps['t_rest']))

	time = np.logspace(-0.5,2,100)
	if hasattr(obj,'t_up'):
		time = np.logspace(np.log10(obj.t_down),np.log10(obj.t_up),100)
	#T_pre = obj.T_evolution(time)
	#R_pre = obj.R_evolution(time)
	#T = np.zeros_like(T_pre)
	#R = np.zeros_like(R_pre)
	#for i in range(len(T_pre)):
	#    t,r = extinct_bb_bias(T_pre[i],R_pre[i],obj.ebv,Rv = obj.Rv,rng = np.array([2000,0.8e4]),z=0,LAW=obj.LAW)
	#    T[i] = t
	#    R[i] = r   
	#  
	T =obj.T_evolution(time)
	R =obj.R_evolution(time)
	plt.figure(figsize=(14,6))
	plt.subplot(1,3,1)
	plt.plot(time,T)

	plt.errorbar(Temps['t_rest']-t0,Temps['T'],np.vstack([Temps['T'] -Temps['T_down'],Temps['T_up']-Temps['T']]),marker = 'o',ls = '', elinewidth=2.5, ecolor= (0.5,0.5,0.5,0.5) , color = 'r', label = r'SN2022oqm')
	plt.yscale('log')
	plt.xscale('log')
	plt.xlabel('rest-frame days from estimated explosion')
	plt.ylabel('T [$^{\circ}K$]')
	plt.xlim(0.4-t0,1.1*np.max(Temps['t_rest']))
	plt.ylim((5000,40000))
	plt.subplot(1,3,2)
	#plt.plot(t,power2(t,1e14,1,3.2e14,0.7,0))
	plt.plot(time,R)

	plt.errorbar(Temps['t_rest']-t0,Temps['R'],np.vstack([Temps['R'] -Temps['R_down'],Temps['R_up']-Temps['R']]),marker = 'o',ls = '', elinewidth=2.5, ecolor= (0.5,0.5,0.5,0.5) , color = 'r', label = r'SN2022oqm')
	plt.xlabel('rest-frame days from estimated explosion')
	plt.ylabel('R [cm]')
	plt.yscale('log')
	plt.xscale('log')
	plt.xlim(0.4-t0,1.1*np.max(Temps['t_rest']))
	plt.ylim((1e14,3e15))

	plt.subplot(1,3,3)

	#plt.plot(t,power2(t,1e14,1,3.2e14,0.7,0))
	t_mid = (Temps['t_rest'][0:-1] + Temps['t_rest'][1:])/2
	v = np.diff(Temps['R'])/np.diff(Temps['t_rest'])/1e5/3600/24
	rerr = np.mean(np.vstack([Temps['R'] -Temps['R_down'],Temps['R_up']-Temps['R']]),axis = 0)
	L = Temps['L']

	Lerr = np.vstack([Temps['L'] -Temps['L_down'],Temps['L_up']-Temps['L']])
	verr = (rerr[0:-1] + rerr[1:])*2/np.diff(Temps['t_rest'])/1e5/3600/24
	plt.plot(time,obj.L_evolution(time))
	#plt.plot(time_2,+Lni)

	#plt.plot(t_mid,v,marker = 'o',ls = '' , color = 'y', label = r'SN2022oqm')
	plt.errorbar(Temps['t_rest']-t0,L,Lerr,marker = 'o',ls = '', elinewidth=2.5, ecolor= (0.5,0.5,0.5,0.5) , color = 'b', label = r'SN2022oqm')

	plt.xlabel('rest-frame days from estimated explosion')
	plt.ylabel('L [erg/s]')
	plt.yscale('log')
	plt.xscale('log')
	plt.xlim(0.4-t0,1.1*np.max(Temps['t_rest']))
	plt.show(block = False)
	pass    



def plot_lc_pl_Ni(dat,params,taum,Mni,t_gamma, c_band, lab_band, offset, kappa = 0.07 ,ebv = 0, Rv = 3.1, LAW = 'Mw' ):
	T0, R0, alpha,beta,t0, t_br, alpha2 = params

	obj = model_cooling_combined(T0, R0, alpha,beta,t0, t_br, alpha2,taum,Mni,t_gamma, distance =3.0856e19, ebv = ebv, Rv = Rv, LAW = LAW)

	filt_list = np.unique(dat['filter'])

	time_2 = np.logspace(-2,np.log10(np.max(dat['t_rest'])),30)
	mags =   obj.mags(time_2,filt_list=filt_list)
	

	cond_dic = {}
	for band in filt_list:
		cond_dic[band] = (dat['filter']==band)&(dat['t']>0)

	plt.figure(figsize=(6,15))
	for i,band in enumerate(filt_list):
		plt.plot(time_2,mags[band]-offset[band],color =c_band[band] ,alpha = 0.5,ls = '-.',linewidth = 3)

		plt.errorbar(dat['t_rest'][cond_dic[band]]-t0, dat['absmag'][cond_dic[band]]-offset[band],dat['AB_MAG_ERR'][cond_dic[band]],marker = '*', ls = '',color = c_band[band], markersize = 10)
		if np.sign(-offset[band]) == 1:
			string = lab_band[band]+' +{0}'.format(-offset[band])
		elif np.sign(-offset[band]) == -1:
			string = lab_band[band]+' -{0}'.format(-offset[band])
		elif np.sign(-offset[band]) == 0:
			string = lab_band[band]
		if lab_band[band]!='':
			plt.text(-1.5,np.mean(dat['absmag'][cond_dic[band]]-offset[band]),string, color =c_band[band] ) 
	plt.xlim((-2,1.1*np.max(dat['t_rest'])))

	plt.gca().invert_yaxis()
	plt.xlabel('Rest-frame days since estimated explosion',fontsize = 14)
	plt.ylabel('M (AB mag)',fontsize = 14)
	pass


def plot_resid_pl_Ni(dat,params,taum,Mni,t_gamma, c_band, lab_band, offset, kappa = 0.07 ,ebv = 0, Rv = 3.1, LAW = 'Mw' , sigma = False):
	T0, R0, alpha,beta,t0, t_br, alpha2 = params

	obj = model_cooling_combined(T0, R0, alpha,beta,t0, t_br, alpha2,taum,Mni,t_gamma, distance =3.0856e19, ebv = ebv, Rv = Rv, LAW = LAW)

	filt_list = np.unique(dat['filter'])

	time_2 = np.logspace(-2,np.log10(np.max(dat['t_rest'])),30)
	mags =   obj.mags(time_2,filt_list=filt_list)
	


	cond_dic = {}
	resid = {}

	# change to model_cooling_broken

	for i,band in enumerate(filt_list):

		cond_dic[band] = (dat['filter']==band)

		t = dat['t_rest'][cond_dic[band]]-t0
		mag = dat['absmag'][cond_dic[band]]
		mags =   obj.mags(t,filt_list=[band])
		res = mag - mags[band]
		if sigma: 
			magerr = dat['AB_MAG_ERR'][cond_dic[band]]
			res = res/magerr
		resid[band] = res
	

	plt.figure(figsize=(15,6))
	for i,band in enumerate(filt_list):
		if ~sigma:     
			plt.errorbar(dat['t_rest'][cond_dic[band]]-t0, resid[band],dat['AB_MAG_ERR'][cond_dic[band]],marker = '*', ls = '',color = c_band[band], markersize = 10, label = lab_band[band])
		else: 
			plt.plot(dat['t_rest'][cond_dic[band]]-t0, resid[band],marker = '*', ls = '',color = c_band[band], markersize = 10, label = lab_band[band])
	plt.plot([0.1,np.max(dat['t_rest'])],[0,0],'k--')
	#plt.xlim((-2,1.1*np.max(dat['t_rest'])))
	plt.xscale('log')
	plt.gca().invert_yaxis()
	plt.xlabel('Rest-frame days since estimated explosion',fontsize = 14)
	plt.ylabel('M - model (AB mag)')
	if sigma: 
		plt.ylabel('M - model ($\sigma$)')
	plt.legend()
	pass



def plot_lc_mod(dat,params, c_band, lab_band, offset):
	T0,R0,alpha,beta,t0,t_br,alpha2, beta2, ebv, Rv = params 

	T2 = T0*(t_br)**(alpha)/((t_br)**(alpha2))
	R2 = R0*(t_br)**(beta)/((t_br)**(beta2))
	time_2 = np.logspace(-2,np.log10(t_br),30)
	time_3 = np.logspace(np.log10(t_br),np.log10(np.max(dat['t_rest'])),30)
	filt_list = np.unique(dat['filter'])
	cond_dic = {}
	for band in filt_list:
		cond_dic[band] = (dat['filter']==band)&(dat['t']>0)
	obj = model_cooling(T0, R0, alpha,beta,t0, distance =3.0856e19, ebv = ebv, Rv = Rv, LAW = LAW)
	mags =   obj.mags(time_2,filt_list=filt_list)
	obj2 = model_cooling(T2, R2, alpha2,beta2,t0, distance =3.0856e19, ebv = ebv, Rv = Rv, LAW = LAW)
	mags2 =   obj2.mags(time_3,filt_list=filt_list)
	plt.figure(figsize=(6,15))
	for i,band in enumerate(filt_list):
		plt.plot(time_2,mags[band]-offset[band],color =c_band[band] ,alpha = 0.5,ls = '-.',linewidth = 3)
		plt.plot(time_3,mags2[band]-offset[band],color =c_band[band] ,alpha = 0.5,ls = '-.',linewidth = 3)

		plt.errorbar(dat['t_rest'][cond_dic[band]]-t0, dat['absmag'][cond_dic[band]]-offset[band],dat['AB_MAG_ERR'][cond_dic[band]],marker = '*', ls = '',color = c_band[band], markersize = 10)
		if np.sign(-offset[band]) == 1:
			string = lab_band[band]+' +{0}'.format(-offset[band])
		elif np.sign(-offset[band]) == -1:
			string = lab_band[band]+' -{0}'.format(-offset[band])
		elif np.sign(-offset[band]) == 0:
			string = lab_band[band]
		if lab_band[band]!='':
			plt.text(-1.5,np.mean(dat['absmag'][cond_dic[band]]-offset[band]),string, color =c_band[band] ) 
	plt.xlim((-2,1.1*np.max(dat['t_rest'])))

	plt.gca().invert_yaxis()
	plt.xlabel('Rest-frame days since estimated explosion',fontsize = 14)
	plt.ylabel('M (AB mag)',fontsize = 14)
	pass


def plot_lc_mod2(dat,params, c_band, lab_band, offset):
	T0,R0,alpha,beta,t0,t_br,t_br2,alpha2, beta2, ebv, Rv = params 


	time_2 = np.logspace(-2,np.log10(np.max(dat['t_rest'])),60)
	filt_list = np.unique(dat['filter'])
	cond_dic = {}
	for band in filt_list:
		cond_dic[band] = (dat['filter']==band)&(dat['t']>0)
	obj = model_cooling_broken(T0, R0, alpha,beta,t0, t_br, t_br, alpha2,beta2, distance =3.0856e19, ebv = ebv, Rv = Rv, LAW = LAW)
	mags =   obj.mags(time_2,filt_list=filt_list)

	plt.figure(figsize=(6,12))
	for i,band in enumerate(filt_list):
		plt.plot(time_2,mags[band]-offset[band],color =c_band[band] ,alpha = 0.5,ls = '-.',linewidth = 3)

		plt.errorbar(dat['t_rest'][cond_dic[band]]-t0, dat['absmag'][cond_dic[band]]-offset[band],dat['AB_MAG_ERR'][cond_dic[band]],marker = '*', ls = '',color = c_band[band], markersize = 10)
		if np.sign(-offset[band]) == 1:
			string = lab_band[band]+' +{0}'.format(-offset[band])
		elif np.sign(-offset[band]) == -1:
			string = lab_band[band]+' -{0}'.format(-offset[band])
		elif np.sign(-offset[band]) == 0:
			string = lab_band[band]
		if lab_band[band]!='':
			plt.text(-1.2,np.mean(dat['absmag'][cond_dic[band]&(dat['t_rest']<4)]-offset[band]),string, color =c_band[band] )    
	plt.xlim((-2,1.1*np.max(dat['t_rest'])))

	plt.gca().invert_yaxis()
	plt.xlabel('Rest-frame days since estimated explosion',fontsize = 14)
	plt.ylabel('M (AB mag)',fontsize = 14)
	pass


def plot_resid_mod(dat,params, c_band, lab_band):
	T0,R0,alpha,beta,t0,t_br,alpha2, beta2, ebv, Rv = params 


	filt_list = np.unique(dat['filter'])
	cond_dic = {}
	resid = {}
	for band in filt_list:
		cond_dic[band] = (dat['filter']==band)&(dat['t']>0)

	# change to model_cooling_broken

	for i,band in enumerate(filt_list):

		obj = model_cooling_broken(T0, R0, alpha,beta,t0, t_br, t_br, alpha2,beta2, distance =3.0856e19, ebv = ebv, Rv = Rv, LAW = LAW)

		t = dat['t_rest'][cond_dic[band]]-t0
		mag = dat['absmag'][cond_dic[band]]
		mags =   obj.mags(t,filt_list=[band])
		res = mag - mags[band]
		resid[band] = res
	

	plt.figure(figsize=(15,6))
	for i,band in enumerate(filt_list):

		plt.errorbar(dat['t_rest'][cond_dic[band]]-t0, resid[band],dat['AB_MAG_ERR'][cond_dic[band]],marker = '*', ls = '',color = c_band[band], markersize = 10, label = lab_band[band])
	plt.plot([0.1,np.max(dat['t_rest'])],[0,0],'k--')
	#plt.xlim((-2,1.1*np.max(dat['t_rest'])))
	plt.xscale('log')
	plt.gca().invert_yaxis()
	plt.xlabel('Rest-frame days since estimated explosion',fontsize = 14)
	plt.ylabel('M - model (AB mag)')
	plt.legend()
	pass



def plot_resid_mod2(dat,params, c_band, lab_band, sigma = False):
	T0,R0,alpha,beta,t0,t_br,t_br2,alpha2, beta2, ebv, Rv = params 


	filt_list = np.unique(dat['filter'])
	cond_dic = {}
	resid = {}
	for band in filt_list:
		cond_dic[band] = (dat['filter']==band)&(dat['t']>0)

	# change to model_cooling_broken

	for i,band in enumerate(filt_list):

		obj = model_cooling_broken(T0, R0, alpha,beta,t0, t_br, t_br2, alpha2,beta2, distance =3.0856e19, ebv = ebv, Rv = Rv, LAW = LAW)

		t = dat['t_rest'][cond_dic[band]]-t0
		mag = dat['absmag'][cond_dic[band]]
		mags =   obj.mags(t,filt_list=[band])
		res = mag - mags[band]
		if sigma: 
			magerr = dat['AB_MAG_ERR'][cond_dic[band]]
			res = res/magerr
		resid[band] = res
	

	plt.figure(figsize=(15,6))
	for i,band in enumerate(filt_list):
		if ~sigma:     
			plt.errorbar(dat['t_rest'][cond_dic[band]]-t0, resid[band],dat['AB_MAG_ERR'][cond_dic[band]],marker = '*', ls = '',color = c_band[band], markersize = 10, label = lab_band[band])
		else: 
			plt.plot(dat['t_rest'][cond_dic[band]]-t0, resid[band],marker = '*', ls = '',color = c_band[band], markersize = 10, label = lab_band[band])
	plt.plot([0.1,np.max(dat['t_rest'])],[0,0],'k--')
	#plt.xlim((-2,1.1*np.max(dat['t_rest'])))
	plt.xscale('log')
	plt.gca().invert_yaxis()
	plt.xlabel('Rest-frame days since estimated explosion',fontsize = 14)
	plt.ylabel('M - model (AB mag)')
	if sigma: 
		plt.ylabel('M - model ($\sigma$)')
	plt.legend()
	pass


def plot_resid(dat,obj, c_band, lab_band, sigma = False,fig = 'create',ax = None):
	filt_list = np.unique(dat['filter'])
	cond_dic = {}
	resid = {}
	for band in filt_list:
		cond_dic[band] = (dat['filter']==band)&(dat['t']>0)

	# change to model_cooling_broken

	for i,band in enumerate(filt_list):
		t = dat['t_rest'][cond_dic[band]]-obj.t0
		mag = dat['absmag'][cond_dic[band]]
		mags =   obj.mags(t,filt_list=[band])
		res = mag - mags[band]
		if sigma: 
			magerr = dat['AB_MAG_ERR'][cond_dic[band]]
			res = res/magerr
		resid[band] = res
	
	if fig == 'create': 
		plt.figure(figsize=(15,6))
		ax = plt.axes()
	
	for i,band in enumerate(filt_list):
		if ~sigma:     
			ax.errorbar(dat['t_rest'][cond_dic[band]]-obj.t0, resid[band],dat['AB_MAG_ERR'][cond_dic[band]],marker = '*', ls = '',color = c_band[band], markersize = 10, label = lab_band[band])
		else: 
			ax.plot(dat['t_rest'][cond_dic[band]]-tobj.t0, resid[band],marker = '*', ls = '',color = c_band[band], markersize = 10, label = lab_band[band])
	plt.plot([0.1,np.max(dat['t_rest'])],[0,0],'k--')
	#plt.xlim((-2,1.1*np.max(dat['t_rest'])))
	plt.xscale('log')
	plt.gca().invert_yaxis()
	plt.xlabel('Rest-frame days since estimated explosion',fontsize = 14)
	plt.ylabel('M - model (AB mag)')
	if sigma: 
		plt.ylabel('M - model ($\sigma$)')
	plt.legend()
	return fig,ax


def plot_lc_with_model(dat,obj,t0, c_band, lab_band, offset, fig = 'create', ax = None,xlab_pos = -1.5):

	time_2 = np.logspace(-2,np.log10(np.max(dat['t_rest'])),30)
	if hasattr(obj,'t_up'):
		t_down = max(obj.t_down,0.001)
		time_2 = np.logspace(np.log10(t_down),np.log10(obj.t_up),30)
	if fig == 'create':
		fig = plt.figure(figsize=(6,15))
		ax = plt.axes()
	filt_list = np.unique(dat['filter'])
	cond_dic = {}
	for band in filt_list:
		cond_dic[band] = (dat['filter']==band)&(dat['t']>0)
	mags =   obj.mags(time_2,filt_list=filt_list)

	for i,band in enumerate(filt_list):
		try:
			ax.plot(time_2,mags[band]-offset[band],color =c_band[band] ,alpha = 0.5,ls = '-.',linewidth = 3, label = '')
		except:
			import ipdb;ipdb.set_trace()

		ax.errorbar(dat['t_rest'][cond_dic[band]]-t0, dat['absmag'][cond_dic[band]]-offset[band],dat['AB_MAG_ERR'][cond_dic[band]],marker = '*', ls = '',color = c_band[band], markersize = 10, label = '')
		if np.sign(-offset[band]) == 1:
			string = lab_band[band]+' +{0}'.format(-offset[band])
		elif np.sign(-offset[band]) == -1:
			string = lab_band[band]+' -{0}'.format(-offset[band])
		elif np.sign(-offset[band]) == 0:
			string = lab_band[band]
		if lab_band[band]!='':
			ax.text(xlab_pos,np.mean(mags[band]-offset[band]),string, color =c_band[band] ) 
			
	ax.set_xlim((-2,1.1*np.max(time_2)))

	ax.invert_yaxis()
	ax.set_xlabel('Rest-frame days since estimated explosion',fontsize = 14)
	ax.set_ylabel('M (AB mag) + offset',fontsize = 14)
	return fig,ax


def plot_lc_2model(dat, obj1, obj2, c_band, lab_band, offset, fig = 'create', ax = None):

	time_2 = np.logspace(-2,np.log10(np.max(dat['t_rest'])),30)
	time_2p=  time_2

	if hasattr(obj1,'t_up'):
		time_2 = np.logspace(np.log10(obj1.t_down),np.log10(obj1.t_up),30)
	if hasattr(obj2,'t_up'):
		time_2p = np.logspace(np.log10(obj2.t_down),np.log10(obj2.t_up),30)

	if fig == 'create':
		fig = plt.figure(figsize=(6,15))
		ax = plt.axes()
	filt_list = np.unique(dat['filter'])
	cond_dic = {}
	for band in filt_list:
		cond_dic[band] = (dat['filter']==band)&(dat['t']>0)
	mags =   obj1.mags(time_2,filt_list=filt_list)
	mags2 =   obj2.mags(time_2p,filt_list=filt_list)

	for i,band in enumerate(filt_list):
		ax.plot(time_2+obj1.t0,mags[band]-offset[band],color =c_band[band] ,alpha = 0.5,ls = '-.',linewidth = 3, label = '')
		ax.plot(time_2p+obj2.t0,mags2[band]-offset[band],color =c_band[band] ,alpha = 0.5,ls = '-',linewidth = 3, label = '')
		import ipdb; ipdb.set_trace()
		ax.errorbar(dat['t_rest'][cond_dic[band]], dat['absmag'][cond_dic[band]]-offset[band],dat['AB_MAG_ERR'][cond_dic[band]],marker = '*', ls = '',color = c_band[band], markersize = 10, label = '')
		if np.sign(-offset[band]) == 1:
			string = lab_band[band]+' +{0}'.format(-offset[band])
		elif np.sign(-offset[band]) == -1:
			string = lab_band[band]+' -{0}'.format(-offset[band])
		elif np.sign(-offset[band]) == 0:
			string = lab_band[band]
		if lab_band[band]!='':
			ax.text(-1.5,np.mean(dat['absmag'][cond_dic[band]]-offset[band]),string, color =c_band[band] ) 
			
	ax.set_xlim((-2,1.1*np.max(time_2)))

	ax.invert_yaxis()
	ax.set_xlabel('Rest-frame days since estimated explosion',fontsize = 14)
	ax.set_ylabel('M (AB mag) + offset',fontsize = 14)
	return fig,ax


def plot_lc_model_only(obj,filt_list,tmin,tmax,t0, c_band, lab_band,offset,validity = False, fig = 'create', ax = None,**kwargs):
	if fig == 'create':
		fig = plt.figure(figsize=(6,15))
		ax = plt.axes()
		print('create figure')

	time_2 = np.logspace(np.log10(tmin+t0),np.log10(tmax+t0),30)

	if validity:
		time_2 = np.logspace(np.log10(obj.t_down),np.log10(obj.t_up),30)
	mags =   obj.mags(time_2,filt_list=filt_list)
	for i,band in enumerate(filt_list):
		ax.plot(time_2+t0,mags[band]-offset[band],color =c_band[band],label = lab_band[band],**kwargs)
		
	#ax.set_xlim((0.01,1.1*np.max(time_2)))
	if fig == 'create':
		plt.gca().invert_yaxis()
	plt.xlabel('Rest-frame days since estimated explosion',fontsize = 14)
	plt.ylabel('M (AB mag) + offset',fontsize = 14)
	#plt.xscale('log')
	plt.legend()
	return fig,ax



@njit
def  L_shock_cooling_Piro2021(ts, Me, Re, E51, kappa=0.07, n=10, delta = 1.1):
	'''
	Luminosity of shock cooling emission from extended material  
	Piro et al. (2021) https://ui.adsabs.harvard.edu/abs/2021ApJ...909..209P/abstract
	
	INPUTS:
		Me: mass of the extended material in unit of solar mass
		Re: radius of the extended material in unit of solar mass
		E51: energy gained from the shock passing through the extended material in unit of 1e51 erg
		ts: time in days
		kappa: assume a constant opacity
		n and delta: the radial dependence of the outer and inner density; the default values are adopted in Piro et al. 2021 with reference given to Chevalier & Soker 1989
		
	The inputs are converted into cgs units before the calculation 
	'''
	c = 29979245800
	ts = ts * 24*3600
	E = E51 * 1e51    
	M = Me * 1.988409870698051e+33
	R = Re * 69570000000
	
  
	K = (n-3) * (3-delta) / (4 * np.pi * (n-delta)) # K = 0.119 for default n and delta
	vt = ( (n-5)*(5-delta) / ((n-3)*(3-delta)) )**0.5 * (2*E/M)**0.5 #the transition velocity between the outer and inner regions
	td = ( 3*kappa*K*M / ((n-1)*vt*c) )**0.5 # the time at which the diffusion reaches the depth where the velocity is vt   
	
	prefactor = np.pi*(n-1)/(3*(n-5)) * c*R*vt**2 / kappa 
	L1 = prefactor * (td/ts)**(4/(n-2))
	L2 = prefactor * np.exp(-0.5 * ((ts/td)**2 - 1)) 
	Ls = np.zeros(len(ts))
	ix1 = ts < td
	Ls[ix1] = L1[ix1]
	Ls[~ix1] = L2[~ix1]
	return td/3600/24, Ls


@njit
def Rph_shock_cooling_Piro2021(ts, Me,Re, E51, kappa=0.07, n=10, delta=1.1):
	'''
	phot
	'''
	c = 29979245800
	ts = ts * 24*3600
	E = E51 * 1e51    
	M = Me * 1.988409870698051e+33
	R = Re * 69570000000
	
	
	K = (n-3) * (3-delta) / (4 * np.pi * (n-delta)) # K = 0.119 for default n and delta
	
	vt = ( (n-5)*(5-delta) / ((n-3)*(3-delta)) )**0.5 * (2*E/M)**0.5 #the transition velocity between the outer and inner regions    
	tph = (3*kappa*K*M/(2*(n-1)*vt**2))**0.5
	
	rph1 = (tph/ts)**(2/(n-1))*vt*ts
	rph2 = ((delta-1)/(n-1)*(ts**2/tph*2-1)+1)**(-1/(delta-1))*vt*ts
	
	Rphs = np.zeros(len(ts))
	ix1 = ts < tph
	Rphs[ix1] = rph1[ix1]
	Rphs[~ix1] = rph2[~ix1]
	return tph/3600/24, Rphs


@njit
def Tph_shock_cooling_Piro2021(ts, Me,Re, E51, kappa=0.07, n=10, delta=1.1):
	'''
	phot
	'''
	_,Rph = Rph_shock_cooling_Piro2021(ts, Me,Re, E51, kappa=kappa, n=n, delta=delta)
	
	_,Lph = L_shock_cooling_Piro2021(ts, Me,Re, E51, kappa=kappa, n=n, delta=delta)
	Tph = (Lph/(4*np.pi*sigma_sb*Rph**2))**0.25
	return Tph

class model_piro_cooling(object):
	def __init__(self, Me,Re, E51,t0, kappa=0.07, n=10, delta=1.1  ,distance = 3.08e26,ebv = 0,Rv = 3.1, LAW = 'MW'):
		self.ebv = ebv
		self.Rv = Rv
		self.distance = distance
		self.LAW  = LAW
		self.t0 = t0
		self.Me = Me
		self.Re = Re
		self.E51 = E51
		self.kappa = kappa
		self.n = n
		self.delta = delta


	def R_evolution(self,time): 
		time = time.flatten()
		R = Rph_shock_cooling_Piro2021(time, self.Me, self.Re, self.E51, kappa=self.kappa, n=self.n, delta = self.delta)[1]
		return R
	def L_evolution(self, time):
		time = time.flatten()
		L_evo = L_shock_cooling_Piro2021(time, self.Me, self.Re, self.E51, kappa=self.kappa, n=self.n, delta = self.delta)[1]
		return L_evo     
	def T_evolution(self,time): 
		time = time.flatten()

		#R_evo = self.R_evolution(time)
		#L_evo = self.L_evolution(time) 
		#T = (L_evo/(4*np.pi*sigma_sb*R_evo**2))**0.25       
		T = Tph_shock_cooling_Piro2021(time, self.Me, self.Re, self.E51, kappa=self.kappa, n=self.n, delta = self.delta)
		return T
	def mags(self,time,filt_list = FILTERS): 
		m_array={}
		for filt in filt_list:
			R_evo = self.R_evolution(time)
			T_evo = self.T_evolution(time)
			mag = generate_cooling_mag(T_evo, R_evo, filt,d = self.distance, Rv=self.Rv, EBV = self.ebv,EBV_MW = 0, LAW = self.LAW)
			m_array[filt] = mag
		return m_array    
	def likelihood(self,dat,sys_err = 0.05,nparams = 5):
		t0 = self.t0 
		ebv = self.ebv
		Rv = self.Rv
		LAW = self.LAW
		d = self.distance
		filt_list = np.unique(dat['filter'])
		chi2_dic = {}
		dof = 0
		chi_tot = 0
		N_band = []
		for filt in filt_list:
			dat_filt = dat[dat['filter']==filt]
			time = dat_filt['t_rest']-t0
			R_evo = self.R_evolution(time)
			T_evo = self.T_evolution(time)
			mag = generate_cooling_mag(T_evo, R_evo, filt,d = d, Rv=Rv, EBV = ebv,EBV_MW = 0, LAW = LAW)
			
			mag_obs = dat_filt['absmag']
			mag_err = dat_filt['AB_MAG_ERR'] + sys_err
			
			c2 = chi_sq(mag_obs,mag_err, mag)
			N = len(c2)
			#N_band.append(N)

			chi2_dic[filt] = c2#*N**2


			dof = dof+N

			chi_tot = chi_tot +np.sum(chi2_dic[filt])
		#N_band = np.array(N_band) 
		chi_tot = chi_tot#/np.sum(N_band**2)
		dof = dof - nparams
		
		rchi2 = chi_tot/dof
			
		
		if chi_tot == 0:
			logprob = -np.inf
		elif dof <= 0:
			logprob = -np.inf
		else:
			#prob = scipy.stats.chi2.pdf(chi_tot, dof)
			#prob = float(prob)
			
			logprob = log_chi_sq_pdf(chi_tot, dof)


		#logprob = np.log(prob)
		return logprob, chi_tot, dof

def likelihood_piro_cooling(data, Me,Re, E51,t0, kappa=0.07, n=10, delta=1.1,distance = 3.0856e19,ebv = 0,Rv = 3.1 , LAW = 'MW',sys_err = 0.05,nparams = 4):
	obj = model_piro_cooling(Me,Re, E51,t0, kappa=kappa, n=n, delta=delta, distance =distance, ebv = ebv, Rv = Rv, LAW = LAW)
	t1 = Rph_shock_cooling_Piro2021(np.array([1]), Me,Re, E51, kappa=kappa, n=n, delta=delta)[0]
	data = data[data['t_rest']<0.95*t1+t0]


	logprob, chi_tot, dof = obj.likelihood(data, sys_err = sys_err ,nparams = nparams)


	return logprob, chi_tot, dof 


def fit_piro_cooling(data, plot_corner = True,sys_err = 0.05, LAW = 'MW',**kwargs): 
	inputs={'priors':[np.array([0.1,10]),
					  np.array([100,2000]),
					  np.array([0.1,10]),
					  np.array([-1,0])]
			,'maxiter':100000
			,'maxcall':500000
			,'nlive':250}                            
	inputs.update(kwargs)
	priors=inputs.get('priors')  
	maxiter=inputs.get('maxiter')  
	maxcall=inputs.get('maxcall')  
	nlive=inputs.get('nlive')  
   
	def prior_transform(u,priors = priors):
		x=uniform_prior_transform(u,priors =priors )
		return x
	def myloglike(x):
		Me,Re, E51,t0 = x   
		loglike = likelihood_piro_cooling(data,Me,Re, E51,t0,n=10,ebv = 0, Rv = 3.1, sys_err = sys_err, LAW = LAW)[0]
		return loglike  
	dsampler = dynesty.DynamicNestedSampler(myloglike, prior_transform, ndim = 4,nlive=nlive,update_interval=600)
	dsampler.run_nested(maxiter=maxiter, maxcall=maxcall)
	dresults = dsampler.results
	if plot_corner:
		# Plot a summary of the run.
		#rfig, raxes = dyplot.runplot(dresults)
		# Plot traces and 1-D marginalized posteriors.
		#tfig, taxes = dyplot.traceplot(dresults)    
		cfig, caxes = dyplot.cornerplot(dresults,labels=['M_e','R_e',r'$E_{51}$',r'$t_{exp}$']
								,label_kwargs={'fontsize':14}, color = '#0042CF',show_titles = True)    
	## add labels!
	# Extract sampling results.
	samples = dresults.samples  # samples
	weights = np.exp(dresults.logwt - dresults.logz[-1])  # normalized weights
	# Compute 10%-90% quantiles.
	quantiles = [dyfunc.quantile(samps, [0.1, 0.9], weights=weights)
				 for samps in samples.T]
	# Compute weighted mean and covariance.
	mean, _ = dyfunc.mean_and_cov(samples, weights)
	
	return mean, quantiles,dresults

def plot_lc_piro(dat,params, c_band, lab_band, offset, kappa = 0.07 ,ebv = 0, Rv = 3.1, LAW = 'Mw' ):
	if len(params)==4:
		Me,Re, E51,t0 = params
		n=10
		delta=1.1 
	elif len(params)==5:
		Me,Re, E51,t0,n = params 
		delta=1.1 
	elif len(params)==6:
		Me,Re, E51,t0,n,delta = params 

	obj = model_piro_cooling(Me,Re, E51,t0, kappa=kappa, n=n, delta=delta, distance =3.0856e19, ebv = ebv, Rv = Rv, LAW = LAW)
	t1 = Rph_shock_cooling_Piro2021(np.array([1]), Me,Re, E51, kappa=kappa, n=n, delta=delta)[0]

	filt_list = np.unique(dat['filter'])

	time_2 = np.logspace(-2,np.log10(0.95*t1),30)
	mags =   obj.mags(time_2,filt_list=filt_list)
	

	cond_dic = {}
	for band in filt_list:
		cond_dic[band] = (dat['filter']==band)&(dat['t']>0)

	plt.figure(figsize=(6,15))
	for i,band in enumerate(filt_list):
		plt.plot(time_2,mags[band]-offset[band],color =c_band[band] ,alpha = 0.5,ls = '-.',linewidth = 3)

		plt.errorbar(dat['t_rest'][cond_dic[band]]-t0, dat['absmag'][cond_dic[band]]-offset[band],dat['AB_MAG_ERR'][cond_dic[band]],marker = '*', ls = '',color = c_band[band], markersize = 10)
		if np.sign(-offset[band]) == 1:
			string = lab_band[band]+' +{0}'.format(-offset[band])
		elif np.sign(-offset[band]) == -1:
			string = lab_band[band]+' -{0}'.format(-offset[band])
		elif np.sign(-offset[band]) == 0:
			string = lab_band[band]
		if lab_band[band]!='':
			plt.text(-1.5,np.mean(dat['absmag'][cond_dic[band]]-offset[band]),string, color =c_band[band] ) 
	plt.xlim((-2,1.1*np.max(dat['t_rest'])))

	plt.gca().invert_yaxis()
	plt.xlabel('Rest-frame days since estimated explosion',fontsize = 14)
	plt.ylabel('M (AB mag)',fontsize = 14)
	pass


def plot_resid_piro(dat,params, c_band, lab_band, offset, kappa = 0.07 ,ebv = 0, Rv = 3.1, LAW = 'Mw', sigma = False):
	if len(params)==4:
		Me,Re, E51,t0 = params
		n=10
		delta=1.1 
	elif len(params)==5:
		Me,Re, E51,t0,n = params 
		delta=1.1 
	elif len(params)==6:
		Me,Re, E51,t0,n,delta = params 

	obj = model_piro_cooling(Me,Re, E51,t0, kappa=kappa, n=n, delta=delta, distance =3.0856e19, ebv = ebv, Rv = Rv, LAW = LAW)
	t1 = Rph_shock_cooling_Piro2021(np.array([1]), Me,Re, E51, kappa=kappa, n=n, delta=delta)[0]

	filt_list = np.unique(dat['filter'])


	cond_dic = {}
	resid = {}

	# change to model_cooling_broken

	for i,band in enumerate(filt_list):

		obj = model_piro_cooling(Me,Re, E51,t0, kappa=kappa, n=n, delta=delta, distance =3.0856e19, ebv = ebv, Rv = Rv, LAW = LAW)
		cond_dic[band] = (dat['filter']==band)&(dat['t']>0)&(dat['t']<0.95*t1)

		t = dat['t_rest'][cond_dic[band]]-t0
		mag = dat['absmag'][cond_dic[band]]
		mags =   obj.mags(t,filt_list=[band])
		res = mag - mags[band]
		if sigma: 
			magerr = dat['AB_MAG_ERR'][cond_dic[band]]
			res = res/magerr
		resid[band] = res
	

	plt.figure(figsize=(15,6))
	for i,band in enumerate(filt_list):
		if ~sigma:     
			plt.errorbar(dat['t_rest'][cond_dic[band]]-t0, resid[band],dat['AB_MAG_ERR'][cond_dic[band]],marker = '*', ls = '',color = c_band[band], markersize = 10, label = lab_band[band])
		else: 
			plt.plot(dat['t_rest'][cond_dic[band]]-t0, resid[band],marker = '*', ls = '',color = c_band[band], markersize = 10, label = lab_band[band])
	plt.plot([0.1,np.max(dat['t_rest'])],[0,0],'k--')
	#plt.xlim((-2,1.1*np.max(dat['t_rest'])))
	plt.xscale('log')
	plt.gca().invert_yaxis()
	plt.xlabel('Rest-frame days since estimated explosion',fontsize = 14)
	plt.ylabel('M - model (AB mag)')
	if sigma: 
		plt.ylabel('M - model ($\sigma$)')
	plt.legend()
	pass





