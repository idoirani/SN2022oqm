from astropy.io import ascii
import os 
import numpy as np 
import pandas
import glob
import matplotlib.pyplot as plt
import sys 
sys.path.insert(1,'./')
from params import * 
from PhotoUtils import *
from blackbody_tools import *

from Ni_tools import *
plt.rcParams.update({
  "text.usetex": True,
})




data = ascii.read(path_data)

data[Date_col].name = 'jd'
data[absmag_col].name = 'absmag'
data[M_err_col].name = 'AB_MAG_ERR'
data[filter_col].name = 'filter'
data[flux_col].name = 'flux'
data[fluxerr_col].name = 'fluxerr'
data[piv_wl_col].name = 'piv_wl'
data[inst_col].name = 'instrument'
path_dat_mod = path_out+'/{0}_data.csv'.format(sn)
ascii.write(data[Date_col,filter_col,absmag_col,M_err_col,flux_col,fluxerr_col,piv_wl_col,inst_col],path_dat_mod,overwrite=True)
data['t'] = data['jd'] - t0_init
data['t_rest'] = data['t']/(1+z)
data.sort('t_rest')

dat = data.copy()
cmd = 'python ./fit_bb_extinction.py --data "{2}" --out_path "{3}" --write True --plots {8} --name {0} --EBV_host {1} --z {4} --d_mpc {5} --dates "{6}" --t0 {7: .2f} --path_params {9}'.format(sn,ebv_host_init,path_dat_mod,path_bb_out,z,d_mpc,path_dates_sn,t0_init,False, './params.py')
os.system(cmd)
Temps = ascii.read(path_bb_out)


day = 24*3600







########################################################################
####################combined  Ni56 and power law fit####################
########################################################################



cond = Temps['t_rest']>7

mean_Ni2, quantiles_Ni2,dresults_Ni2 = fit_Ni_pl(Temps[cond],sys_err = 0.25, priors = [np.array([15000,30000]),
                                                                          np.array([10**14,5*10**14]),
                                                                          np.array([-1.3,0]),
                                                                          np.array([0.8,1]),
                                                                          np.array([-0.5,0.5]),
                                                                          np.array([1,5]),
                                                                          np.array([-1.3,0]),
                                                                          np.array([2,30]),
                                                                          np.array([0.01,0.5]),
                                                                          np.array([1,100])])  
time = np.logspace(-0.5,2,100)

dyplot.cornerplot(dresults_Ni2,labels=['$T_{0}$','$R_{0}$',r'$\alpha$',r'$\beta$',r'$t_{exp}$',r'$t_{br}$',r'$\alpha_{2}$',r'$\tau_{m}$',r'$M_{Ni^{56}}$',r'$t_{\gamma}$']
                                ,label_kwargs={'fontsize':14}, color = '#0042CF',show_titles = True)

samples_Ni2 = dresults_Ni2.samples  # samples
weights_Ni2 = np.exp(dresults_Ni2.logwt - dresults_Ni2.logz[-1])
best_Ni2 =  samples_Ni2[weights_Ni2 == np.max(weights_Ni2)][0]

T0_2, R0_2, alpha_2,beta_2,t0_2,t_br_2,alpha2_2, taum_2,Mni_2,t_gamma_2 = best_Ni2  
obj = model_cooling_combined(T0_2, R0_2, alpha_2,beta_2,t0_2,t_br_2,alpha2_2, taum_2,Mni_2,t_gamma_2, distance  = 3.0856e19, ebv = 0, Rv = 3.1, LAW = 'MW')
       
plot_bb_with_model(obj,Temps,t0_2)



########################################################################################
#################################### Light curve power law fits ########################
########################################################################################


max_t = 5


priors =             [np.array([15000,80000]),
					  np.array([10**14,10**15]),
					  np.array([-1.3,0]),
					  np.array([0.5,1]),
					  np.array([-0.5,np.min(dat['t_rest'])]),
                      np.array([2,min(5,max_t-1)]),
                      np.array([2,min(5,max_t-1)]),
					  np.array([-1.3,0]),
					  np.array([0.5,1])]

cond = (dat['t']<max_t)&(dat['t']>0)&(dat['filter']!='v_swift')&(dat['filter']!='b_swift')
mean, quantiles,dresults = fit_broken_powerlaw2(dat[cond],priors=priors,sys_err = 0.1, LAW = 'MW',plot_corner=False)


plt.figure(figsize=(15,15))
cfig, caxes = dyplot.cornerplot(dresults,labels=['$T_{0}$','$R_{0}$',r'$\alpha$',r'$\beta$',r'$t_{exp}$',r'$t_{br}$',r'$t_{br,2}$',r'$\alpha_{2}$',r'$\beta_{2}$']
                                ,label_kwargs={'fontsize':14}, color = '#0042CF',show_titles = True)
samples = dresults.samples  # samples
weights = np.exp(dresults.logwt - dresults.logz[-1])  # normalized weights
# Compute 10%-90% quantiles.
quantiles = [dyfunc.quantile(samps, [0.025,0.5, 0.975], weights=weights)
			 for samps in samples.T]  
for ax in caxes:
    for axx in ax:
        axx.tick_params(labelsize = 12) 
R_quant = quantiles[1]
R01 = '+'+'{0: .2f}'.format((R_quant[1]-R_quant[0])/1e14) 
R02 = '-'+'{0: .2f}'.format((R_quant[2]-R_quant[1])/1e14) 
R0 = R_quant[0]/1e14
stri1 = '^{'+R01+'}' 
stri2 = '_{'+R02+'}' 
stri = r'$R_{0}$'+'= ${0: .2f}{1}{2}$'.format(R0,stri1,stri2)+r'$\times$'+r'$10^{14}$'
caxes[1,1].set_title(stri)

plt.tight_layout()
plt.show(block = False)


max_like = samples[weights == np.max(weights)][0]
#max_like = np.array([22250,2.7e14,-1,0.88,0,2.2,2.2,-0.34,0.9])
max_like = np.array([22250,2.5e14,-1,0.9,0,2.2,2.2,-0.34,0.9])
T,R,alph,bet,t_exp,t_br,t_br2,alph2, bet2 = max_like
errs = [[quantiles[i][2] - quantiles[i][1],quantiles[i][1], quantiles[i][1] - quantiles[i][0]] for i in range(len(quantiles))] 
errs = np.array(errs)
errs[1,:] = errs[1,:]/1e14
errs[4,1] = errs[4,1]+t0_oqm

errs_str = ['{0: .1f}'.format(errs[i,1])+'^{+'+'{0: .1f}'.format(errs[i,2])+'}_{-'+'{0: .1f}'.format(errs[i,0])+'}' for i in range(len(quantiles))]
head_str = ['$T_{0}$','$R_{0}$',r'$\alpha$',r'$\beta$',r'$t_{exp}$',r'$t_{br}$',r'$t_{br,2}$',r'$\alpha_{2}$',r'$\beta_{2}$']
head_str = [r'\colhead{'+head+'}' for head in head_str]
#T,R,alph,bet,t_exp,t_br,t_br2,alph2, bet2 = mean 

time_1 = np.logspace(-1,np.log10(max_t),1000)

plot_lc_mod2(dat[dat['t_rest']<30],[T,R,alph,bet,t_exp,t_br,t_br2,alph2, bet2, 0,3.1], c_band = c_band, lab_band = lab_band, offset = offset)
plt.xlim((-1.5,6.2))
plt.ylim((-6,-22.5))
plt.text(0.03,0.97,'(d)',fontsize = 15, transform=plt.gca().transAxes)
plt.tight_layout()
plt.show(block = False)

plot_resid_mod2(dat[dat['t_rest']<30],[T,R,alph,bet,t_exp,t_br,t_br2,alph2, bet2, 0,3.1], c_band = c_band, lab_band = lab_band)
plt.show(block = False)



obj = model_cooling_broken(T,R,alph,bet,t_exp,t_br,t_br2,alph2, bet2)
plt.figure(figsize=(5,12))
plt.subplot(3,1,1)
plt.text(0.02,0.92,'(a)',fontsize = 15, transform=plt.gca().transAxes)
plt.plot(time_1,obj.T_evolution(time_1))

plt.errorbar(Temps['t_rest'],Temps['T'],np.vstack([Temps['T'] -Temps['T_down'],Temps['T_up']-Temps['T']]),marker = 'o',ls = '', elinewidth=2.5, ecolor= (0.5,0.5,0.5,0.5) , color = 'r', label = r'SN2022oqm')
plt.yscale('log')
plt.xscale('log')
#plt.xlabel('Rest-frame days from estimated explosion')
plt.ylabel('T [$^{\circ}K$]',fontsize = 15)
plt.xlim(0.4-t_exp,20)
plt.ylim((4500,70000))

plt.subplot(3,1,2)
plt.text(0.02,0.92,'(b)',fontsize = 15, transform=plt.gca().transAxes)

#plt.plot(t,power2(t,1e14,1,3.2e14,0.7,0))
plt.plot(time_1,obj.R_evolution(time_1))

plt.errorbar(Temps['t_rest'],Temps['R'],np.vstack([Temps['R'] -Temps['R_down'],Temps['R_up']-Temps['R']]),marker = 'o',ls = '', elinewidth=2.5, ecolor= (0.5,0.5,0.5,0.5) , color = 'r', label = r'SN2022oqm')
#plt.xlabel('Rest-frame days from estimated explosion')
plt.ylabel('R [cm]',fontsize = 15)
plt.yscale('log')
plt.xscale('log')
plt.xlim(0.4-t_exp,20)
plt.ylim((1e14,2.2e15))

plt.subplot(3,1,3)
plt.text(0.02,0.92,'(c)',fontsize = 15, transform=plt.gca().transAxes)

#plt.plot(t,power2(t,1e14,1,3.2e14,0.7,0))
t_mid = (Temps['t_rest'][0:-1] + Temps['t_rest'][1:])/2
v = np.diff(Temps['R'])/np.diff(Temps['t_rest'])/1e5/3600/24
rerr = np.mean(np.vstack([Temps['R'] -Temps['R_down'],Temps['R_up']-Temps['R']]),axis = 0)
L = Temps['L']
Lerr = np.vstack([Temps['L'] -Temps['L_down'],Temps['L_up']-Temps['L']])
verr = (rerr[0:-1] + rerr[1:])*2/np.diff(Temps['t_rest'])/1e5/3600/24
plt.plot(time_1,obj.L_evolution(time_1))

#plt.plot(t_mid,v,marker = 'o',ls = '' , color = 'y', label = r'SN2022oqm')
plt.errorbar(Temps['t_rest'],L,Lerr,marker = 'o',ls = '', elinewidth=2.5, ecolor= (0.5,0.5,0.5,0.5) , color = 'b', label = r'SN2022oqm')

plt.xlabel('Rest-frame days from estimated explosion',fontsize = 15)
plt.ylabel('L [erg/s]',fontsize = 15)
plt.yscale('log')
plt.xscale('log')
plt.xlim(0.4-t_exp,20)
plt.ylim(0.9e42,1.1e44)
#adjust subplots to reduce horizontal space between them
plt.subplots_adjust(hspace = 0.05)
plt.tight_layout()
plt.show(block = False)

















