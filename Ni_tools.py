import os
import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import interp1d

from matplotlib.ticker import MultipleLocator, FixedLocator
from numba import njit

Msol = 1.988409870698051e+33
c = 29979245800.0

@njit
def energy_Ni56_decay(td, MNi):
        ''' 
        compute decay energy from Nickel: only gamma ray and position kinetic eneygy included here, not counting neutrino
        INPUTS:
                td: days since explosion
                MNi: nickel mass
        '''
        E = MNi*(6.451*np.exp(-td/8.8)+1.445*np.exp(-td/111.3))*1e43




        
        return E

@njit
def energy_Ni56_gam(td, MNi):
        ''' 
        compute decay energy from Nickel: only gamma ray
        INPUTS:
                td: days since explosion
                MNi: nickel mass
        '''
        E_gam = MNi*(6.541*np.exp(-td/8.8)+1.38*np.exp(-td/111.3))*1e43


        
        return E_gam

@njit
def energy_Ni56_pos(td, MNi):
        ''' 
        compute decay energy from Nickel: only position 
        INPUTS:
                td: days since explosion
                MNi: nickel mass
        '''
        E = MNi*(-np.exp(-td/8.8)+4.64*np.exp(-td/111.3))*1e41

        
        return E



@njit
def gamma_ray_escape_correction(td, t0):
        positron_kinetic_energy = 0.12
        gamma_energy = 3.73
        fpos = positron_kinetic_energy / gamma_energy
        fga = 1 - fpos
        fdep = fpos + fga*(1-np.exp(-t0**2/td**2))
        
        return fdep

@njit
def energy_deposition(td, MNi, t0):
        positron_kinetic_energy = 0.12
        gamma_energy = 3.73
        fpos = positron_kinetic_energy / gamma_energy
        fga = 1 - fpos
        fdep = fpos + fga*(1-np.exp(-t0**2/td**2))
    
        Edecay = energy_Ni56_decay(td, MNi)
        Edeposition = Edecay * fdep

        return Edeposition


@njit
def energy_deposition2(td, MNi, t0):

        fdep =(1-np.exp(-t0**2/td**2))
    
        Egam = energy_Ni56_gam(td, MNi)
        Epos = energy_Ni56_pos(td, MNi)

        Edeposition = Egam * fdep + Epos

        return Edeposition



#diffusion time scale parameter: function of kinetic energy (ejecta mass and ejecta velocity), opacity 
def taum_func(Mej=3, Vej=6000, kappa=0.1, beta=13.7):
    '''
    Mej: in unit of solar mass
    Vej: in unit of km/s
    kappa: 0.1 cm**2/g
    
    return diffusion time scale in days
    '''
    Mej_cgs = Mej*Msol
    Vej_cgs = Vej*1e5
    c = 29979245800.0
    Ek = 0.3*Mej_cgs*Vej_cgs**2
    #print(Ek)
    taum = 1.05/(beta*c)**0.5*kappa**0.5*Mej_cgs**(3.0/4)*Ek**(-0.25)
    taum_day = taum/(24*3600.0)
    return taum_day


def TauP(B14, Pms):  
    return 4.7*B14**(-2)*Pms**2
    
#SN luminsoity function
def LSN(ts, Pt_func, taum):
    '''
    units: ts, taum are in unit of days
    '''
    Ls = []
    fint = 0
    N = 200000
    xs = np.linspace(0.001, np.max(ts), num=N+1)
    dtp = np.max(ts)/N         
    for t in xs:
        tp = t
        Ptp = Pt_func(tp)
        f = Ptp*2*tp/taum*np.exp((tp/taum)**2)/taum*dtp
        fint = fint+f
        L = fint*np.exp(-(t/taum)**2)
        Ls.append(L)
    Lout = np.interp(ts,xs, Ls)
    return Lout

 #SN luminsoity function

    
#dipole spin-down luminosity of a magnetar with a 45◦ angle between the magnetic axis and spin axis
#Appendix D of Inserra et al. 2013 https://ui.adsabs.harvard.edu/abs/2013ApJ...770..128I/abstract
#B14: magnetic field strength; Pms: initial spin period; taup: spin down timescale
def Lmagnetar(t, B14, Pms, taup): 
    return 4.9e46*B14**2*Pms**(-4)/(1+t/taup)**2

#@njit
#def Pt_func_magnetar(t,B14=2.4,Pms=14.8):
#    taup = 4.7*B14**(-2)*Pms**2
#    Pt = 4.9e46*B14**2*Pms**(-4)/(1+t/taup)**2
#    return Pt
@njit
def Pt_func_magnetar(t,B14=2.4,Pms=14.8):
    #import ipdb; ipdb.set_trace()
    taup = 4.7/np.power(B14,2)
    taup = taup*np.power(Pms,2)
    Pt = np.power(B14,2)   
    Pt = Pt/np.power(Pms,4)   
    rat = t/taup

    Pt = Pt/(1+rat)
    Pt = Pt/(1+rat)
    Pt = 4.9e46*Pt
    return Pt


@njit
def Pt_func_Ni56(t, Mni = 0.5,t0 = 45):
    #Pt = energy_Ni56_decay(t, Mni)*gamma_ray_escape_correction(t, t0)
    Pt = energy_deposition2(t, Mni,t0)
    return Pt        


@njit
def LSN_magnetar(t_array,taum,B14=2.4,Pms=14.8,N = 5000, kappa=0.1):
    '''
    units: t_array, taum are in unit of days
    '''
    Ls = []
    fint = 0
    delt = 0
    fac = 1
    max_t = np.max(t_array)
    xs = np.linspace(0.001, max_t, N+1)
    dtp = np.max(t_array)/N    
    fint_true = 0
    first = True
    for t in xs:
        tp = t
        Ptp = Pt_func_magnetar(t,B14=B14,Pms=Pms)
        logff = (tp/taum)**2
        
        f = np.log(Ptp*2*tp/taum/taum*dtp) + logff 
        
        #if f>600:
        #    delt = delt + 600
        #    f = f-delt
        #    fac = np.exp(delt)
        #    fint = fint/np.exp(600)


        if (f>700):
            if first:
                delt = 700
                fint = fint/np.exp(delt)
                fint = fint+np.exp(f-delt)
                first = False
            else: 
                fint = fint+np.exp(f-delt)
        elif (f<700):
            fint_true = fint_true + np.exp(f)
            fint = fint_true


        logL = np.log(fint) -(t/taum)**2 + delt
        L = np.exp(logL)
        Ls.append(L)
    Lout = np.interp(t_array,xs, Ls)
    return Lout





@njit
def LSN_Ni56(t_array,taum, Mni = 0.5,t0 = 45,N = 5000, kappa=0.1):
    '''
    units: t_array, taum are in unit of days
    '''
    Ls = []
    fint = 0
    delt = 0
    fac = 1
    max_t = np.max(t_array)
    xs = np.linspace(0.001, max_t, N+1)
    dtp = np.max(t_array)/N    
    fint_true = 0
    first = True
    for t in xs:
        tp = t
        Ptp = Pt_func_Ni56(tp, Mni = Mni,t0 = t0)
        logff = (tp/taum)**2
        
        f = np.log(Ptp*2*tp/taum/taum*dtp) + logff 
        
        #if f>600:
        #    delt = delt + 600
        #    f = f-delt
        #    fac = np.exp(delt)
        #    fint = fint/np.exp(600)


        if (f>700):
            if first:
                delt = 700
                fint = fint/np.exp(delt)
                fint = fint+np.exp(f-delt)
                first = False
            else: 
                fint = fint+np.exp(f-delt)
        elif (f<700):
            fint_true = fint_true + np.exp(f)
            fint = fint_true


        logL = np.log(fint) -(t/taum)**2 + delt
        L = np.exp(logL)
        Ls.append(L)
    Lout = np.interp(t_array,xs, Ls)
    return Lout





@njit
def LSN_Ni56_orig(t_array,taum, Mni = 0.5,t0 = 45,N = 5000, kappa=0.1):
    '''
    units: t_array, taum are in unit of days
    '''
    Ls = []
    fint = 0
    fac = 1
    max_t = np.max(t_array)
    xs = np.linspace(0.001, max_t, N+1)
    dtp = np.max(t_array)/N         
    for t in xs:
        tp = t
        Ptp = Pt_func_Ni56(tp, Mni = Mni,t0 = t0)
        if np.exp((tp/taum)**2)>1e250:
            fac = 1e250 
        ff = np.exp((tp/taum)**2)/fac
        f = Ptp*2*tp/taum*ff/taum*dtp
        #if (f == np.inf).any():
        #    import ipdb; ipdb.set_trace()

        fint = fint/fac+f
        L = fint*np.exp(-(t/taum)**2)*fac
            
        Ls.append(L)
    Lout = np.interp(t_array,xs, Ls)
    if (np.isnan(Lout)).any():
        import ipdb; ipdb.set_trace()
    return Lout






LSN_Ni56(np.array([0.1]),10, Mni = 0.5,t0 = 45,N = 5000, kappa=0.1)



