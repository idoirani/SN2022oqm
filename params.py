import astropy.io.ascii as ascii
import numpy as np



path_out =      './output'
path_package =  './'
path_filters =  './Filters/'

path_dates_sn=     './test_files/dates_22oqm.txt'       #'path/to/dates/file'
path_out =         './test_files/output'       #'path/to/out/file'
path_data =        './test_files/2022oqm_photometry.txt'       #'path/to/photometry/file'
path_bb_out =       './test_files/bb_fits.txt'       #'path/to/bb_fits
#data columns for photometry file
Date_col    = 'jd'
absmag_col  =  'absmag'
M_err_col   = 'AB_MAG_ERR'
filter_col  = 'filter'
flux_col    = 'flux'  
fluxerr_col = 'fluxerr'
piv_wl_col  = 'piv_wl'
inst_col = 'instrument'


t_first = 2459770.76  # relative to reference time, this will be the upper limit on t0 
t_nd =    2459771.70   # relative to reference time, t_nd-1 will be the lower limit on t0 
t0_init = (t_first - t_nd)/2 + t_nd 

d_mpc = 58
dm = 5*np.log10(d_mpc)+25
sn = 'SN2022oqm' 
z = 0.0113 #redshift
max_t = 10 #maximum time to include in fit
min_t = 2.5  #minimum time to include in fit
k34 = 0.62 #0.62 for ibc or 1 for H-rich
Rv = 3.1
Rv_fit = False # recomennded only if multiband UV data is included
LAW = 'MW' #   MW,Cal00,LMC,SMC implemented
sys_err = 0.1 #instrument cross-calibration term
mode = 'write' ##read|write|replot 
plot_BB = True #plot BB fits during individual epochs fits

priors_phys  =        {'R13':np.array([0.1,6]),
				       'v85':np.array([0.1,3]),
				       'fM':np.array([0.1,200]),
				       'Menv':np.array([0.3,30]),
					   'ebv':np.array([0,0.25]),
					   'Rv':np.array([2,5])}


ebv_host_init = 0 #intial host extinction to correct
results = {}
import pickle
covar = False
corner_plot = True
lc_plot = True  #create best fit light curve plots
bb_plot = True  #create best fit blackbody plots relative to the blackbody fits to indivdual epochs
show = True     #show plots
modify_BB_MSW = False #do not edit






#dictionary of filter transmission curves
dic_transmission=     {'ATLAS_c':'Misc_Atlas.cyan.dat',
					 'ATLAS_o':'Misc_Atlas.orange.dat',
					 'ZTF_g':'P48_g.txt',    
					 'ZTF_r':'P48_r.txt',    
					 'ZTF_i':'ZTF_transmission_i_from_twiki.ascii',                        
					 'u_P60':'SDSS_u.txt',
					 'g_P60':'SDSS_g.txt',
					 'r_P60':'SDSS_r.txt',
					 'i_P60':'SDSS_i.txt',
					 'LT_u':'LT_u.txt',  
					 'LT_g':'LT_g.txt',  
					 'LT_r':'LT_r.txt',
					 'LT_i':'LT_i.txt',   
					 'LT_z':'LT_z.txt',
					 'LCO_u':'LCO_AA_SDSS.up.txt',
					 'LCO_g':'LCO_AA_SDSS.gp.txt',
					 'LCO_r':'LCO_AA_SDSS.rp.txt',
					 'LCO_i':'LCO_AA_SDSS.ip.txt',
					 'LCO_r':'LCO_AA_SDSS.rp.txt',
					 'LCO_i':'LCO_AA_SDSS.ip.txt',
					 'LCO_V':'LasCumbres_LasCumbres.Bessel_V.dat',
					 'LCO_B':'LasCumbres_LasCumbres.Bessel_B.dat',
					 'UVM2'   :'Swift_UVM2.rtf',
					 'UVW1'   :'Swift_UVW1.rtf',  
					 'UVW2'   :'Swift_UVW2.rtf', 
					 'u_swift':'Swift_u.rtf',
					 'v_swift':'Swift_V.rtf',  
					 'b_swift':'Swift_B.rtf',
					 'NOT_u':'NOT_u.txt',
					 'NOT_g':'NOT_g.txt',
					 'NOT_r':'NOT_r.txt',
					 'NOT_i':'NOT_i.txt',
					 'NOT_z':'NOT_z.txt',
					 'KAIT_B':   'B_kait4_shifted.txt',
					 'KAIT_V':   'V_kait4_shifted.txt',
					 'KAIT_R':   'R_kait4_shifted.txt',
					 'KAIT_I':   'I_kait4_shifted.txt',
					 'KAIT_CLEAR':   'e2v_QE_midband.csv',
					 'Ni_B':   'B_Nickel2.txt',
					 'Ni_V':   'V_Nickel2.txt',
					 'Ni_R':   'R_Nickel2.txt',
					 'Ni_I':   'I_Nickel2.txt',
					 'Ni_CLEAR':   'e2v_QE_midband.csv',
					 #'KAIT_CLEAR':'',
					 'MMIRS_J':'MMT_MMIRS.J.dat',
					 'MMIRS_H':'MMT_MMIRS.H.dat',
					 'MMIRS_Ks':'MMT_MMIRS.Ks.dat'}

dic_transmission = {x:path_filters + dic_transmission[x] for x in dic_transmission.keys()}


filter_transmission_fast = {}
for x in dic_transmission.keys():
	try:
 		filter_transmission_fast[x] = np.loadtxt(dic_transmission[x])
	except:
		filter_transmission_fast[x] = np.loadtxt(dic_transmission[x],delimiter  = ',')	


filter_transmission = {x:ascii.read(dic_transmission[x]) for x in dic_transmission.keys()}



### plotting parameters ###
## band colors
c_band ={'UVW2':'#060606'
		,'UVM2':'#FF37DE'
		,'UVW1':'#AE0DBB'
		,'u_swift':'#6D00C2'
		,'b_swift':'#1300FF'
		,'ZTF_g':'#00BA41'
		,'v_swift':'#00DCA7'
		,'u_P60': '#6D00C2'
		,'g_P60':'#00BA41'
		,'r_P60':'#EA0000'
		,'i_P60':'#D3DE00'
        ,'LCO_u':'#6D00C2'
        ,'LCO_g':'#00BA41'
        ,'LCO_r':'#EA0000'
        ,'LCO_i':'#D3DE00'
        ,'LCO_V':'#00DCA7'
        ,'LCO_B':'#1300FF'
        ,'NOT_u':'#6D00C2'
        ,'NOT_g':'#00BA41'
        ,'NOT_r':'#EA0000'
        ,'NOT_i':'#D3DE00'
        ,'NOT_z':'#680500'
        ,'KAIT_B':'#1300FF'
        ,'KAIT_V':'#00DCA7'
        ,'KAIT_R':'#EA0000'
        ,'KAIT_CLEAR':'#B2607E'
        ,'KAIT_I':'#D3DE00'
        ,'Ni_B':'#1300FF'
        ,'Ni_V':'#00DCA7'
        ,'Ni_R':'#EA0000'
        ,'Ni_CLEAR':'#B2607E'
        ,'Ni_I':'#D3DE00'
        ,'MMIRS_J':'#E79600'
        ,'MMIRS_H':'#969696'
        ,'MMIRS_Ks':'#6C3613'
		,'ZTF_r':'#EA0000'
		,'ZTF_i':'#D3DE00'
		,'LT_u':'#6D00C2'
		,'LT_z':'#680500'      
		,'LT_g':'#00BA41'
		,'LT_r':'#EA0000'
		,'LT_i':'#D3DE00'
		,'ATLAS_c':'#05E5DB'
		,'ATLAS_o':'#E56105'}

## band labels


lab_band = {'UVW2':'W2'
           ,'UVM2':'M2'
           ,'UVW1':'W1'
           ,'u_swift':'U'
           ,'b_swift':'b'
           ,'ZTF_g':'g'
           ,'g_P60':'g'
           ,'v_swift':'v'
           ,'r_P60':'r'
           ,'u_P60':'u'
           ,'ZTF_r':'r'
           ,'ZTF_i':'i'
           ,'i_P60':'i'
           ,'LT_z':'z'     
           ,'LT_u':'u'
           ,'LT_g':'g'  
           ,'LT_r':'r/R'
           ,'LT_i':'i/I'
         ,'KAIT_B':'B'
         ,'KAIT_V':'V'
         ,'KAIT_R':'R'
         ,'KAIT_I':'I'
         ,'KAIT_CLEAR':'Clear'
         ,'KAIT_I':''
         ,'Ni_B':'B'
         ,'Ni_V':'V'
         ,'Ni_R':'R'
         ,'Ni_CLEAR':'CLEAR'
         ,'Ni_I':'I'
         ,'MMIRS_J':'J'
         ,'MMIRS_H':'H'
         ,'MMIRS_Ks':'Ks'
		,'ATLAS_c':'c'
		,'ATLAS_o':'o'
        ,'LCO_u':'u'
        ,'LCO_g':'g'
        ,'LCO_r':'r'
        ,'LCO_i':'i'
        ,'LCO_V':'V'
        ,'LCO_B':'B'
        ,'NOT_u':'u'
        ,'NOT_g':'g'
        ,'NOT_r':'r'
        ,'NOT_i':'i'
        ,'NOT_z':'z'} 


## band offsets
offset   = {'UVW2':-4.5
		   ,'UVM2':-3.5
		   ,'UVW1':-2.5
		   ,'u_swift':-2
		   ,'u_P60':-2
		   ,'LT_u':-2
		   ,'b_swift':-1
		   ,'ZTF_g':0
		   ,'g_P60':0
		   ,'LT_g':0 
		   ,'v_swift':1
		   ,'r_P60':2
		   ,'ZTF_r':2
		   ,'LT_r':2
		   ,'ZTF_i':3
		   ,'i_P60':3
		   ,'LT_i':3
		   ,'LT_z':4 
         ,'KAIT_B':-1
         ,'KAIT_V':-1
         ,'KAIT_R':2
         ,'KAIT_I':3
         ,'KAIT_CLEAR':1
         ,'KAIT_I':3
         ,'Ni_B':-1
         ,'Ni_V':-1
         ,'Ni_R':2
         ,'Ni_CLEAR':1
         ,'Ni_I':3
         ,'MMIRS_J':5
         ,'MMIRS_H':6
         ,'MMIRS_Ks':7
		,'ATLAS_c':-0.5
		,'ATLAS_o':0.5
        ,'LCO_u':-2
        ,'LCO_g':0
        ,'LCO_r':2
        ,'LCO_i':3
        ,'LCO_V':1
        ,'LCO_B':-1
        ,'NOT_u':-2
        ,'NOT_g':0
        ,'NOT_r':2
        ,'NOT_i':3
        ,'NOT_z':4}  



markers={
	'r_sdss'   :'o'
	,'g_sdss'  :'o'
	,'i_sdss'  :'o'
	,'z_sdss'  :'o'
	,'u_sdss'  :'o'
	,'ZTF_r'   :'s' 
	,'ZTF_g'   :'s' 
	,'ZTF_i'   :'s' 
	,'u_swift' :'p' 
	,'v_swift' :'p' 
	,'b_swift' :'p' 
	,'UVM2'    :'p'
	,'UVW2'    :'p'
	,'UVW1'    :'p' 
	,'u_P60'   :'.'
	,'g_P60'   :'.'
	,'r_P60'   :'.'
	,'i_P60'   :'.'
	,'LT_u'    :'^'
	,'LT_g'    :'^'
	,'LT_r'    :'^'
	,'LT_i'    :'^'
	,'LT_z'    :'^'
	,'NOT_u'    :'P'
	,'NOT_g'    :'P'
	,'NOT_r'    :'P'
	,'NOT_i'    :'P'
	,'NOT_z'    :'P'
	,'LCO_u'   :'v'
	,'LCO_g'   :'v'
	,'LCO_r'   :'v'
	,'LCO_i'   :'v'
	,'LCO_V'   :'v'
	,'LCO_B'   :'v'
	,'KAIT_B':'P'
	,'KAIT_V':'P'
	,'KAIT_R':'P'
	,'KAIT_I':'P'
	,'KAIT_CLEAR':'P'
	,'Ni_B':'P'
	,'Ni_V':'P'
	,'Ni_R':'P'
	,'Ni_I':'P'
	,'Ni_CLEAR':'P'
	,'MMIRS_J':'*'
	,'MMIRS_H':'*'
	,'MMIRS_Ks':'*'	
	,'ATLAS_c':'x' 
	,'ATLAS_o':'x'}


#list of filters
FILTERS =   list(dic_transmission.keys())