from fakepta.fake_pta import Pulsar, make_fake_array, copy_array, plot_pta
import numpy as np
import matplotlib.pyplot as plt
import pickle, json
import scipy.constants as sc
from fakepta.correlated_noises import add_correlated_red_noise_gp

''' Make fake array of 25 psrs '''
# psrs = make_fake_array(npsrs=25, Tobs=10, ntoas=1000, isotropic=True, gaps=True, toaerr=10**(-6), pdist=1., backends='NUPPI', gp_noises=True)
# plot_pta(psrs, plot_name=False)

''' OR Copy existing array + add noise'''
psrs_0 = pickle.load(open('/home/mikel/PTA/scripts/data/pkl/GWB-EPTA-DR2_v1.3_newsys_trim.pkl', 'rb'))
noisedict = json.load(open('/home/mikel/PTA/scripts/noises/EPTA/noisedict_dr2_newsys_trim.json', 'rb'))
custom_models = json.load(open('/home/mikel/PTA/scripts/noises/EPTA/custom_models_newsys_trim.json', 'rb'))
psrs = copy_array(psrs_0, noisedict, custom_models)

''' Set residuals to zero and re-inject noises'''
for psr in psrs:
    print('Injecting noises for', psr.name)
    psr.make_ideal()
    psr.add_white_noise()
    psr.add_red_noise()
    psr.add_dm_noise()
    psr.add_chromatic_noise()

''' Inject GWB'''
print('Injecting GWB')
add_correlated_red_noise_gp(psrs, orf='hd')

''' Inject CGW '''
params = {}
params['log10_h'] = -13.5
params['costheta'] = 0.12
params['phi'] = 3.2
params['cosinc'] = 0.3
params['phase0'] = 1.6
params['psi'] = 1.2
params['log10_Mc'] = 9.2
params['log10_fgw'] = -8.3

print('Injecting CGW')
for psr in psrs:
    psr.add_cgw(params['costheta'], params['phi'], params['cosinc'], params['log10_Mc'], params['log10_fgw'], params['log10_h'], params['phase0'], params['psi'], psrterm=True)

''' Save pickle '''
pickle.dump(psrs, open('.simulated/data/fake_25_psrs_gp_gwb+cgw.pkl', 'wb'))
print('Done')
