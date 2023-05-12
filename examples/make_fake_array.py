from fakepta.fake_pta import Pulsar, make_fake_array, copy_array, plot_pta
import numpy as np
import matplotlib.pyplot as plt
import pickle, json
import scipy.constants as sc
from fakepta.correlated_noises import add_correlated_red_noise_gp

''' Make fake array of 25 psrs with :
    - Tobs : observation time for each pulsar (in years)
    - ntoas : number of TOAs for each pulsars
    - isotropic : if True, pulsars are isotropically distributed in the sky, if False, randomly distributed
    - gaps : if True, pulsars will have unevenly sampled TOAs, if False, evenly sampled TOAs
    - toaerr : measurement error for every TOA (in seconds)
    - pdist : pulsar distance (in kiloparsec)
    - backends : list of backend names
    - gp_noises : if True, the noises are injected with the "fakepta.Pulsar.add_time_correlated_noise_gp" method, if False, with "fakepta.Pulsar.add_time_correlated_noise"
    - noisedict : dictionnary of ENTERPRISE noise parameter values with format {"parameter_name":value}
    - custom_models : dictionnary containing the number of frequency bins for red noise (RN), dispersion measure noise (DM) and scattering variation (Sv) for injection
                      with format {"psrname":{"RN":nfbin_rn, "DM":nfbin_dm, "Sv":nfbin_sv}}
'''
# psrs = make_fake_array(npsrs=25, Tobs=10, ntoas=1000, isotropic=True, gaps=True, toaerr=10**(-6), pdist=1., backends='NUPPI', gp_noises=True, noisedict=None, custom_models=None)
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
