import importlib
import inspect
import json
import logging
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
from enterprise_extensions import deterministic as det
from scipy.optimize import fsolve

try:
    import healpy as hp
except:
    logging.warning('healpy module not found.')

# load spectrum functions from "spectrum.py"
module = importlib.import_module('fakepta.spectrum')
spec = inspect.getmembers(module, inspect.isfunction)
spec_params = {}
for s_name, s_obj in spec:
    pnames = [*inspect.signature(s_obj).parameters]
    pnames.remove('f')
    spec_params[s_name] = pnames
spec = dict(spec)

default_backend_config = {
    'NUPPI': {
            'sub_bands': [1400], # Sub-bands used by this backend
            'RMS': 1e-6,
            'p': 1. # Proportion of observations from this backend 
            }
}

class Pulsar:

    def __init__(self, toas, theta, phi, pdist=(1., 0.2),
                 backend_config=None, custom_noisedict=None, custom_model=None,
                 tm_params=None, ephem=None, toaerr=None):

        if backend_config is None:
            backend_config = default_backend_config

        self.backend_config = backend_config  # {backend_name: [subband_freqs]}
        self.backends = list(backend_config.keys())
        
        # Store the relative occurence of each backend
        self.backend_weights = np.array([backend_config[backend].get('p', 1.) 
                                for backend in self.backends])
        self.backend_weights /= np.sum(self.backend_weights)
        
        # Store which backends have sub band ToAs
        self.ecorr_backends = [backend for backend in self.backends 
                            if len(backend_config[backend]['sub_bands']) > 1]

        self.nepochs = len(toas)

        # For each epoch, randomly assign one backend
        self.epoch_backends = np.random.choice(self.backends, 
                                                size=self.nepochs,
                                                replace=True,
                                                p=self.backend_weights)

        # Build TOA, freq, and backend_flag arrays
        exp_toas, exp_freqs, exp_backend_flags = self.get_expanded_toas(
            toas, backend_config
        )
        self.toas = np.array(exp_toas)
        self.freqs = np.array(exp_freqs)
        self.backend_flags = np.array(exp_backend_flags)
        
        if toaerr is not None:
            # Use a global RMS
            self.toaerrs = toaerr * np.ones(len(self.toas))
        else:
            self.toaerrs = np.zeros(len(self.toas))
            # Use RMS per backend
            for backend in self.backends:
                mask = self.backend_flags == backend
                self.toaerrs[mask] = self.backend_config[backend]['RMS']

        self.residuals = np.zeros(len(self.toas))
        self.Tspan = np.amax(self.toas) - np.amin(self.toas)

        if custom_model is None:
            self.custom_model = {'RN': 30, 'DM': 100, 'Sv': None}
        else:
            self.custom_model = custom_model
        self.signal_model = {}
        self.flags = {}
        self.flags['pta'] = ['FAKE'] * len(self.toas)

        self.theta = theta
        self.phi = phi
        self.pos = np.array([np.cos(phi)*np.sin(theta),
                             np.sin(phi)*np.sin(theta),
                             np.cos(theta)])
        if ephem is not None:
            self.ephem = ephem
            self.planetssb = ephem.get_planet_ssb(self.toas)
            self.pos_t = np.tile(self.pos, (len(self.toas), 1))
        else:
            self.planetssb = None
            self.pos_t = None

        self.pdist = pdist
        self.name = self.get_psrname()
        self.init_tm_pars(tm_params)
        self.make_Mmat()
        self.fitpars = [*self.tm_pars]
        self.init_noisedict(custom_noisedict)

    def get_expanded_toas(self, toas, backend_config, freq_scale=5):
        # Each epoch contributes len(backend_config[assigned_backend]) TOAs
        exp_toas = []
        exp_freqs = []
        exp_backend_flags = []

        for t, backend in zip(toas, self.epoch_backends):
            subband_freqs = backend_config[backend]['sub_bands']
            n_sub = len(subband_freqs)
            # NOTE: in theory those should not be strictly equal
            exp_toas.extend([t] * n_sub)
            for f in subband_freqs:
                exp_freqs.append(abs(f + np.random.normal(scale=freq_scale)))
            exp_backend_flags.extend([backend] * n_sub)

        return exp_toas, exp_freqs, exp_backend_flags

    def init_noisedict(self, custom_noisedict=None):
        noisedict = {}

        if custom_noisedict is None:
            for backend in self.backends:
                noisedict[self.name+'_'+backend+'_efac']           = 1.
                noisedict[self.name+'_'+backend+'_log10_tnequad']  = -8.
                noisedict[self.name+'_'+backend+'_log10_t2equad']  = -8.
                noisedict[self.name+'_'+backend+'_log10_ecorr']    = -8.
            custom_noisedict={}

        elif np.any([self.name in key for key in custom_noisedict]):
            # fully qualified keys already present (e.g. from a real noisedict file)
            for key, val in custom_noisedict.items():
                if self.name in key:
                    noisedict[key] = val

        elif np.all([backend+'_efac' in custom_noisedict.keys() for backend in self.backends]):
            # per-backend keys without pulsar name prefix
            for backend in self.backends:
                noisedict[self.name+'_'+backend+'_efac']          = custom_noisedict[backend+'_efac']
                noisedict[self.name+'_'+backend+'_log10_tnequad'] = custom_noisedict[backend+'_log10_tnequad']
                for opt in ['log10_t2equad', 'log10_ecorr']:
                    key = backend+'_'+opt
                    if key in custom_noisedict.keys():
                        noisedict[self.name+'_'+backend+'_'+opt] = custom_noisedict[key]

        else:
            # scalar fallback: same value for all backends
            for backend in self.backends:
                noisedict[self.name+'_'+backend+'_efac']          = custom_noisedict.get('efac', 1.)
                noisedict[self.name+'_'+backend+'_log10_tnequad'] = custom_noisedict.get('log10_tnequad', -8.)
                for opt in ['log10_t2equad', 'log10_ecorr']:
                    if opt in custom_noisedict.keys():
                        noisedict[self.name+'_'+backend+'_'+opt] = custom_noisedict[opt]

        # GP noise — unchanged logic
        for gp in ['red_noise', 'dm_gp', 'chrom_gp']:
            if np.any([gp in key for key in custom_noisedict.keys() or []]):
                try:
                    key_amp = self.name+'_'+gp+'_log10_A' if self.name+'_'+gp+'_log10_A' in custom_noisedict.keys() else gp+'_log10_A'
                    key_gam = self.name+'_'+gp+'_gamma'   if self.name+'_'+gp+'_gamma'   in custom_noisedict.keys() else gp+'_gamma'
                    noisedict[self.name+'_'+gp+'_log10_A'] = custom_noisedict[key_amp]
                    noisedict[self.name+'_'+gp+'_gamma']   = custom_noisedict[key_gam]
                except:
                    pass

        self.noisedict = noisedict

    def init_tm_pars(self, timing_model):

        self.tm_pars = {}
        self.tm_pars['F0'] = (200, 1e-13)
        self.tm_pars['F1'] = (0., 1e-20)
        self.tm_pars['DM'] = (0., 5e-4)
        self.tm_pars['DM1'] = (0., 1e-4)
        self.tm_pars['DM2'] = (0., 1e-5)
        self.tm_pars['ELONG'] = (0., 1e-5)
        self.tm_pars['ELAT'] = (0., 1e-5)
        if timing_model is not None:
            self.tm_pars.update(timing_model)

    def make_Mmat(self, t0=0.):

        npar = len([*self.tm_pars]) + 1
        self.Mmat = np.zeros((len(self.toas), npar))
        self.Mmat[:, 0] = np.ones(len(self.toas))
        self.Mmat[:, 1] = -(self.toas - t0) / self.tm_pars['F0'][0]
        self.Mmat[:, 2] = -0.5 * (self.toas - t0)**2 / self.tm_pars['F0'][0]
        self.Mmat[:, 3] = 1 / self.freqs**2
        self.Mmat[:, 4] = (self.toas - t0) / self.freqs**2 / self.tm_pars['F0'][0]
        self.Mmat[:, 5] = 0.5 * (self.toas - t0)**2 / self.freqs**2 / self.tm_pars['F0'][0]
        self.Mmat[:, 6] = np.cos(2*np.pi/sc.Julian_year * (self.toas - t0))
        self.Mmat[:, 7] = np.sin(2*np.pi/sc.Julian_year * (self.toas - t0))

    def update_position(self, theta, phi, update_name=False):
        
        self.theta = theta
        self.phi = phi
        self.pos = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
        if update_name:
            self.name = self.get_psrname()

    def update_noisedict(self, prefix, dict_vals):

        params = {}
        for key in [*dict_vals]:
            params[prefix+'_'+key] = dict_vals[key]
        self.noisedict.update(params)

    def make_ideal(self):

        # set residuals to zero and clean signal model dict

        self.residuals = np.zeros(len(self.toas))
        for signal in [*self.signal_model]:
            self.signal_model.pop(signal)
            for key in [*self.noisedict]:
                if signal in key:
                    self.noisedict.pop(key)

    def add_white_noise(self, add_ecorr=False, randomize=False, rng=None):

        if rng is None:
            rng = np.random.default_rng()

        if randomize:
            for key in list(self.noisedict):
                if 'efac'   in key: self.noisedict[key] = rng.uniform(0.5, 2.5)
                if 'equad'  in key: self.noisedict[key] = rng.uniform(-8., -5.)
                if 'ecorr'  in key: self.noisedict[key] = rng.uniform(-10., -7.)

        # EFAC + EQUAD: per backend, applied to all its sub-band TOAs
        toaerrs2 = np.zeros(len(self.toaerrs))
        for backend in self.backends:
            mask = self.backend_flags == backend
            efac   = self.noisedict[self.name+'_'+backend+'_efac']
            log10_tnequad = self.noisedict[self.name+'_'+backend+'_log10_tnequad']
            toaerrs2[mask] = efac**2 * self.toaerrs[mask]**2 + 10**(2*log10_tnequad)
        self.residuals += rng.normal(scale=np.sqrt(toaerrs2))

        # ECORR: one multivariate draw per epoch per backend group
        if add_ecorr:
            groups = self.quantise_ecorr()
            for backend, epoch_list in groups:
                if backend in self.ecorr_backends:
                    ecorr_key = self.name+'_'+backend+'_log10_ecorr'
                    sigma_j = 10**self.noisedict[ecorr_key]
                    for epoch_idx in epoch_list:
                        n = len(epoch_idx)
                        if n < 2:
                            self.residuals[epoch_idx] += rng.normal(scale=sigma_j)
                            continue
                        cov = sigma_j**2 * np.ones((n, n))
                        self.residuals[epoch_idx] += rng.multivariate_normal(
                            mean=np.zeros(n), cov=cov
                        )

    def quantise_ecorr(self, dt=0.5):
        """
        For each backend (group), collect epoch buckets containing all its
        sub-band TOAs. Returns list of (backend_name, [epoch_index_arrays]).
        """
        dt_sec = dt * 24 * 3600
        times = self.toas
        result = []

        for backend in self.backends:
            group_mask = self.backend_flags == backend
            group_idx = np.where(group_mask)[0]
            if len(group_idx) == 0:
                continue

            sort_order = np.argsort(times[group_idx])
            sorted_idx = group_idx[sort_order]
            sorted_times = times[sorted_idx]

            epoch_groups = []
            t0 = sorted_times[0]
            current_epoch = [sorted_idx[0]]

            for k in range(1, len(sorted_idx)):
                if sorted_times[k] - t0 < dt_sec:
                    current_epoch.append(sorted_idx[k])
                else:
                    epoch_groups.append(np.array(current_epoch))
                    t0 = sorted_times[k]
                    current_epoch = [sorted_idx[k]]
            epoch_groups.append(np.array(current_epoch))

            result.append((backend, epoch_groups))

        return result

    def add_red_noise(self, spectrum='powerlaw', f_psd=None, rng=None, **kwargs):

        rn_components = self.custom_model['RN']
        if rn_components is not None:

            if f_psd is None:
                f_psd = np.arange(1, rn_components+1) / self.Tspan

            if 'red_noise' in self.signal_model:
                self.residuals -= self.reconstruct_signal(['red_noise'])

            if spectrum == 'custom':
                psd = kwargs['custom_psd']
            elif spectrum in [*spec]:
                if len(kwargs) == 0:
                    try:
                        kwargs = {pname : self.noisedict[self.name+'_red_noise_'+pname] for pname in spec_params[spectrum]}
                    except:
                        logging.error('PSD parameters must be in noisedict or parsed as input.')
                        return
                psd = spec[spectrum](f_psd, **kwargs)
                self.update_noisedict(self.name+'_red_noise', kwargs)

                self.add_time_correlated_noise(signal='red_noise', spectrum=spectrum, idx=0., 
                                               psd=psd, f_psd=f_psd, rng=rng)

    def add_dm_noise(self, spectrum='powerlaw', f_psd=None, rng=None, **kwargs):

        dm_components = self.custom_model['DM']
        if dm_components is not None:
            
            if f_psd is None:
                f_psd = np.arange(1, dm_components+1) / self.Tspan

            if 'dm_gp' in self.signal_model:
                self.residuals -= self.reconstruct_signal(['dm_gp'])

            if spectrum == 'custom':
                psd = kwargs['custom_psd']
            elif spectrum in [*spec]:
                if len(kwargs) == 0:
                    try:
                        kwargs = {pname : self.noisedict[self.name+'_dm_gp_'+pname] for pname in spec_params[spectrum]}
                    except:
                        logging.error('PSD parameters must be in noisedict or parsed as input.')
                        return
                psd = spec[spectrum](f_psd, **kwargs)
                self.update_noisedict(self.name+'_dm_gp', kwargs)

            self.add_time_correlated_noise(signal='dm_gp', spectrum=spectrum, idx=2., 
                                           psd=psd, f_psd=f_psd, rng=rng)

    def add_chromatic_noise(self, spectrum='powerlaw', f_psd=None, rng=None, **kwargs):

        sv_components = self.custom_model['Sv']
        if sv_components is not None:
            
            if f_psd is None:
                f_psd = np.arange(1, sv_components+1) / self.Tspan

            if 'chrom_gp' in self.signal_model:
                self.residuals -= self.reconstruct_signal(['chrom_gp'])

            if spectrum == 'custom':
                psd = kwargs['custom_psd']
            elif spectrum in [*spec]:
                if len(kwargs) == 0:
                    try:
                        kwargs = {pname : self.noisedict[self.name+'_chrom_gp_'+pname] for pname in spec_params[spectrum]}
                    except:
                        logging.error('PSD parameters must be in noisedict or parsed as input.')
                        return
                psd = spec[spectrum](f_psd, **kwargs)
                self.update_noisedict(self.name+'_chrom_gp', kwargs)

            self.add_time_correlated_noise(signal='chrom_gp', spectrum=spectrum, idx=4, 
                                           psd=psd, f_psd=f_psd, rng=rng)

    def add_system_noise(self, backend=None, components=30, spectrum='powerlaw', f_psd=None, 
                         rng=None, **kwargs):

        assert backend is not None, '"backend" name where system noise is injected must be given'

        if f_psd is None:
            f_psd = np.arange(1, components+1) / self.Tspan

        if 'system_noise_'+str(backend) in self.signal_model:
            self.residuals -= self.reconstruct_signal(['system_noise_'+str(backend)])

        if spectrum == 'custom':
            psd = kwargs['custom_psd']
        elif spectrum in [*spec]:
            if len(kwargs) == 0:
                try:
                    kwargs = {pname : self.noisedict[self.name+'_system_noise_'+str(backend)+'_'+pname] for pname in spec_params[spectrum]}
                except:
                    logging.error('PSD parameters must be in noisedict or parsed as input.')
                    return
            psd = spec[spectrum](f_psd, kwargs)
            self.update_noisedict(self.name+'_system_noise_'+str(backend), kwargs)

        self.add_time_correlated_noise(signal='system_noise_'+str(backend), idx=0., 
                                       backend=backend, psd=psd, f_psd=f_psd, rng=rng)

    def add_time_correlated_noise(self, signal='', spectrum='powerlaw', psd=None, f_psd=None, idx=0, 
                                  freqf=1400, backend=None, rng=None):
        
        if rng is None:
            rng = np.random.default_rng()

        # generate time correlated noise with given PSD and chromatic index

        if backend is not None:
            signal = backend + '_' + signal
            mask = self.backend_flags == backend
            if not np.any(mask):
                logging.error(backend, 'not found in backend_flags.')
                return
        else:
            mask = np.ones(len(self.toas), dtype='bool')

        df = np.diff(np.append(0., f_psd))
        assert len(psd) == len(f_psd), '"psd" and "f_psd" must be same length. The frequencies "f_psd" correspond to the frequencies where the "psd" is evaluated.'
        psd = np.repeat(psd, 2)

        coeffs = rng.normal(loc=0., scale=np.sqrt(psd))

        # save noise properties in signal model
        self.signal_model[signal] = {}
        self.signal_model[signal]['spectrum'] = spectrum
        self.signal_model[signal]['f'] = f_psd
        self.signal_model[signal]['psd'] = psd[::2]
        self.signal_model[signal]['fourier'] = np.vstack((coeffs[::2] / df**0.5, coeffs[1::2] / df**0.5))
        self.signal_model[signal]['nbin'] = len(f_psd)
        self.signal_model[signal]['idx'] = idx
        
        for i in range(len(f_psd)):
            self.residuals[mask] += (freqf/self.freqs)**idx * df[i]**0.5 * coeffs[2*i] * np.cos(2*np.pi*f_psd[i]*self.toas[mask])
            self.residuals[mask] += (freqf/self.freqs)**idx * df[i]**0.5 * coeffs[2*i+1] * np.sin(2*np.pi*f_psd[i]*self.toas[mask])

    def make_time_correlated_noise_cov(self, signal='', freqf=1400):

        # returns covariance matrix of time correlated noise with given PSD and chromatic index

        if 'system_noise' in signal:
            backend = signal.split('system_noise_')[1]
        else:
            backend = None

        if backend is not None:
            signal = backend + '_' + signal
            mask = self.backend_flags == backend
            if not np.any(mask):
                logging.error(backend, 'not found in backend_flags.')
                return
        else:
            mask = np.ones(len(self.toas), dtype='bool')

        # save noise properties in signal model
        f = self.signal_model[signal]['f']
        psd = self.signal_model[signal]['psd']
        components = self.signal_model[signal]['nbin']
        idx = self.signal_model[signal]['idx']

        df = np.diff(np.append(0, f))
        psd = np.repeat(psd * df, 2)
        basis = np.zeros((len(self.toas[mask]), 2*components))
        for i in range(components):
            basis[:, 2*i] = (freqf/self.freqs)**idx * np.cos(2*np.pi*f[i]*self.toas[mask])
            basis[:, 2*i+1] = (freqf/self.freqs)**idx * np.sin(2*np.pi*f[i]*self.toas[mask])
        cov = np.dot(basis, np.dot(np.diag(psd), basis.T))
        return cov
        
    def add_cgw(self, costheta, phi, cosinc, log10_mc, log10_fgw, log10_h, phase0, psi, psrterm=False):

        # add continuous gravitational wave from circular black hole binary

        if 'cgw' in self.signal_model:
            ncgw = len(self.signal_model['cgw'])
        else:
            self.signal_model['cgw'] = {}
            ncgw = 0
        
        self.signal_model['cgw'][str(ncgw)] = {'costheta':costheta, 'phi':phi, 'cosinc':cosinc,
                                                'log10_mc':log10_mc, 'log10_fgw':log10_fgw, 'log10_h':log10_h,
                                                'phase0':phase0, 'psi':psi, 'psrterm':psrterm}

        cgw = det.cw_delay(self.toas, self.pos, self.pdist,
                            cos_gwtheta=costheta, gwphi=phi,
                            cos_inc=cosinc, log10_mc=log10_mc, 
                            log10_fgw=log10_fgw, evolve=True,
                            log10_h=log10_h, phase0=phase0, 
                            psi=psi, psrTerm=psrterm)
        self.residuals += cgw

    def add_deterministic(self, waveform, **kwargs):

        fname = waveform.__name__
        if fname in self.signal_model:
            ndet = len(self.signal_model[fname])
        else:
            self.signal_model[fname] = {}
            ndet = 0

        self.signal_model[fname][str(ndet)] = kwargs

        self.residuals += waveform(toas=self.toas, **kwargs)


    def radec_to_thetaphi(ra, dec):

        # RA in format : [H, M]
        # dec in format : [deg, arcmin]

        theta = np.pi/2 -  np.pi/180 * (dec[0] + dec[1]/60)
        phi = 2*np.pi * (ra[0] + ra[1]/60) / 24
        return theta, phi
    
    def thetaphi_to_radec(theta, phi):

        # theta angle
        # phi angle
        DEC = (theta - np.pi/2) * 180 / np.pi
        dec = [int(np.floor(DEC)), int((DEC-np.floor(DEC))*60)]
        RA = phi * 24 / (2*np.pi)
        ra = [int(np.floor(RA)), int((RA-np.floor(RA))*60)]
        return ra, dec

    def get_psrname(self):

        # RA
        h = int(24*self.phi/(2*np.pi))
        m = int((24*self.phi/(2*np.pi) - h) * 60)
        h = '0'+str(h) if len(str(h)) < 2 else str(h)
        m = '0'+str(m) if len(str(m)) < 2 else str(m)
        # DEC
        dec = round(180 * (np.pi/2 - self.theta) / np.pi, 2)
        sign = '+' if dec >= 0 else '-'
        decl, decr = str(abs(dec)).split('.')
        decl = '0'+str(decl) if len(str(decl)) < 2 else str(decl)
        decr = '0'+str(decr) if len(str(decr)) < 2 else str(decr)

        return 'J'+h+m+sign+decl+decr
    
    def make_noise_covariance_matrix(self):

        # make total noise covariance matrix

        if self.backends is None:
            toaerrs = np.sqrt(self.noisedict[self.name+'_efac']**2 * self.toaerrs**2 + 10**(2*self.noisedict[self.name+'_log10_tnequad']))
        else:
            toaerrs = np.zeros(len(self.toas))
            for backend in self.backends:
                mask_backend = self.backend_flags == backend
                toaerrs[mask_backend] = np.sqrt(self.noisedict[self.name+'_'+backend+'_efac']**2 * self.toaerrs[mask_backend]**2 + 10**(2*self.noisedict[self.name+'_'+backend+'_log10_tnequad']))
        white_cov = toaerrs**2

        red_cov = np.zeros((len(self.toas), len(self.toas)))
        if self.custom_model['RN'] is not None:
            red_cov += self.make_time_correlated_noise_cov(signal='red_noise')
        if self.custom_model['DM'] is not None:
            red_cov += self.make_time_correlated_noise_cov(signal='dm_gp')
        if self.custom_model['Sv'] is not None:
            red_cov += self.make_time_correlated_noise_cov(signal='chrom_gp')
        return white_cov, red_cov
    
    def draw_noise_model(self, residuals=None):
        
        white_cov, red_cov = self.make_noise_covariance_matrix()
        cov = np.diag(white_cov) + red_cov
        if residuals is None:
            resids = np.random.multivariate_normal(mean=np.zeros(len(self.toas)), cov=cov)
        else:
            inv_cov = np.linalg.inv(cov)
            resids = np.dot(red_cov.T, np.dot(inv_cov, residuals))
        return resids
    
    def reconstruct_signal(self, signals=None, freqf=1400):

        # reconstruct time domain realisation of injected noises and signals

        if signals is None:
            signals = [*self.signal_model]
        sig = np.zeros(len(self.toas))
        for signal in signals:
            if signal == 'cgw':
                for ncgw in len(self.signal_model['cgw']):
                    sig += det.cw_delay(self.toas, self.pos, self.pdist,
                                        **self.signal_model['cgw'][str(ncgw)])
            if (signal in ['red_noise', 'dm_gp', 'chrom_gp']) or ('common' in signal):
                f = self.signal_model[signal]['f']
                idx = self.signal_model[signal]['idx']
                df = np.diff(np.append(0., f))
                c = self.signal_model[signal]['fourier']
                for c_k, f_k, df_k in zip(c.T, f, df):
                    sig += df_k * c_k[0] * (freqf/self.freqs)**idx * np.cos(2*np.pi*f_k * self.toas)
                    sig += df_k * c_k[1] * (freqf/self.freqs)**idx * np.sin(2*np.pi*f_k * self.toas)
            if 'system_noise' in signal:
                backend = signal.split('system_noise_')[1]
                mask = self.backend_flags == backend
                f = self.signal_model[signal]['f']
                df = np.diff(np.append(0., f))
                c = self.signal_model[signal]['fourier']
                for c_k, f_k, df_k in zip(c.T, f, df):
                    sig[mask] += df_k * c_k[0] * np.cos(2*np.pi*f_k * self.toas[mask])
                    sig[mask] += df_k * c_k[1] * np.sin(2*np.pi*f_k * self.toas[mask])
        return sig
    
    def remove_signal(self, signals=None, freqf=1400):

        # remove signal from residuals, signal model and noisedict

        res = self.reconstruct_signal(signals, freqf=freqf)
        self.residuals -= res
        for signal in signals:
            self.signal_model.pop(signal)
            for key in [*self.noisedict]:
                if signal in key:
                    self.noisedict.pop(key)


def make_fake_array(npsrs=25, Tobs=None, ntoas=None, gaps=True, toaerr=None,
                    pdist=None, isotropic=False, backend_config=None,
                    noisedict=None, custom_model=None, ephem=None, f_psd=None,
                    add_ecorr=False, rng=None):
    
    if rng is None:
        rng = np.random.default_rng()

    if isotropic:
        i = np.arange(0, npsrs, dtype=float) + 0.5
        golden_ratio = (1 + 5**0.5)/2
        costhetas = 1 - 2*i/npsrs
        phis = np.mod(2 * np.pi * i / golden_ratio, 2*np.pi)
    else:
        costhetas = rng.uniform(-1., 1., size=npsrs)
        phis = rng.uniform(0., 2*np.pi, size=npsrs)

    # Observation time for each pulsar
    if Tobs is None:
        Tobs = rng.uniform(10, 20, size=npsrs)
    elif isinstance(Tobs, (float, int)):
        Tobs = Tobs * np.ones(npsrs)

    # Number of TOAs for each pulsar
    yr = 365.25 * 24 * 3600
    if ntoas is None:
        cadence = 7 * 24 * 3600
        F0 = rng.uniform(200, 300, size=npsrs)
        d_cadence = (F0 * cadence - np.floor(F0 * cadence)) / F0
        cadence = cadence - d_cadence
        ntoas = np.int32(Tobs * 365.25 * 24 * 3600 / cadence)
    elif isinstance(ntoas, (float, int)):
        F0 = 200 * np.ones(npsrs)
        ntoas = np.int32(ntoas * np.ones(npsrs))
        cadence = Tobs * yr / (ntoas - 1)

    # Make unevenly sampled TOAs
    Tmax = np.amax(Tobs)
    if gaps:
        gap_odds = [True, True, True, False] # one out of five
        keep = [rng.choice(gap_odds, size=ntoa) for ntoa in ntoas]
        toas = [(Tmax - Tobs[i])*yr + np.arange(1, ntoas[i]+1)*cadence[i] for i in range(npsrs)]
        toas = [toas[i][keep[i]] for i in range(npsrs)]
    else:
        toas = [(Tmax - Tobs[i])*yr + np.arange(1, ntoas[i]+1)*cadence[i] for i in range(npsrs)]

    if toaerr is None:
        toaerr = [None] * npsrs
    elif isinstance(toaerr, float):
        toaerr = toaerr * np.ones(npsrs)
    elif toaerr == 'randomize':
        toaerr = np.power(10, rng.uniform(-7., -5., size=npsrs))

    # Pulsar distances
    if pdist is None:
        dists = rng.uniform(0.5, 1.5, size=npsrs)
        pdist = [[dist, 0.2*dist] for dist in dists]
    elif isinstance(pdist, float):
        pdist = [[pdist, 0.2*pdist]] * npsrs

    # backend_config: one dict per pulsar, or a single dict applied to all,
    # or None for a simple default single-band backend
    if backend_config is None:
        # default: one backend, one frequency, no sub-bands
        backend_config = [default_backend_config] * npsrs
    elif isinstance(backend_config, dict):
        # same config for all pulsars
        backend_config = [backend_config] * npsrs
    elif isinstance(backend_config, list):
        # one config per pulsar — use as-is
        assert len(backend_config) == npsrs, \
            '"backend_config" list must have length "npsrs"'

    # Sanity checks
    assert len(Tobs)   == npsrs, '"Tobs" must be same size as "npsrs"'
    assert len(ntoas)  == npsrs, '"ntoas" must be same size as "npsrs"'
    assert len(toaerr) == npsrs, '"toaerr" must be same size as "npsrs"'
    assert len(pdist)  == npsrs, '"pdist" must be same size as "npsrs"'

    # Create pulsars
    psrs = []
    for i in range(npsrs):
        psr = Pulsar(
            toas[i],
            np.arccos(costhetas[i]), phis[i],
            pdist[i],
            backend_config=backend_config[i],
            custom_noisedict=noisedict,
            custom_model=custom_model,
            tm_params={'F0': (F0[i], rng.uniform(1e-13, 1e-12))},
            ephem=ephem,
            toaerr=toaerr[i]
        )
        logging.info('Creating psr %s', psr.name)
        psr.add_white_noise(add_ecorr=add_ecorr, rng=rng)

        try:
            psr.add_red_noise(spectrum='powerlaw',
                              log10_A=psr.noisedict[psr.name+'_red_noise_log10_A'],
                              gamma=psr.noisedict[psr.name+'_red_noise_gamma'],
                              f_psd=f_psd,
                              rng=rng)
        except:
            psr.add_red_noise(spectrum='powerlaw',
                              log10_A=rng.uniform(-17., -13),
                              gamma=rng.uniform(1, 5),
                              f_psd=f_psd,
                              rng=rng)

        try:
            psr.add_dm_noise(spectrum='powerlaw',
                             log10_A=psr.noisedict[psr.name+'_dm_gp_log10_A'],
                             gamma=psr.noisedict[psr.name+'_dm_gp_gamma'],
                             f_psd=f_psd,
                             rng=rng)
        except:
            psr.add_dm_noise(spectrum='powerlaw',
                             log10_A=rng.uniform(-17., -13),
                             gamma=rng.uniform(1, 5),
                             f_psd=f_psd,
                             rng=rng)

        try:
            psr.add_chromatic_noise(spectrum='powerlaw',
                                    log10_A=psr.noisedict[psr.name+'_chrom_gp_log10_A'],
                                    gamma=psr.noisedict[psr.name+'_chrom_gp_gamma'],
                                    f_psd=f_psd,
                                    rng=rng)
        except:
            psr.add_chromatic_noise(spectrum='powerlaw',
                                    log10_A=rng.uniform(-17., -13),
                                    gamma=rng.uniform(1, 5),
                                    f_psd=f_psd,
                                    rng=rng)

        psrs.append(psr)

    return psrs

# Plot sky positions of pulsars
def plot_pta(psrs, plot_name=True):

    ax = plt.axes(projection='mollweide')
    ax.grid(True, **{'alpha':0.25})
    plt.xticks(np.pi - np.linspace(0., 2*np.pi, 5), ['0h', '6h', '12h', '18h', '24h'], fontsize=14)
    plt.yticks(fontsize=14)
    for psr in psrs:
        s = 50 * (10**(-6) / np.mean(psr.toaerrs))
        plt.scatter(np.pi - np.array(psr.phi), np.pi/2 - np.array(psr.theta), marker=(5, 1), s=s, color='r')
        if plot_name:
            plt.annotate(psr.name, (np.pi - psr.phi + 0.05, np.pi/2 - psr.theta - 0.1), color='k', fontsize=10)
    plt.show()

# Copy existing array
def copy_array(psrs, custom_noisedict, custom_models=None):

    if custom_models is None:
        custom_models = {}
        for psr in psrs:
            custom_models[psr.name] = None

    fake_psrs = []
    for psr in psrs:
        fake_psr = Pulsar(psr.toas, psr.theta, phi=psr.phi, pdist=1., custom_model=custom_models[psr.name])
        fake_psr.name = psr.name
        fake_psr.toas = psr.toas
        fake_psr.toaerrs = psr.toaerrs
        fake_psr.residuals = psr.residuals
        fake_psr.Mmat = psr.Mmat
        fake_psr.fitpars = psr.fitpars
        fake_psr.pdist = psr.pdist
        fake_psr.backend_flags = psr.backend_flags
        fake_psr.backends = np.unique(psr.backend_flags)
        fake_psr.freqs = psr.freqs
        fake_psr.planetssb = psr.planetssb
        fake_psr.pos_t = psr.pos_t
        fake_psr.init_noisedict(custom_noisedict)
        # OR set fake_psr.noisedict to be custom noisedict
        fake_psrs.append(fake_psr)
    return fake_psrs