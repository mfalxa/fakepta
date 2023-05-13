import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
from enterprise_extensions import deterministic as det
import scipy.constants as sc


class Pulsar:

    def __init__(self, toas, toaerr, theta, phi, pdist, custom_noisedict=None, custom_model=None, backends=None):

        self.toas = toas
        self.toaerrs = toaerr * np.ones(len(self.toas))
        self.residuals = np.zeros(len(self.toas))
        self.Tspan = np.amax(self.toas) - np.amin(self.toas)
        self.backends = backends
        if custom_model is None:
            self.custom_model = {'RN':30, 'DM':100, 'Sv':None}
        else:
            self.custom_model = custom_model
        self.freqs = abs(np.random.normal(loc=1400, scale=200, size=len(self.toas)))
        self.flags = {}
        self.flags['pta'] = 'FAKE'
        # Initialize useless design matrix to avoid bug with enterprise if timing model included
        self.Mmat = np.ones((len(self.toas), 2))
        if self.backends is not None:
            self.backend_flags = np.random.choice(self.backends, size=len(self.toas), replace=True)
        self.theta = theta
        self.phi = phi
        self.pos = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
        self.pdist = pdist
        self.name = self.get_psrname()
        self.init_noisedict(custom_noisedict)

    def init_noisedict(self, custom_noisedict=None):

        if custom_noisedict is None:
            noisedict = {}
            if self.backends is None:
                noisedict[self.name+'_efac'] = 1.
                noisedict[self.name+'_log10_tnequad'] = -8.
                noisedict[self.name+'_log10_t2equad'] = -8.
            else:
                for backend in self.backends:
                    noisedict[self.name+'_'+backend+'_efac'] = 1.
                    noisedict[self.name+'_'+backend+'_log10_tnequad'] = -8.
                    noisedict[self.name+'_'+backend+'_log10_t2equad'] = -8.
            self.noisedict = noisedict
        else:
            keys = [*custom_noisedict]
            noisedict = {}
            for key in keys:
                if self.name in key:
                    noisedict[key] = custom_noisedict[key]
            self.noisedict = noisedict

    def update_position(self, theta, phi, update_name=False):
        self.theta = theta
        self.phi = phi
        self.pos = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
        if update_name:
            self.name = self.get_psrname()

    def make_ideal(self):

        self.residuals = np.zeros(len(self.toas))

    def add_white_noise(self, randomize=False):

        if randomize:
            for key in [*self.noisedict]:
                if 'efac' in key:
                    self.noisedict[key] = np.random.uniform(-0.5, 2.5)
                if 'equad' in key:
                    self.noisedict[key] = np.random.uniform(-8., -5.)
        if self.backends is None:
            self.toaerrs = np.sqrt(self.noisedict[self.name+'_efac']**2 * self.toaerrs**2 + 10**(2*self.noisedict[self.name+'_log10_tnequad']))
            self.residuals += np.random.normal(scale=self.toaerrs)
        else:
            for backend in self.backends:
                mask_backend = self.backend_flags == backend
                self.toaerrs[mask_backend] = np.sqrt(self.noisedict[self.name+'_'+backend+'_efac']**2 * self.toaerrs[mask_backend]**2 + 10**(2*self.noisedict[self.name+'_'+backend+'_log10_tnequad']))
                self.residuals[mask_backend] += np.random.normal(scale=self.toaerrs[mask_backend])

    def add_red_noise(self, gp=True, log10_A=None, gamma=None, cos_phase=True, rand_coeff=False):

        rn_components = self.custom_model['RN']
        if rn_components is not None:
            if gp:
                self.add_time_correlated_noise_gp(signal='red_noise', log10_A=log10_A, gamma=gamma, idx=0., components=rn_components)
            else:
                self.add_time_correlated_noise(signal='red_noise', log10_A=log10_A, gamma=gamma, idx=0., components=rn_components, cos_phase=cos_phase, rand_coeff=rand_coeff)

    def add_dm_noise(self, gp=True, log10_A=None, gamma=None, cos_phase=True, rand_coeff=False):

        dm_components = self.custom_model['DM']
        if dm_components is not None:
            if gp:
                self.add_time_correlated_noise_gp(signal='dm_gp', log10_A=log10_A, gamma=gamma, idx=2, components=dm_components)
            else:
                self.add_time_correlated_noise(signal='dm_gp', log10_A=log10_A, gamma=gamma, idx=2, components=dm_components, cos_phase=cos_phase, rand_coeff=rand_coeff)

    def add_chromatic_noise(self, gp=True, log10_A=None, gamma=None, cos_phase=True, rand_coeff=False):

        sv_components = self.custom_model['Sv']
        if sv_components is not None:
            if gp:
                self.add_time_correlated_noise_gp(signal='chrom_gp', log10_A=log10_A, gamma=gamma, idx=4, components=sv_components)
            else:
                self.add_time_correlated_noise(signal='chrom_gp', log10_A=log10_A, gamma=gamma, idx=4, components=sv_components, cos_phase=cos_phase, rand_coeff=rand_coeff)

    def add_system_noise(self, backend=None, gp=True, components=30, log10_A=None, gamma=None, cos_phase=True, rand_coeff=False):

        rn_components = components
        if rn_components is not None:
            if gp:
                self.add_time_correlated_noise_gp(signal='red_noise', log10_A=log10_A, gamma=gamma, idx=0., components=rn_components, backend=backend)
            else:
                self.add_time_correlated_noise(signal='red_noise', log10_A=log10_A, gamma=gamma, idx=0., components=rn_components, cos_phase=cos_phase, rand_coeff=rand_coeff, backend=backend)

    def add_time_correlated_noise(self, signal='', log10_A=None, gamma=None, idx=4, components=None, freqf=1400, cos_phase=True, rand_coeff=False, backend=None):

        if backend is not None:
            signal = backend + '_' + signal
            mask = self.backend_flags == backend
            if not np.any(mask):
                print(backend, 'not found in backend_flags.')
                return
        else:
            mask = np.ones(len(self.toas), dtype='bool')
        if log10_A is None:
            if self.name+'_'+str(signal)+'_log10_A' in self.noisedict:
                log10_A = self.noisedict[self.name+'_'+str(signal)+'_log10_A']
            else:
                log10_A = np.random.uniform(-17., -13.)
                self.noisedict[self.name+'_'+str(signal)+'_log10_A'] = log10_A
        if gamma is None:
            if self.name+'_'+str(signal)+'_gamma' in self.noisedict:
                gamma = self.noisedict[self.name+'_'+str(signal)+'_gamma']
            else:
                gamma = np.random.uniform(1., 6.)
                self.noisedict[self.name+'_'+str(signal)+'_gamma'] = gamma
        f = np.array([np.arange(1, components+1), np.arange(1, components+1)]) / self.Tspan
        fyr = 1/sc.Julian_year
        psd = (10**log10_A)** 2 / (12.0 * np.pi**2) * fyr**(gamma-3) * f**(-gamma) / self.Tspan
        if rand_coeff:
            coeffs = np.random.normal(loc=0., scale=np.sqrt(psd))
        else:
            coeffs = np.sqrt(psd)
        if cos_phase:
            for i in range(components):
                phase = np.random.uniform(0., 2*np.pi)
                self.residuals[mask] += (freqf/self.freqs)**idx * coeffs[0, i] * np.cos(2*np.pi*f[0, i]*self.toas[mask] + phase) * np.sqrt(2)
        else:
            for i in range(components):
                self.residuals[mask] += (freqf/self.freqs)**idx * coeffs[0, i] * np.cos(2*np.pi*f[0, i]*self.toas[mask])
                self.residuals[mask] += (freqf/self.freqs)**idx * coeffs[1, i] * np.sin(2*np.pi*f[1, i]*self.toas[mask])

    def add_time_correlated_noise_gp(self, signal='', log10_A=None, gamma=None, idx=4, components=None, freqf=1400, backend=None):

        if backend is not None:
            signal = backend + '_' + signal
            mask = self.backend_flags == backend
            if not np.any(mask):
                print(backend, 'not found in backend_flags.')
                return
        else:
            mask = np.ones(len(self.toas), dtype='bool')
        if log10_A is None:
            if self.name+'_'+str(signal)+'_log10_A' in self.noisedict:
                log10_A = self.noisedict[self.name+'_'+str(signal)+'_log10_A']
            else:
                log10_A = np.random.uniform(-17., -13.)
                self.noisedict[self.name+'_'+str(signal)+'_log10_A'] = log10_A
        if gamma is None:
            if self.name+'_'+str(signal)+'_gamma' in self.noisedict:
                gamma = self.noisedict[self.name+'_'+str(signal)+'_gamma']
            else:
                gamma = np.random.uniform(1., 6.)
                self.noisedict[self.name+'_'+str(signal)+'_gamma'] = gamma
        f = np.arange(1, components+1) / self.Tspan
        fyr = 1/sc.Julian_year
        psd = (10**log10_A)** 2 / (12.0 * np.pi**2) * fyr**(gamma-3) * f**(-gamma) / self.Tspan
        psd = np.repeat(psd, 2)
        basis = np.zeros((len(self.toas[mask]), 2*components))
        for i in range(components):
            basis[:, 2*i] = (freqf/self.freqs)**idx * np.cos(2*np.pi*f[i]*self.toas[mask])
            basis[:, 2*i+1] = (freqf/self.freqs)**idx * np.sin(2*np.pi*f[i]*self.toas[mask])
        cov = np.dot(basis, np.dot(np.diag(psd), basis.T))
        gp = np.random.multivariate_normal(mean=np.zeros(len(self.toas[mask])), cov=cov)
        self.residuals[mask] += gp
        
    def add_cgw(self, costheta, phi, cosinc, log10_mc, log10_fgw, log10_h, phase0, psi, psrterm=False):

        cgw = det.cw_delay(self.toas, self.pos, self.pdist,
                            cos_gwtheta=costheta, gwphi=phi,
                            cos_inc=cosinc, log10_mc=log10_mc, 
                            log10_fgw=log10_fgw, evolve=True,
                            log10_h=log10_h, phase0=phase0, 
                            psi=psi, psrTerm=psrterm)
        self.residuals += cgw

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


def make_fake_array(npsrs=25, Tobs=None, ntoas=None, gaps=True, toaerr=None, pdist=None, isotropic=False, backends=None, noisedict=None, custom_models=None, gp_noises=True):

    if isotropic:
        # Fibonacci sequence on sphere
        i = np.arange(0, npsrs, dtype=float) + 0.5
        golden_ratio = (1 + 5**0.5)/2
        costhetas = 1 - 2*i/npsrs
        phis = np.mod(2 * np.pi * i / golden_ratio, 2*np.pi)
    else:
        costhetas = np.random.uniform(-1., 1., size=npsrs)
        phis = np.random.uniform(0., 2*np.pi, size=npsrs)

    # Observation time for each pulsar
    if Tobs is None:
        Tobs = np.random.uniform(10, 20, size=npsrs)
    elif isinstance(Tobs, float) or isinstance(Tobs, int):
        Tobs = Tobs * np.ones(npsrs)

    # Number of TOAs for each pulsar
    if ntoas is None:
        ntoas = np.random.randint(1000, 5000, npsrs)
    elif isinstance(ntoas, float) or isinstance(ntoas, int):
        ntoas = np.int32(ntoas * np.ones(npsrs))

    # Init TOAs from latest observation time
    yr = 365.25*24*3600
    Tmax = np.amax(Tobs)

    # Make unevenly sampled TOAs if gaps is True
    if gaps:
        toas = [np.linspace((Tmax - Tobs[i])*yr, Tmax*yr, 4*ntoas[i]) for i in range(npsrs)]
        toas = [toas[i][np.sort(np.random.choice(np.arange(len(toas[i])), replace=False, size=ntoas[i]))] for i in range(npsrs)]
    else:
        toas = [np.linspace((Tmax - Tobs[i])*yr, Tmax*yr, ntoas[i]) for i in range(npsrs)]
    if toaerr is None:
        toaerr = np.power(10, np.random.uniform(-7., -5., size=npsrs))
    elif isinstance(toaerr, float):
        toaerr = toaerr * np.ones(npsrs)

    # Init pulsar distances
    if pdist is None:
        dists = np.random.uniform(0.5, 1.5, size=npsrs)
        pdist = [[dist, 0.2*dist] for dist in dists]
    elif isinstance(pdist, float):
        pdist = [[pdist, 0.2*pdist]] * npsrs

    # Init backends
    if backends is None:
        backends = []
        for _ in range(npsrs):
            n_backends = np.random.randint(1, 5)
            backends.append(['backend_'+str(k) for k in range(n_backends)])
    elif isinstance(backends, str):
        backends = [[backends]] * npsrs
    elif isinstance(backends, list):
        if not isinstance(backends[0], list):
            backends = [backends] * npsrs

    assert (len(Tobs) == npsrs), '"Tobs" must be same size as "npsrs"'
    assert (len(ntoas) == npsrs), '"ntoas" must be same size as "npsrs"'
    assert (len(toaerr) == npsrs), '"toaerr" must be same size as "npsrs"'
    assert (len(pdist) == npsrs), '"pdist" must be same size as "npsrs"'
    assert (len(backends) == npsrs), '"backends" must be same size as "npsrs"'

    # Create pulsars and add noises
    psrs = []
    for i in range(npsrs):
        if custom_models is None:
            custom_model = None
        psr = Pulsar(toas[i], toaerr[i], np.arccos(costhetas[i]), phis[i], pdist[i], backends=backends[i], custom_noisedict=noisedict, custom_model=custom_model)
        print('Creating psr', psr.name)
        psr.add_white_noise()
        psr.add_red_noise(gp=gp_noises)
        psr.add_dm_noise(gp=gp_noises)
        psr.add_chromatic_noise(gp=gp_noises)
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
        fake_psr = Pulsar(psr.toas, 10**(-6), psr.theta, phi=psr.phi, pdist=1., backends=np.unique(psr.backend_flags), custom_model=custom_models[psr.name])
        fake_psr.name = psr.name
        fake_psr.toaerrs = psr.toaerrs
        fake_psr.residuals = psr.residuals
        fake_psr.Mmat = psr.Mmat
        fake_psr.pdist = psr.pdist
        fake_psr.backend_flags = psr.backend_flags
        fake_psr.backends = np.unique(psr.backend_flags)
        fake_psr.freqs = psr.freqs
        fake_psr.init_noisedict(custom_noisedict)
        # OR set fake_psr.noisedict to be custom noisedict
        fake_psrs.append(fake_psr)
    return fake_psrs
