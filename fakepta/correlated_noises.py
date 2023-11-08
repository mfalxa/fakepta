from .fake_pta import Pulsar
import numpy as np
import scipy.constants as sc
from scipy.interpolate import interp1d
import healpy as hp

# Misc functions
def get_correlation(psr_a, psr_b, res_a, res_b):

    angle = np.arccos(np.dot(psr_a.pos, psr_b.pos))
    corr = np.dot(res_a, res_b) / len(res_a)

    return corr, angle

def get_correlations(psrs, res):

    corrs = np.array([])
    angles = np.array([])
    autocorrs = np.array([])
    for i in range(len(psrs)):
        for j in range(i+1):
            c, a = get_correlation(psrs[i], psrs[j], res[i], res[j])
            if i == j:
                autocorrs = np.append(autocorrs, c)
            else:
                corrs = np.append(corrs, c)
                angles = np.append(angles, a)
    return corrs, angles, autocorrs

def bin_curve(corrs, angles, bins):

    edges = np.linspace(0., np.pi, bins+1)
    bin_angles = edges[:-1] + 0.5*(edges[1]-edges[0])
    mean = []
    std = []
    for i in range(bins):
        mask = angles > edges[i]
        mask *= angles < edges[i+1]
        mean.append(np.mean(corrs[mask]))
        std.append(np.std(corrs[mask]))
    return np.array(mean), np.array(std), np.array(bin_angles)

# ORFs
def create_gw_antenna_pattern(pos, gwtheta, gwphi):

    m = np.array([np.sin(gwphi), -np.cos(gwphi), np.zeros(len(gwphi))]).T
    n = np.array([-np.cos(gwtheta) * np.cos(gwphi), -np.cos(gwtheta) * np.sin(gwphi), np.sin(gwtheta)]).T
    omhat = np.array([-np.sin(gwtheta) * np.cos(gwphi), -np.sin(gwtheta) * np.sin(gwphi), -np.cos(gwtheta)]).T

    fplus = 0.5 * (np.dot(m, pos) ** 2 - np.dot(n, pos) ** 2) / (1 + np.dot(omhat, pos))
    fcross = (np.dot(m, pos) * np.dot(n, pos)) / (1 + np.dot(omhat, pos))
    cosMu = -np.dot(omhat, pos)

    return fplus, fcross, cosMu

def hd(psrs):
    orfs = np.zeros((len(psrs), len(psrs)))
    for i in range(len(psrs)):
        for j in range(len(psrs)):
            if i == j:
                orfs[i, j] = 1.
            else:
                omc2 = (1 - np.dot(psrs[i].pos, psrs[j].pos)) / 2
                orfs[i, j] =  1.5 * omc2 * np.log(omc2) - 0.25 * omc2 + 0.5
    return orfs

def anisotropic(psrs, h_map):

    orfs = np.zeros((len(psrs), len(psrs)))
    npixels = len(h_map)
    pixels = hp.pix2ang(hp.npix2nside, np.arange(npixels), nest=False)
    gwtheta = pixels[0]
    gwphi = pixels[1]
    for i in range(len(psrs)):
        for j in range(len(psrs)):
            if i == j:
                k_ab = 2.
            else:
                k_ab = 1.
            fp_a, fc_a, _ = create_gw_antenna_pattern(psrs.pos[i], gwtheta, gwphi)
            fp_b, fc_b, _ = create_gw_antenna_pattern(psrs.pos[j], gwtheta, gwphi)
            orfs[i, j] = 1.5 * k_ab * np.sum((fp_a*fp_b + fc_a*fc_b) * h_map) / npixels

def monopole(psrs):
    npsr = len(psrs)
    return np.ones((npsr, npsr))

def dipole(psrs):
    orfs = np.zeros((len(psrs), len(psrs)))
    for i in range(len(psrs)):
        for j in range(len(psrs)):
            if i == j:
                orfs[i, j] = 1.
            else:
                omc2 = np.dot(psrs[i].pos, psrs[j].pos)
                orfs[i, j] = omc2
    return orfs

def curn(psrs):
    npsr = len(psrs)
    return np.eye(npsr)

# PSD
def powerlaw(f, log10_A, gamma):

    fyr = 1/sc.Julian_year
    psd_rn = (10**log10_A)** 2 / (12.0 * np.pi**2) * fyr**(gamma-3) * f**(-gamma)
    return psd_rn

# Noise generating function
def add_correlated_red_noise_gp(psrs, orf='hd', log10_A=-15., gamma=13/3, rn_components=30, custom_psd=None, f_psd=None, h_map=None):

    Tspan = np.amax([psr.toas.max() for psr in psrs]) - np.amin([psr.toas.min() for psr in psrs])
    if f_psd is None:
        f = np.arange(1, rn_components+1) / Tspan
    else:
        f = f_psd
    df = np.diff(np.append(0., f))
    if custom_psd is not None:
        assert f_psd is None, '"f_psd" must not be None. The frequencies "f_psd" correspond to frequencies where the "custom_psd" is evaluated.'
        assert len(custom_psd) == len(f_psd), '"custom_psd" and "f_psd" must be same length. The frequencies "f_psd" correspond to frequencies where the "custom_psd" is evaluated.'
        psd_gwb = custom_psd * df
    else:
        psd_gwb = powerlaw(f, log10_A, gamma) * df
    psd_gwb = np.repeat(psd_gwb, 2)
    ntoas = 100
    cov = np.zeros((len(psrs)*ntoas, len(psrs)*ntoas))
    basis = []
    for psr in psrs:
        basis_psr = np.zeros((ntoas, 2*rn_components))
        toas = np.linspace(psr.toas.min(), psr.toas.max(), ntoas)
        for i in range(rn_components):
            basis_psr[:, 2*i] = np.cos(2*np.pi*f[i]*toas)
            basis_psr[:, 2*i+1] = np.sin(2*np.pi*f[i]*toas)
        basis.append(basis_psr)
    orf_funcs = {'hd':hd, 'monopole':monopole, 'dipole':dipole, 'curn':curn}
    if orf in [*orf_funcs]:
        orfs = orf_funcs[orf](psrs)
    elif orf == 'anisotropic':
        orfs = anisotropic(psrs, h_map)
    for i in range(len(psrs)):
        for j in range(len(psrs)):
            cov_ij = np.dot(basis[i], np.dot(np.diag(orfs[i, j]*psd_gwb), basis[j].T))
            cov[i*ntoas:(i+1)*ntoas, j*ntoas:(j+1)*ntoas] = cov_ij
    gwb_gp = np.random.multivariate_normal(mean=np.zeros(len(psrs)*ntoas), cov=cov)
    for k in range(len(psrs)):
        toas = np.linspace(psrs[k].toas.min(), psrs[k].toas.max(), ntoas)
        f = interp1d(toas, gwb_gp[k*ntoas:(k+1)*ntoas], kind='cubic')
        psrs[k].residuals += f(psrs[k].toas)

