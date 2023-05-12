from .fake_pta import Pulsar
import numpy as np
import scipy.constants as sc
from scipy.interpolate import interp1d

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

def add_correlated_red_noise_gp(psrs, orf='hd', log10_A=-15., gamma=13/3, rn_components=30):

    Tspan = np.amax([psr.toas.max() for psr in psrs]) - np.amin([psr.toas.min() for psr in psrs])
    f = np.arange(1, rn_components+1) / Tspan
    fyr = 1/sc.Julian_year
    psd_rn = (10**log10_A)** 2 / (12.0 * np.pi**2) * fyr**(gamma-3) * f**(-gamma) / Tspan
    psd_rn = np.repeat(psd_rn, 2)
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
    orf_funcs = {'hd':hd, 'monopole':monopole, 'dipole':dipole}
    orfs = orf_funcs[orf](psrs)
    for i in range(len(psrs)):
        for j in range(len(psrs)):
            cov_ij = np.dot(basis[i], np.dot(np.diag(orfs[i, j]*psd_rn), basis[j].T))
            cov[i*ntoas:(i+1)*ntoas, j*ntoas:(j+1)*ntoas] = cov_ij
    gwb_gp = np.random.multivariate_normal(mean=np.zeros(len(psrs)*ntoas), cov=cov)
    for k in range(len(psrs)):
        toas = np.linspace(psrs[k].toas.min(), psrs[k].toas.max(), ntoas)
        f = interp1d(toas, gwb_gp[k*ntoas:(k+1)*ntoas], kind='cubic')
        psrs[k].residuals += f(psrs[k].toas)

