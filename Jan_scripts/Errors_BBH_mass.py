import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.cosmology import Planck18
import astropy.units as u
import astropy.cosmology.units as cu

from GWFish.modules.detection import Network, new_tf_vectors
from GWFish.modules.fishermatrix import compute_network_errors
from GWFish.modules.waveforms import t_of_f_PN

rng = np.random.default_rng(seed=17)

LGWA_COLOR = '#377eb8'

cosmology_model = Planck18

param_dict = {
    'mass_1': 1.0,
    'mass_2': 1.0,
    'luminosity_distance': 2,
    'theta_jn': 5 / 6 * np.pi,
    'ra': 3.45,
    'dec': -0.41,
    'psi': 1.6,
    'phase': 0,
    'geocent_time': 1187008882
}

network = Network(['LGWA'])

N = 20
dL = np.logspace(2, 5, N)
redshift = np.zeros_like(dL)

m = np.logspace(1, 4, N)
pop_errors = np.zeros((N, N, 9))
nav = 10
for k1 in range(N):
    param_dict['luminosity_distance'] = dL[k1]
    y = dL[k1] * u.Mpc
    z = y.to(cu.redshift, cu.redshift_distance(Planck18, kind="luminosity"))
    redshift[k1] = z
    for k2 in range(N):
        param_dict['mass_1'] = m[k2]
        param_dict['mass_2'] = m[k2]
        detector_frequencyvector = network.detectors[0].frequencyvector

        timevector = t_of_f_PN(param_dict, network.detectors[0].frequencyvector)
        timevector, frequencyvector, params = new_tf_vectors(param_dict, network.detectors[0], timevector,
                                                             time_reset=True)
        network.detectors[0].frequencyvector = frequencyvector

        #param_dict['dec'] = np.arccos(rng.uniform(-1., 1.)) - np.pi / 2.
        #param_dict['ra'] = rng.uniform(0, 2. * np.pi)
        #param_dict['psi'] = rng.uniform(0, 2. * np.pi)

        parameters = pd.DataFrame.from_dict({k: v*np.array([1.]) for k, v in param_dict.items()})

        detected, snr, errors, sky_localization = compute_network_errors(network, parameters, waveform_model='IMRPhenomXPHM')
        pop_errors[k1, k2, :] = pop_errors[k1, k2, :] + errors[0]/nav

    network.detectors[0].frequencyvector = detector_frequencyvector


plt.figure()
plt.loglog(redshift, pop_errors[:, -1, 0], color=LGWA_COLOR, label='LGWA')
plt.xlabel('Redshift')
plt.ylabel(r'Error, $m_1$ [$M_{sol}$]')
plt.legend()
#plt.xlim([6e-3, 1])
#plt.ylim([0.2, 200])
plt.savefig('Errors_m1.png', dpi=600)

plt.figure()
plt.pcolor(redshift, m, np.transpose(np.log10(pop_errors[:, :, 0])), color=LGWA_COLOR, label='LGWA')
plt.xlabel('Redshift')
plt.ylabel(r'Component mass [$M_{sol}$]')
cbr = plt.colorbar()
cbr.set_label('Error (log10), $m_1$ [$M_{sol}$]')
plt.xscale('log')
plt.yscale('log')
plt.xlim([redshift[0], redshift[-1]])
plt.ylim([m[0], m[-1]])
plt.savefig('Errors_BBH_z-m_em1.png', dpi=600)

plot_edL = pop_errors[:, :, 2]
greater = plot_edL > dL[:, None]
plot_edL[greater] = np.NaN

plt.figure()
plt.pcolor(redshift, m, np.transpose(np.log10(pop_errors[:, :, 2])), color=LGWA_COLOR, label='LGWA')
plt.xlabel('Redshift')
plt.ylabel(r'Component mass [$M_{sol}$]')
cbr = plt.colorbar()
cbr.set_label('Error (log10), $d_L$ [Mpc]')
plt.xscale('log')
plt.yscale('log')
plt.xlim([redshift[0], redshift[-1]])
plt.ylim([m[0], m[-1]])
plt.savefig('Errors_BBH_dL-m_edL.png', dpi=600)
