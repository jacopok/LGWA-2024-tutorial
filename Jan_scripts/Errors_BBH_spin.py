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
    'a_1': 0.8,
    'tilt_1': np.pi/3,
    'a_2': 0.2,
    'tilt_2': -np.pi/5,
    'geocent_time': 1187008882
}

network = Network(['LGWA'])

N = 25
dL = np.logspace(2, 5, N)
redshift = np.zeros_like(dL)

m = np.logspace(1, 6, N)
pop_errors = np.zeros((N, N, 13))
pop_errors = np.zeros((N, N, 10))
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
        fisher_parameters = list(parameters.keys())
        fisher_parameters.remove('geocent_time')
        fisher_parameters.remove('ra')
        fisher_parameters.remove('dec')


        detected, snr, errors, sky_localization = compute_network_errors(
            network, parameters, fisher_parameters=fisher_parameters, waveform_model='IMRPhenomXPHM')
        pop_errors[k1, k2, :] = pop_errors[k1, k2, :] + errors[0]/nav

    network.detectors[0].frequencyvector = detector_frequencyvector


plot_edL = pop_errors[:, :, 6]
greater = plot_edL > 1
plot_edL[greater] = np.NaN

plt.figure()
plt.pcolor(redshift, m, np.transpose(np.log10(pop_errors[:, :, 6])), color=LGWA_COLOR, label='LGWA')
plt.xlabel('Redshift')
plt.ylabel(r'Component mass [$M_{sol}$]')
cbr = plt.colorbar()
cbr.set_label('Error (log10), spin $a_1$')
plt.xscale('log')
plt.yscale('log')
plt.xlim([redshift[0], redshift[-1]])
plt.ylim([m[0], m[-1]])
plt.savefig('Errors_BBH_z-m_ea1.png', dpi=600)

plot_edL = pop_errors[:, :, 7]
greater = plot_edL > np.pi/2
plot_edL[greater] = np.NaN

plt.figure()
plt.pcolor(redshift, m, np.transpose(np.log10(pop_errors[:, :, 7])), color=LGWA_COLOR, label='LGWA')
plt.xlabel('Redshift')
plt.ylabel(r'Component mass [$M_{sol}$]')
cbr = plt.colorbar()
cbr.set_label(r'Error (log10), spin tilt $\tau_1$')
plt.xscale('log')
plt.yscale('log')
plt.xlim([redshift[0], redshift[-1]])
plt.ylim([m[0], m[-1]])
plt.savefig('Errors_BBH_z-m_etau1.png', dpi=600)