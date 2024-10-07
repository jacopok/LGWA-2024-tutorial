import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from GWFish.modules.detection import Network, new_tf_vectors
from GWFish.modules.fishermatrix import compute_network_errors
from GWFish.modules.waveforms import t_of_f_PN

rng = np.random.default_rng(seed=17)

LGWA_COLOR = '#377eb8'

param_dict = {
    'mass_1': 1.0,
    'mass_2': 1.0,
    'luminosity_distance': 2,
    'theta_jn': 5 / 6 * np.pi,
    'ra': 3.45,
    'dec': -0.41,
    'psi': 1.6,
    'phase': 0,
    'geocent_time': 1187008882,
    'max_frequency_cutoff': 0.5
}

network = Network(['LGWA'])

N = 20
dL = np.logspace(-2, 2, N)
fmax = np.logspace(-2, 0, N)
pop_errors = np.zeros((N, N, 9))
nav = 10
for k2 in range(N):
    param_dict['max_frequency_cutoff'] = fmax[k2]
    detector_frequencyvector = network.detectors[0].frequencyvector

    timevector = t_of_f_PN(param_dict, network.detectors[0].frequencyvector)
    timevector, frequencyvector, params = new_tf_vectors(param_dict, network.detectors[0], timevector, time_reset=True)
    network.detectors[0].frequencyvector = frequencyvector

    for k1 in range(N):
        param_dict['luminosity_distance'] = dL[k1]
        #param_dict['dec'] = np.arccos(rng.uniform(-1., 1.)) - np.pi / 2.
        #param_dict['ra'] = rng.uniform(0, 2. * np.pi)
        #param_dict['psi'] = rng.uniform(0, 2. * np.pi)

        parameters = pd.DataFrame.from_dict({k: v*np.array([1.]) for k, v in param_dict.items()})

        fisher_parameters = parameters

        fisher_parameters = fisher_parameters.drop(columns='geocent_time')
        fisher_parameters = fisher_parameters.drop(columns='max_frequency_cutoff')
        # breakpoint()
        detected, snr, errors, sky_localization = compute_network_errors(
            network, parameters, fisher_parameters=fisher_parameters.columns.to_numpy(), waveform_model='IMRPhenomXPHM')
        pop_errors[k1, k2, :] = pop_errors[k1, k2, :] + errors[0]/nav

    network.detectors[0].frequencyvector = detector_frequencyvector


plt.figure()
plt.pcolor(dL, fmax, np.transpose(np.log10(pop_errors[:, :, 0])), color=LGWA_COLOR, label='LGWA')
plt.xlabel('Luminosity distance [Mpc]')
plt.ylabel(r'Frequency, max-cutoff [Hz]')
cbr = plt.colorbar()
cbr.set_label('Error (log10), $m_1$ [$M_{sol}$]')
plt.xscale('log')
plt.yscale('log')
plt.xlim([dL[0], dL[-1]])
plt.ylim([fmax[0], fmax[-1]])
plt.savefig('Errors_dL_fmax_em1.png', dpi=600)

plot_edL = pop_errors[:, :, 2]
greater = plot_edL > dL[:, None]
plot_edL[greater] = np.NaN

plt.figure()
plt.pcolor(dL, fmax, np.transpose(np.log10(pop_errors[:, :, 2])), color=LGWA_COLOR, label='LGWA')
plt.xlabel('Luminosity distance [Mpc]')
plt.ylabel(r'Frequency, max-cutoff [Hz]')
cbr = plt.colorbar()
cbr.set_label('Error (log10), $d_L$ [Mpc]')
plt.xscale('log')
plt.yscale('log')
plt.xlim([dL[0], dL[-1]])
plt.ylim([fmax[0], fmax[-1]])
plt.savefig('Errors_dL_fmax_edL.png', dpi=600)