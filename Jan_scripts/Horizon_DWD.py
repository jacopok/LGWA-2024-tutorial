import numpy as np
import matplotlib.pyplot as plt

from GWFish.modules.detection import Detector
import GWFish.modules.horizon as hz

rng = np.random.default_rng(seed=17)

ET_COLOR = '#4daf4a'
LGWA_COLOR = '#377eb8'
LISA_COLOR = '#e41a1c'

param_dict = {
    'mass_1': 1, 
    'mass_2': 1,
    'theta_jn': 0,
    'ra': 3.45,
    'dec': -0.41,
    'psi': 1.6,
    'phase': 0,
    'geocent_time': 186908882,
    'max_frequency_cutoff': 1
}

N = 17

detector = Detector('LGWA')
#detector.mission_lifetime = 1*3.16e7
ff_lgwa = np.logspace(-1.7, np.log10(0.3), N)
hor_dwd_dl_lgwa_wdwd = np.zeros((N, 2))
for k in range(N):
    for n in range(5):
        param_dict['max_frequency_cutoff'] = ff_lgwa[k]
        param_dict['dec'] = np.arccos(rng.uniform(-1., 1.)) - np.pi / 2.
        param_dict['ra'] = rng.uniform(0, 2. * np.pi)
        param_dict['psi'] = rng.uniform(0, 2. * np.pi)
        hor_dl, hor_z = hz.horizon(param_dict, detector, target_SNR=4., waveform_model='TaylorF2', redefine_tf_vectors=True)
        hor_dwd_dl_lgwa_wdwd[k, 0] = np.maximum(hor_dwd_dl_lgwa_wdwd[k, 0], hor_dl)
        hor_dl, hor_z = hz.horizon(param_dict, detector, target_SNR=9., waveform_model='TaylorF2', redefine_tf_vectors=True)
        hor_dwd_dl_lgwa_wdwd[k, 1] = np.maximum(hor_dwd_dl_lgwa_wdwd[k, 1], hor_dl)

detector.mission_lifetime = 3*3.16e7
hor_dwd_dl_lgwa_wdwd_3yr = np.zeros((N, 2))
for k in range(N):
    for n in range(15):
        param_dict['max_frequency_cutoff'] = ff_lgwa[k]
        param_dict['dec'] = np.arccos(rng.uniform(-1., 1.)) - np.pi / 2.
        param_dict['ra'] = rng.uniform(0, 2. * np.pi)
        param_dict['psi'] = rng.uniform(0, 2. * np.pi)
        hor_dl, hor_z = hz.horizon(param_dict, detector, target_SNR=4., waveform_model='TaylorF2', redefine_tf_vectors=True)
        hor_dwd_dl_lgwa_wdwd_3yr[k, 0] = np.maximum(hor_dwd_dl_lgwa_wdwd_3yr[k, 0], hor_dl)
        hor_dl, hor_z = hz.horizon(param_dict, detector, target_SNR=9., waveform_model='TaylorF2', redefine_tf_vectors=True)
        hor_dwd_dl_lgwa_wdwd_3yr[k, 1] = np.maximum(hor_dwd_dl_lgwa_wdwd_3yr[k, 1], hor_dl)

print('WD-WD LGWA calculated')

network = Detector('LISA')
ff_lisa = np.logspace(np.log10(6e-3), np.log10(6e-2), N)
hor_dwd_dl_lisa_wdwd = np.zeros((N, 2))
for k in range(N):
    for n in range(5):
        param_dict['max_frequency_cutoff'] = ff_lisa[k]
        param_dict['dec'] = np.arccos(rng.uniform(-1., 1.)) - np.pi / 2.
        param_dict['ra'] = rng.uniform(0, 2. * np.pi)
        param_dict['psi'] = rng.uniform(0, 2. * np.pi)
        hor_dl, hor_z = hz.horizon(param_dict, network, target_SNR=4., waveform_model='TaylorF2', redefine_tf_vectors=True)
        hor_dwd_dl_lisa_wdwd[k, 0] = np.maximum(hor_dwd_dl_lisa_wdwd[k, 0], hor_dl)
        hor_dl, hor_z = hz.horizon(param_dict, network, target_SNR=9., waveform_model='TaylorF2', redefine_tf_vectors=True)
        hor_dwd_dl_lisa_wdwd[k, 1] = np.maximum(hor_dwd_dl_lisa_wdwd[k, 1], hor_dl)

print('WD-WD LISA calculated')


param_dict['mass_1'] = 1.4

detector = Detector('LGWA')
hor_dwd_dl_lgwa_wdns = np.zeros((N, 2))
for k in range(N):
    for n in range(5):
        param_dict['max_frequency_cutoff'] = ff_lgwa[k]
        param_dict['dec'] = np.arccos(rng.uniform(-1., 1.)) - np.pi / 2.
        param_dict['ra'] = rng.uniform(0, 2. * np.pi)
        param_dict['psi'] = rng.uniform(0, 2. * np.pi)
        hor_dl, hor_z = hz.horizon(param_dict, detector, target_SNR=4., waveform_model='TaylorF2', redefine_tf_vectors=True)
        hor_dwd_dl_lgwa_wdns[k, 0] = np.maximum(hor_dwd_dl_lgwa_wdns[k, 0], hor_dl)
        hor_dl, hor_z = hz.horizon(param_dict, detector, target_SNR=9., waveform_model='TaylorF2', redefine_tf_vectors=True)
        hor_dwd_dl_lgwa_wdns[k, 1] = np.maximum(hor_dwd_dl_lgwa_wdns[k, 1], hor_dl)

print('WD-NS LGWA calculated')

network = Detector('LISA')
ff_lisa = np.logspace(np.log10(6e-3), np.log10(6e-2), N)
hor_dwd_dl_lisa_wdns = np.zeros((N, 2))
for k in range(N):
    for n in range(5):
        param_dict['max_frequency_cutoff'] = ff_lisa[k]
        param_dict['dec'] = np.arccos(rng.uniform(-1., 1.)) - np.pi / 2.
        param_dict['ra'] = rng.uniform(0, 2. * np.pi)
        param_dict['psi'] = rng.uniform(0, 2. * np.pi)
        hor_dl, hor_z = hz.horizon(param_dict, network, target_SNR=4., waveform_model='TaylorF2', redefine_tf_vectors=True)
        hor_dwd_dl_lisa_wdns[k, 0] = np.maximum(hor_dwd_dl_lisa_wdns[k, 0], hor_dl)
        hor_dl, hor_z = hz.horizon(param_dict, network, target_SNR=9., waveform_model='TaylorF2', redefine_tf_vectors=True)
        hor_dwd_dl_lisa_wdns[k, 1] = np.maximum(hor_dwd_dl_lisa_wdns[k, 1], hor_dl)

print('WD-NS LISA calculated')

plt.figure()
plt.loglog(ff_lgwa, hor_dwd_dl_lgwa_wdns[:, 0], color=LGWA_COLOR, label='LGWA - WDNS (10 yr)')
plt.fill_between(ff_lgwa, hor_dwd_dl_lgwa_wdns[:, 0], color=LGWA_COLOR, alpha=0.2)
plt.loglog(ff_lgwa, hor_dwd_dl_lgwa_wdns[:, 1], color=LGWA_COLOR)
plt.fill_between(ff_lgwa, hor_dwd_dl_lgwa_wdns[:, 1], color=LGWA_COLOR, alpha=0.2)
plt.loglog(ff_lgwa, hor_dwd_dl_lgwa_wdwd[:, 0], color=LGWA_COLOR, linestyle='--', label='LGWA - DWD (10 yr)')
plt.loglog(ff_lgwa, hor_dwd_dl_lgwa_wdwd[:, 1], color=LGWA_COLOR, linestyle='--')
plt.loglog(ff_lgwa, hor_dwd_dl_lgwa_wdwd_3yr[:, 0], color=LGWA_COLOR, linestyle=':', label='LGWA - DWD (3 yr)')
plt.loglog(ff_lgwa, hor_dwd_dl_lgwa_wdwd_3yr[:, 1], color=LGWA_COLOR, linestyle=':')
plt.loglog(ff_lisa, hor_dwd_dl_lisa_wdns[:, 0], color=LISA_COLOR, label='LISA - WDNS (3 yr)')
plt.fill_between(ff_lisa, hor_dwd_dl_lisa_wdns[:, 0], color=LISA_COLOR, alpha=0.2)
plt.loglog(ff_lisa, hor_dwd_dl_lisa_wdwd[:, 0], color=LISA_COLOR, linestyle='--', label='LISA - DWD (3 yr)')
plt.fill_between(ff_lisa, hor_dwd_dl_lisa_wdwd[:, 0], color=LISA_COLOR, alpha=0.2)
plt.xlabel('Frequency, max-cutoff [Hz]')
plt.ylabel('Horizon distance [Mpc]')
plt.legend()
plt.grid()
plt.text(0.01, 1.15, 'SNR=4', rotation=14, color=LISA_COLOR, fontsize=10)
plt.text(0.15, 130, 'SNR=4', rotation=30, color=LGWA_COLOR, fontsize=10)
plt.text(0.2, 40, 'SNR=9', rotation=30, color=LGWA_COLOR, fontsize=10)
plt.xlim([6e-3, 0.3])
plt.ylim([0.2, 270])
plt.savefig('Horizon_DWD_test.png', dpi=600)
