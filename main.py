import sys
import mne
import numpy as np
import matplotlib.pyplot as plt
import math
from math import log
from mne.time_frequency import psd_multitaper
import os


def fun(rawn, s, first_num, second_num):
    arr = np.array([], dtype='i').reshape((0, 3))
    for j in range(0, len(rawn.annotations), 1):
        if (rawn.annotations.description[j] == f' {s}{first_num}') or (
                rawn.annotations.description[j] == f'{s}{first_num}') or (
                rawn.annotations.description[j] == f'{s} {first_num}') or (
                rawn.annotations.description[j] == f'{s}  {first_num}'):
            tmin = int(round(rawn.annotations.onset[j] * rawn.info['sfreq']))

            for jj in range(j, len(rawn.annotations), 1):

                if (rawn.annotations.description[jj] == f' {s}{second_num}') or (
                        rawn.annotations.description[jj] == f'{s}{second_num}') or (
                        rawn.annotations.description[jj] == f'{s} {second_num}') or (
                        rawn.annotations.description[jj] == f'{s}  {second_num}'):
                    tmax = int(round(rawn.annotations.onset[jj] * rawn.info['sfreq']))
                    # print(rawn.annotations.onset[jj])
                    # print(rawn.annotations.onset)
                    # print(rawn.annotations)
                    # print(rawn.info['sfreq'])
                    # print(rawn.info)
                    # print(rawn.n_times)
                    # print("-------------------------------------")
                    break
            for k in np.arange(tmin, tmax, 8 * rawn.info['sfreq']):
                arr = np.append(arr, [[int(k), 0, 1]], axis=0)

    return arr


def _divide_side(lobe, x):
    """Make a separation between left and right lobe evenly."""
    lobe = np.asarray(lobe)
    median = np.median(x[lobe])

    left = lobe[np.where(x[lobe] < median)[0]]
    right = lobe[np.where(x[lobe] > median)[0]]
    medians = np.where(x[lobe] == median)[0]

    left = np.sort(np.concatenate([left, lobe[medians[1::2]]]))
    right = np.sort(np.concatenate([right, lobe[medians[::2]]]))
    return list(left), list(right)


args = sys.argv[1:]
fnam = args[0]
print(fnam)
cur_path = os.getcwd()
print('getcwd', cur_path)

rawn = mne.io.read_raw_eeglab(f'{cur_path}/milah/moved/{fnam}/{fnam}.set', preload=True)

# rawn.plot()

outname = 'rawn.annotations,rawn.info["sfreq"],len(rawn.annotations),set(rawn.annotations.duration),set(rawn.annotations.description),rawn.annotations.onset[0],rawn.annotations.description[0],rawn.info[custom_ref_applied],rawn.annotations.onset\n'
out = [rawn.annotations, rawn.info["sfreq"], len(rawn.annotations), set(rawn.annotations.duration),
       set(rawn.annotations.description), rawn.annotations.onset[0], rawn.annotations.description[0],
       rawn.info['custom_ref_applied'], rawn.annotations.onset]
with open(f'{cur_path}/tmpdir/mneread-{fnam}.csv', mode='wt', encoding='utf-8') as myfile:
    myfile.write(outname)
    myfile.write(str(out))

chlis = rawn.info['ch_names']

if ('ECG' in chlis):
    print("Drop ECG")
    rawn = rawn.drop_channels(['ECG'])
if ('EKG' in chlis):
    print("Drop EKG")
    rawn = rawn.drop_channels(['EKG'])
if ('B1+' in chlis):
    print("Drop B1+")
    rawn = rawn.drop_channels(['B1+'])
if ('Cz' not in chlis):
    print("Adding Cz")
    rawn = mne.add_reference_channels(rawn, ref_channels=['Cz'])
rawn = rawn.set_eeg_reference(ref_channels='average')
print('av reference done--------------------------------------------------------------------------------------')

# rawn.plot()

arrc = np.array([], dtype='i').reshape((0, 3))
arro = np.array([], dtype='i').reshape((0, 3))

arrc = fun(rawn, "S", "1", "2")
arro = fun(rawn, "S", "2", "1")

print('8secs done--------------------------------------------------------------------------------------')

if (len(arrc) > 0) and (len(arro) > 0):
    layout_from_raw = mne.channels.make_eeg_layout(rawn.info)
    picks = mne.pick_types(rawn.info, eeg=True, exclude=[])
    chs_in_lobe = rawn.info['nchan'] // 4
    pos = np.array([ch['loc'][:3] for ch in rawn.info['chs']])
    x, y, z = pos.T

    frontal = picks[np.argsort(y)[-chs_in_lobe:]]
    picks = np.setdiff1d(picks, frontal)
    occipital = picks[np.argsort(y[picks])[:chs_in_lobe]]
    picks = np.setdiff1d(picks, occipital)
    temporal = picks[np.argsort(z[picks])[:chs_in_lobe]]
    picks = np.setdiff1d(picks, temporal)  # parietal

    lt, rt = _divide_side(temporal, x)
    lf, rf = _divide_side(frontal, x)
    lo, ro = _divide_side(occipital, x)
    lp, rp = _divide_side(picks, x)  # Parietal lobe from the remaining picks.
    print('divide done--------------------------------------------------------------------------------------')

    # %%

    roi_dict = dict(lt=lt, rt=rt, lf=lf, rf=rf, lo=lo, ro=ro, lp=lp, rp=rp)
    rawdic = mne.channels.combine_channels(rawn, roi_dict, method='mean')
    print('combine done--------------------------------------------------------------------------------------')

    # %%

    baseline = (0, 0)
    epochsEC = mne.Epochs(rawdic, arrc, event_id=1, tmin=0, tmax=8, baseline=baseline)
    epochsEO = mne.Epochs(rawdic, arro, event_id=1, tmin=0, tmax=8, baseline=baseline)
    print('epoch done--------------------------------------------------------------------------------------')

    scaling = 1000000
    waveband = [1, 4, 8, 12, 16, 20, 25, 35]
    EOpsds_mean0_av1_tot = 0
    ECpsds_mean0_av1_tot = 0
    EOpsds_mean0_av1_list = []
    ECpsds_mean0_av1_list = []
    for j in range(len(waveband) - 1):
        print(j)
        fmin = waveband[j]
        fmax = waveband[j + 1]
        ECpsds0, freqs0 = psd_multitaper(epochsEC, fmin=fmin, fmax=fmax, n_jobs=1)
        ECpsds0 *= scaling * scaling / 1000
        ECpsds_mean0 = ECpsds0.mean(0)

        EOpsds0, freqs0 = psd_multitaper(epochsEO, fmin=fmin, fmax=fmax, n_jobs=1)
        EOpsds0 *= scaling * scaling / 1000
        EOpsds_mean0 = EOpsds0.mean(0)

        EOpsds_mean0_av1 = EOpsds_mean0.sum(1)
        ECpsds_mean0_av1 = ECpsds_mean0.sum(1)

        EOpsds_mean0_av1_tot = EOpsds_mean0_av1_tot + EOpsds_mean0_av1
        ECpsds_mean0_av1_tot = ECpsds_mean0_av1_tot + ECpsds_mean0_av1

        EOpsds_mean0_av1_list.append(EOpsds_mean0_av1)
        ECpsds_mean0_av1_list.append(ECpsds_mean0_av1)
    EOpsds_mean0_av1_list_norm = EOpsds_mean0_av1_list / EOpsds_mean0_av1_tot
    ECpsds_mean0_av1_list_norm = ECpsds_mean0_av1_list / ECpsds_mean0_av1_tot
    print('waverange  done--------------------------------------------------------------------------------------')

    aa = [ECpsds_mean0_av1_tot, EOpsds_mean0_av1_tot]

    aout = np.asarray(aa)
    aout.tofile(f'{cur_path}/tmpdir/mneECOpsds_av1_tot-{fnam}.csv', sep=',', format='%10.5f')

    aa = [ECpsds_mean0_av1_list]

    aout = np.asarray(aa)
    aout.tofile(f'{cur_path}/tmpdir/mneECpsds_av1_list-{fnam}.csv', sep=',', format='%10.5f')

    aa = [EOpsds_mean0_av1_list]

    aout = np.asarray(aa)
    aout.tofile(f'{cur_path}/tmpdir/mneEOpsds_av1_list-{fnam}.csv', sep=',', format='%10.5f')

    aa = [ECpsds_mean0_av1_list_norm,
          EOpsds_mean0_av1_list_norm]

    aout = np.asarray(aa)
    aout.tofile(f'{cur_path}/tmpdir/mneECOpsds_av1_list_norm-{fnam}.csv', sep=',', format='%10.5f')
    print("save waverange done--------------------------------------------------------------------------------------")

    fmin = 7
    fmax = 13
    ECpsds0, freqs0 = psd_multitaper(epochsEC, fmin=fmin, fmax=fmax, n_jobs=1)
    EOpsds0, freqs0 = psd_multitaper(epochsEO, fmin=fmin, fmax=fmax, n_jobs=1)
    ECpsds0 *= scaling * scaling / 1000
    ECpsds_mean0av = ECpsds0.mean(0).mean(0)
    ECpsds_std0av = ECpsds0.mean(0).std(0)
    EOpsds0 *= scaling * scaling / 1000
    print('E1--------------')

    EOpsds_mean0av = EOpsds0.mean(0).mean(0)
    EOpsds_std0av = EOpsds0.mean(0).std(0)
    print('E2--------------')

    fn = np.array(freqs0)
    ecn = np.array(ECpsds_mean0av)
    eon = np.array(EOpsds_mean0av)
    print('E3--------------')

    fnalphaindex = np.where((fn >= 7) & (fn <= 13))
    fna = fn[fnalphaindex]
    ecna = ecn[fnalphaindex]
    eona = eon[fnalphaindex]
    print('E4--------------')

    imax = np.where(ecna == ecna.max())
    IAPF = fna[imax][0]
    IAPFpower = ecna[imax]
    dIAPFpower = (ecna[imax] - eona[imax])

    DIFF = abs(eona - ecna)

    ecnabf = DIFF[np.arange(imax[0], ecna.size, 1)]
    fnabf = fna[np.arange(imax[0], ecna.size, 1)]

    ecnatf = DIFF[np.arange(0, imax[0], 1)]
    fnatf = fna[np.arange(0, imax[0], 1)]

    tmp = np.where(ecnabf > 0)[0]
    if len(tmp) > 0:
        BF = fnabf[tmp[0]]
    else:
        BF = 13

    tmp = np.where(ecnatf > 0)[0]
    if len(tmp) > 0:
        TF = fnatf[tmp[len(tmp) - 1]]
    else:
        TF = 7

    IABW = BF - TF
    IABWA1 = IAPF - TF
    IABWA2 = BF - IAPF

    A1powerindex = np.where((fna >= TF) & (fna <= IAPF))
    A1ECpower = np.mean(ecna[A1powerindex])
    A1powerindex = np.where((fna <= BF) & (fna >= IAPF))
    A2ECpower = np.mean(ecna[A1powerindex])
    A1powerindex = np.where((fna >= TF) & (fna <= IAPF))
    A1EOpower = np.mean(eona[A1powerindex])
    A1powerindex = np.where((fna <= BF) & (fna >= IAPF))
    A2EOpower = np.mean(eona[A1powerindex])

    A1SupSign = math.copysign(1, (A1ECpower - A1EOpower) / A1ECpower)
    A2SupSign = math.copysign(1, (A2ECpower - A2EOpower) / A2ECpower)
    IAPFSupSign = math.copysign(1, (dIAPFpower / IAPFpower))

    A1Sup = log(abs((A1ECpower - A1EOpower) / A1ECpower) * 100).real
    A2Sup = log(abs((A2ECpower - A2EOpower) / A2ECpower) * 100).real
    IAPFSup = log(abs(dIAPFpower / IAPFpower) * 100)

    IAF = np.sum(fna * ecna) / np.sum(ecna)

    aa = [IAF, IAPF, IAPFpower, IAPFSup, TF, BF, IABW, IABWA1, IABWA2, A1ECpower, A2ECpower, A1Sup, A2Sup,
          A1SupSign,
          A2SupSign, IAPFSupSign]

    aout = np.asarray(aa, dtype=object)
    aout.tofile(f'{cur_path}/tmpdir/mneindexes-{fnam}.csv', sep=',', format='%10.5f')
    print("save mneindexes done--------------------------------------------------------------------------------------")

print("end--------------------------------------------------------------------------------------")

f, ax = plt.subplots()
ax.plot(freqs0, ECpsds_mean0av, color='k')
ax.fill_between(freqs0, ECpsds_mean0av - ECpsds_std0av, ECpsds_mean0av + ECpsds_std0av, color='k', alpha=.5)
ax.set(title='EC  Multitaper PSD (gradiometers)', xlabel='Frequency (Hz)', ylabel='Power Spectral Density (dB)')
plt.savefig(f'EC  Multitaper PSD (gradiometers).png')
plt.show()

f, ax = plt.subplots()
ax.plot(freqs0, EOpsds_mean0av, color='k')
ax.fill_between(freqs0, EOpsds_mean0av - EOpsds_std0av, EOpsds_mean0av + EOpsds_std0av, color='k', alpha=.5)
ax.set(title='EO  Multitaper PSD (gradiometers)', xlabel='Frequency (Hz)', ylabel='Power Spectral Density (dB)')
plt.savefig(f'EO  Multitaper PSD (gradiometers).png')
plt.show()

print(TF)
print(IAPF)
A1powerindex = np.where((fna >= TF) & (fna <= IAPF))
print(A1powerindex)
print(fna)
