from math import log
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
import numpy as np
import matplotlib.pyplot as plt
import mne
import math


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

fnamedirsetlist = []
fnamesetlist = []
fnamelist = []
a = []
b = []

fnam = 'D_Nov_042_fon1'
rawn = mne.io.read_raw_eeglab('milah\\moved\\D_Nov_042_fon1\\D_Nov_042_fon1.set',
                              preload=True)

outname = 'rawn.annotations,rawn.info["sfreq"],len(rawn.annotations),set(rawn.annotations.duration),set(rawn.annotations.description),rawn.annotations.onset[0],rawn.annotations.description[0],rawn.info[custom_ref_applied],rawn.annotations.onset\n'
out = [rawn.annotations, rawn.info["sfreq"], len(rawn.annotations), set(rawn.annotations.duration),
       set(rawn.annotations.description), rawn.annotations.onset[0], rawn.annotations.description[0],
       rawn.info['custom_ref_applied'], rawn.annotations.onset]
with open('./tmpdir/mneread-' + fnam + '.csv', mode='wt', encoding='utf-8') as myfile:
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


#%%

arrc = np.array([], dtype='i').reshape((0, 3))
arro = np.array([], dtype='i').reshape((0, 3))

#%%

for j in range(0, len(rawn.annotations), 1):
    if (rawn.annotations.description[j] == ' S1') or (rawn.annotations.description[j] == 'S1') or (
            rawn.annotations.description[j] == 'S 1') or (rawn.annotations.description[j] == 'S  1'):
        tminc = int(round(rawn.annotations.onset[j] * rawn.info['sfreq']))
        #tmaxc=int(rawn.n_times)
        tmaxc = int(rawn.n_times)

        for jj in range(j, len(rawn.annotations), 1):

            if (rawn.annotations.description[jj] == ' S2') or (rawn.annotations.description[jj] == 'S2') or (
                    rawn.annotations.description[jj] == 'S 2') or (rawn.annotations.description[jj] == 'S  2'):
                tmaxc = int(round(rawn.annotations.onset[jj] * rawn.info['sfreq']))
                break
        for k in np.arange(tminc, tmaxc, 8 * rawn.info['sfreq']):
            arrc = np.append(arrc, [[int(k), 0, 1]], axis=0)

#%%


for jj in range(0, len(rawn.annotations), 1):
    if (rawn.annotations.description[jj] == ' S2') or (rawn.annotations.description[jj] == 'S2') or (
            rawn.annotations.description[jj] == 'S 2') or (rawn.annotations.description[jj] == 'S  2'):
        tmino = int(round(rawn.annotations.onset[jj] * rawn.info['sfreq']))
        tmaxo = int(rawn.n_times)

        for jjj in range(jj, len(rawn.annotations), 1):
            if (rawn.annotations.description[jjj] == ' S1') or (rawn.annotations.description[jjj] == 'S1') or (
                    rawn.annotations.description[jjj] == 'S 1') or (rawn.annotations.description[jjj] == 'S  1'):
                tmaxo = int(round(rawn.annotations.onset[jjj] * rawn.info['sfreq']))
                break
        for k in np.arange(tmino, tmaxo, 8 * rawn.info['sfreq']):
            arro = np.append(arro, [[int(k), 0, 1]], axis=0)

print('8secs done--------------------------------------------------------------------------------------')


#%%




if (len(arrc) > 0) and (len(arro) > 0):
    layout_from_raw = mne.channels.make_eeg_layout(rawn.info)
    midline = np.where([name.endswith('z') for name in layout_from_raw.names])[0]

    picks = mne.pick_types(rawn.info, eeg=True, exclude=[])

    chs_in_lobe = rawn.info['nchan'] // 4

    pos = np.array([ch['loc'][:3] for ch in rawn.info['chs']])
    x, y, z = pos.T

    frontal = picks[np.argsort(y)[-chs_in_lobe:]]
    picks = np.setdiff1d(picks, frontal)

    occipital = picks[np.argsort(y[picks])[:chs_in_lobe]]
    picks = np.setdiff1d(picks, occipital)

    temporal = picks[np.argsort(z[picks])[:chs_in_lobe]]
    picks = np.setdiff1d(picks, temporal)  #parietal

    parietal = picks

    lt, rt = _divide_side(temporal, x)
    lf, rf = _divide_side(frontal, x)
    lo, ro = _divide_side(occipital, x)
    lp, rp = _divide_side(picks, x)  # Parietal lobe from the remaining picks.
    print('divide done--------------------------------------------------------------------------------------')



    roi_dict = dict(lt=lt, rt=rt, lf=lf, rf=rf, lo=lo, ro=ro, lp=lp, rp=rp)
    rawdic = mne.channels.combine_channels(rawn, roi_dict, method='mean')
    print('combine done--------------------------------------------------------------------------------------')

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
        ECpsds_std0 = ECpsds0.std(0)

        EOpsds0, freqs0 = psd_multitaper(epochsEO, fmin=fmin, fmax=fmax, n_jobs=1)
        EOpsds0 *= scaling * scaling / 1000
        EOpsds_mean0 = EOpsds0.mean(0)
        EOpsds_std0 = EOpsds0.std(0)

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
    aout.tofile('./tmpdir/mneECOpsds_av1_tot-' + fnam + '.csv', sep=',', format='%10.5f')

    aa = [ECpsds_mean0_av1_list]

    aout = np.asarray(aa)
    aout.tofile('./tmpdir/mneECpsds_av1_list-' + fnam + '.csv', sep=',', format='%10.5f')

    aa = [EOpsds_mean0_av1_list]

    aout = np.asarray(aa)
    aout.tofile('./tmpdir/mneEOpsds_av1_list-' + fnam + '.csv', sep=',', format='%10.5f')

    aa = [ECpsds_mean0_av1_list_norm,
          EOpsds_mean0_av1_list_norm]

    aout = np.asarray(aa)
    aout.tofile('./tmpdir/mneECOpsds_av1_list_norm-' + fnam + '.csv', sep=',', format='%10.5f')

    print('waverange saved--------------------------------------------------------------------------------------')

    fmin = 7
    fmax = 13
    ECpsds0, freqs0 = psd_multitaper(epochsEC, fmin=fmin, fmax=fmax, n_jobs=1)
    EOpsds0, freqs0 = psd_multitaper(epochsEO, fmin=fmin, fmax=fmax, n_jobs=1)
    ECpsds0 *= scaling * scaling / 1000
    ECpsds_mean0 = ECpsds0.mean(0)
    ECpsds_std0 = ECpsds0.std(0)
    ECpsds_mean0av = ECpsds0.mean(0).mean(0)
    ECpsds_std0av = ECpsds0.mean(0).std(0)
    EOpsds0 *= scaling * scaling / 1000
    EOpsds_mean0 = EOpsds0.mean(0)
    EOpsds_std0 = EOpsds0.std(0)
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

    aa = [IAF, IAPF, IAPFpower, IAPFSup, TF, BF, IABW, IABWA1, IABWA2, A1ECpower, A2ECpower, A1Sup, A2Sup, A1SupSign,
          A2SupSign, IAPFSupSign]

    aout = np.asarray(aa)
    aout.tofile('./tmpdir/mneindexes-' + fnam + '.csv', sep=',', format='%10.5f')

print("end")



#%%




#%%

f, ax = plt.subplots()
ax.plot(freqs0, ECpsds_mean0av, color='k')
ax.fill_between(freqs0, ECpsds_mean0av - ECpsds_std0av, ECpsds_mean0av + ECpsds_std0av, color='k', alpha=.5)
ax.set(title='EC  Multitaper PSD (gradiometers)', xlabel='Frequency (Hz)', ylabel='Power Spectral Density (dB)')
plt.show()

f, ax = plt.subplots()
ax.plot(freqs0, EOpsds_mean0av, color='k')
ax.fill_between(freqs0, EOpsds_mean0av - EOpsds_std0av, EOpsds_mean0av + EOpsds_std0av, color='k', alpha=.5)
ax.set(title='EO  Multitaper PSD (gradiometers)', xlabel='Frequency (Hz)', ylabel='Power Spectral Density (dB)')
plt.show()

#%%



#%%

print(TF)

#%%

print(IAPF)

#%%

A1powerindex = np.where((fna >= TF) & (fna <= IAPF))

#%%

print(A1powerindex)

#%%

print(fna)

#%%

