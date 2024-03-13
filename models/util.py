import pyPPG
import numpy as np
from pyPPG import PPG, Fiducials, Biomarkers
from pyPPG.datahandling import load_data, plot_fiducials, save_data, load_fiducials
import pyPPG.preproc as PP
import pyPPG.fiducials as FP
import pyPPG.biomarkers as BM
import pyPPG.ppg_sqi as SQI

from models.custom_dataloader import UCIBPDatasetRaw

def sqi(ppg_np):

    np.savetxt("temp.csv", ppg_np, delimiter=",")       
    signal = load_data("temp.csv", 125)


    fL=0.5000001
    fH=250
    order=4
    sm_wins={'ppg':50,'vpg':10,'apg':10,'jpg':10}

    prep = PP.Preprocess(fL=fL, fH=fH, order=order, sm_wins=sm_wins)
    signal.ppg, signal.vpg, signal.apg, signal.jpg = prep.get_signals(s=signal)

    s = PPG(signal, 125)
    fpex = FP.FpCollection(s=s)
    fiducials = fpex.get_fiducials(s)
    fp = Fiducials(fp=fiducials)

    return round(np.mean(SQI.get_ppgSQI(ppg=s.ppg, fs=s.fs, annotation=fp.sp))*100, 2)


# Need to unit test the function
def convert_UCIBP_mat2npy(dataset_path, save_path):
    UCIBP_dataset = UCIBPDatasetRaw(dataset_path)

    abp_samples = np.zeros((135819, 1024))
    ppg_samples = np.zeros((135819, 1024))

    sample_count = 0

    # column 1 SBP and column 2 DBP
    pressures = np.zeros((135819, 3))

    for i in range(len(UCIBP_dataset)):
        data_sample_ppg, data_sample_abp, ori_data_sample_ppg, ori_data_sample_abp = UCIBP_dataset[i]

        if len(ori_data_sample_ppg) >=8*60*125:
            num_samples = len(ori_data_sample_ppg)//1024

            for j in range(num_samples):
                abp_sample = ori_data_sample_abp[j*1024:(j+1)*1024]
                ppg_sample = ori_data_sample_ppg[j*1024:(j+1)*1024]
                if np.max(abp_sample) <= 200 and np.min(abp_sample) >= 50:
                    
                    pressures[sample_count, 0] = np.max(abp_sample)
                    pressures[sample_count, 1] = np.min(abp_sample)
                    abp_samples[sample_count, :] = abp_sample
                    ppg_samples[sample_count, :] = ppg_sample
                    sample_count += 1

    pressures = pressures.astype(np.float32)
    abp_samples = abp_samples.astype(np.float32)
    ppg_samples = ppg_samples.astype(np.float32)

    # Calculating MAP from SBP and DBP
    # MAP = (SBP + 2*DBP)/3
    pressures[:, 2] = (pressures[:, 0] + 2*pressures[:, 1])/2

    with open(f"{save_path}/abp_samples.npy", 'wb') as f:
        np.save(f, abp_samples)

    with open(f"{save_path}/ppg_samples.npy", 'wb') as f:
        np.save(f, ppg_samples)

    with open(f"{save_path}/pressures.npy", 'wb') as f:
        np.save(f, pressures)

    return 



