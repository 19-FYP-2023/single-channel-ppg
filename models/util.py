import pyPPG
import numpy as np
from pyPPG import PPG, Fiducials, Biomarkers
from pyPPG.datahandling import load_data, plot_fiducials, save_data, load_fiducials
import pyPPG.preproc as PP
import pyPPG.fiducials as FP
import pyPPG.biomarkers as BM
import pyPPG.ppg_sqi as SQI

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