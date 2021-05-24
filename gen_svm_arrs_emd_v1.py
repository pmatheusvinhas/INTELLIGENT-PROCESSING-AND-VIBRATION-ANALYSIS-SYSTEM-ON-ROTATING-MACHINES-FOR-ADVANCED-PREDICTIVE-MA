#!/usr/bin/env python3
'''
Copyright (C) 2021 Paulo Matheus Vinhas (1000bbits)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

## https://pyemd.readthedocs.io/en/latest/intro.html
## https://github.com/laszukdawid/PyEMD

from PyEMD import EMD
from scipy import stats
from scipy import signal
import numpy  as np
import pylab as plt
import matplotlib.pyplot as pyplot
import time
import math

start = time.time()

# Sampling Frequency
FS = 12000.0 # 12KHz
# Samples number
SAMPLES = 2800
# Trainning windows length
TR_WINDOWS = 30
# Testing windows length
EXE_WINDOWS = 13
# Max number of elements (depends of dataset)
MAX_ELEMENS = 121991 
# Sub signals numbers as list (e.g. sensors)
SUB_SIGS_L = [3,3,1,2]
# Debugging features. Run the program in just 1 window printing the values ​​on the console to evaluate the results.
DEBUG_FEATURES = 0 # Not implemented

TRAINING_SET = SAMPLES
if TRAINING_SET > MAX_ELEMENS:
    print("You have exceeded the max. number of elements in dataset")
    exit(1)

# Trainning windows
s0_0_tr = [np.genfromtxt("data/bearingdatacenter/inner race defect X169-0.014-X169_DE_time.csv", skip_header=arr*TRAINING_SET ,max_rows=TRAINING_SET) for arr in range(TR_WINDOWS)]
s0_1_tr = [np.genfromtxt("data/bearingdatacenter/inner race defect X169-0.014-X169_FE_time.csv", skip_header=arr*TRAINING_SET ,max_rows=TRAINING_SET) for arr in range(TR_WINDOWS)]
s0_2_tr = [np.genfromtxt("data/bearingdatacenter/inner race defect X169-0.014-X169_BA_time.csv", skip_header=arr*TRAINING_SET ,max_rows=TRAINING_SET) for arr in range(TR_WINDOWS)]
s1_0_tr = [np.genfromtxt("data/bearingdatacenter/outer race defect X130-0.07-X130_DE_time.csv", skip_header=arr*TRAINING_SET ,max_rows=TRAINING_SET) for arr in range(TR_WINDOWS)]
s1_1_tr = [np.genfromtxt("data/bearingdatacenter/outer race defect X130-0.07-X130_FE_time.csv", skip_header=arr*TRAINING_SET ,max_rows=TRAINING_SET) for arr in range(TR_WINDOWS)]
s1_2_tr = [np.genfromtxt("data/bearingdatacenter/outer race defect X130-0.07-X130_BA_time.csv", skip_header=arr*TRAINING_SET ,max_rows=TRAINING_SET) for arr in range(TR_WINDOWS)]
s2_tr =   [np.genfromtxt("data/bearingdatacenter/ball defectX3006-0.028-X049_DE_time.csv", skip_header=arr*TRAINING_SET ,max_rows=TRAINING_SET) for arr in range(TR_WINDOWS)]
s3_0_tr = [np.genfromtxt("data/bearingdatacenter/Normal - 0hp -1797rpm - X097_DE_time.csv", skip_header=arr*TRAINING_SET ,max_rows=TRAINING_SET) for arr in range(TR_WINDOWS)]
s3_1_tr = [np.genfromtxt("data/bearingdatacenter/Normal - 0hp -1797rpm - X097_FE_time.csv", skip_header=arr*TRAINING_SET ,max_rows=TRAINING_SET) for arr in range(TR_WINDOWS)]
s_tr = np.array([s0_0_tr,s0_1_tr,s0_2_tr,s1_0_tr,s1_1_tr,s1_2_tr,s2_tr,s3_0_tr,s3_1_tr])

# Testing windows 
offset = TRAINING_SET*TR_WINDOWS
s0_0_exe = [np.genfromtxt("data/bearingdatacenter/inner race defect X169-0.014-X169_DE_time.csv", skip_header=offset+(arr*TRAINING_SET) ,max_rows=TRAINING_SET) for arr in range(EXE_WINDOWS)]
s0_1_exe = [np.genfromtxt("data/bearingdatacenter/inner race defect X169-0.014-X169_FE_time.csv", skip_header=offset+(arr*TRAINING_SET) ,max_rows=TRAINING_SET) for arr in range(EXE_WINDOWS)]
s0_2_exe = [np.genfromtxt("data/bearingdatacenter/inner race defect X169-0.014-X169_BA_time.csv", skip_header=offset+(arr*TRAINING_SET) ,max_rows=TRAINING_SET) for arr in range(EXE_WINDOWS)]
s1_0_exe = [np.genfromtxt("data/bearingdatacenter/outer race defect X130-0.07-X130_DE_time.csv", skip_header=offset+(arr*TRAINING_SET) ,max_rows=TRAINING_SET) for arr in range(EXE_WINDOWS)]
s1_1_exe = [np.genfromtxt("data/bearingdatacenter/outer race defect X130-0.07-X130_FE_time.csv", skip_header=offset+(arr*TRAINING_SET) ,max_rows=TRAINING_SET) for arr in range(EXE_WINDOWS)]
s1_2_exe = [np.genfromtxt("data/bearingdatacenter/outer race defect X130-0.07-X130_BA_time.csv", skip_header=offset+(arr*TRAINING_SET) ,max_rows=TRAINING_SET) for arr in range(EXE_WINDOWS)]
s2_exe =   [np.genfromtxt("data/bearingdatacenter/ball defectX3006-0.028-X049_DE_time.csv", skip_header=offset+(arr*TRAINING_SET) ,max_rows=TRAINING_SET) for arr in range(EXE_WINDOWS)]
s3_0_exe = [np.genfromtxt("data/bearingdatacenter/Normal - 0hp -1797rpm - X097_DE_time.csv", skip_header=offset+(arr*TRAINING_SET) ,max_rows=TRAINING_SET) for arr in range(EXE_WINDOWS)]
s3_1_exe = [np.genfromtxt("data/bearingdatacenter/Normal - 0hp -1797rpm - X097_FE_time.csv", skip_header=offset+(arr*TRAINING_SET) ,max_rows=TRAINING_SET) for arr in range(EXE_WINDOWS)]
s_exe = np.array([s0_0_exe,s0_1_exe,s0_2_exe,s1_0_exe,s1_1_exe,s1_2_exe,s2_exe,s3_0_exe,s3_1_exe])

# params: 
#   imf: enumarated imf
#   features: destination list of characteristics
def do_arrs(imf, features):
    ## Time features
    # mean 
    features.append(str(stats.tmean(imf)))
    #print("mean: " + str(stats.tmean(imf)))

    # variance
    features.append(str(stats.tvar(imf)))
    #print("var: "+ str(stats.tvar(imf)))

    # kurtosis
    features.append(str(stats.kurtosis(imf)))
    #print("kurt: " + str(stats.kurtosis(imf)))

    # median absolute deviation
    features.append(str(stats.median_abs_deviation(imf)))
    #print("MAD: "+ str(stats.median_abs_deviation(imf)))

    # range
    maax = stats.tmax(imf)
    miin = stats.tmin(imf)   
    features.append(str(maax-miin))
    #print("range: "+ str(maax-miin) )

    # clearance factor
    clf = 0.0
    for i in range(1, 2800):
        clf = math.sqrt(abs(imf[i]))
    clf = maax/(((1/2800)*clf)**2)
    features.append(str(clf))

    # standard deviation
    features.append(str(stats.tstd(imf)))
    #print("standard devi: " + str(stats.tstd(imf)))

    ## Frequency features
    # mean frequency
    a = b = dsp = p = t = soma_pots = spec_entropy = 0.0
    mags, freqs, line = pyplot.magnitude_spectrum(imf, Fs=FS)
    if len(mags) == len(freqs):
        for u in range(1, len(mags)):
            a += mags[u]*freqs[u]
            b += mags[u]
            p = abs(freqs[u])**2
            t += p*freqs[u]
            spec_entropy += p*np.log10(p)
            soma_pots += p
        fmean = a/b
    features.append(str(fmean))
    #print("mean freq: " +  str(fmean))

    # power spectral density
    dsp = soma_pots / len(freqs)
    features.append(str(dsp))
    #print("power spectral density: ", dsp)

    # mean power frequency
    mnf = t/soma_pots
    features.append(str(mnf))
    #print("mean power freq: ", mnf)  

    # spectral entropy
    features.append(str(-spec_entropy))

# Trainning
vec = open("vecs_5.txt", "w")
clss_tr = open("classes_tr5.txt", "w")
for i in range(len(s_tr)):
    print("i = %d" % i)
    for y in range(TR_WINDOWS):
        print("y = %d" % y)
        t = np.linspace(0, TRAINING_SET, TRAINING_SET)
        IMF = EMD().emd(s_tr[i][y],t)
        N = IMF.shape[0]+1
        tr_features = []
        for n, imf in enumerate(IMF):
            if n==0 or n==3 or n==6:
                do_arrs(imf, tr_features)
        for idx, feat in enumerate(tr_features):
            vec.write(str(feat)+" ")
        vec.write("\n")
vec.close() 
ii = 0
for j in range(len(SUB_SIGS_L)):
    if ii == len(SUB_SIGS_L):
        ii=0
    for y in range(TR_WINDOWS*SUB_SIGS_L[ii]):
        clss_tr.write(str(j)+"\n")
    ii+=1
clss_tr.close()

# Testing
data = open("data_5.txt", "w")
clss_exe = open("classes_exe5.txt", "w")
for i in range(len(s_exe)):
    for y in range(EXE_WINDOWS):
        t = np.linspace(0, TRAINING_SET, TRAINING_SET)
        IMF = EMD().emd(s_tr[i][y],t)
        N = IMF.shape[0]+1
        exe_features = []
        for n, imf in enumerate(IMF):
            if n==0 or n==3 or n==6:
                do_arrs(imf, exe_features)
        for idx, feat in enumerate(exe_features):
            data.write(str(feat)+" ")
        data.write("\n")
data.close()
ii=0
for j in range(len(SUB_SIGS_L)):
    if ii == len(SUB_SIGS_L):
        ii=0
    for y in range(EXE_WINDOWS*SUB_SIGS_L[ii]):
        clss_exe.write(str(j)+"\n")
    ii+=1
clss_exe.close()

end = time.time()
print(f"Time: {end - start}s")