# Codes used in paper "Intelligent processing and vibration analysis system on rotating machines for Advanced Predictive Maintenance in industry 4.0"

## Installation
1. Clone this repository with `--recursive`
2. Install Python3 and pip.
3. `pip3 install scipy scikit-learn`
4. `sudo python3 lib/PyEMD/setup.py install`

## How to use
1. `./gen_svm_arrs_emd_v1.py` to generate the files.
2. `./svm_sklearn_v4.py -v vecs_5.txt -r classes_tr5.txt -d data_5.txt -t classes_exe5.txt` to train and test. 


