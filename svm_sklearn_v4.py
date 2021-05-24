#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Copyright (C) 2021 Ricardo Pires
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

import getopt
import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import GridSearchCV 
import sys
import time
optlist, args = getopt.gnu_getopt( sys.argv[ 1: ], 'r:v:d:t:' )
for( opcao, argumento ) in optlist:
    if opcao == '-r':
        nomeArqRotulos = argumento
    elif opcao == '-v':
        nomeArqVetores = argumento
    elif opcao == '-d':
        nomeArqDados = argumento
    elif opcao == '-t':
        nomeArqTemplate = argumento
vetores = np.loadtxt( nomeArqVetores )
rotulos = np.loadtxt( nomeArqRotulos )
dados = np.loadtxt( nomeArqDados )
template = np.loadtxt( nomeArqTemplate )
start = time.time()
clf = svm.SVC( C=10, gamma = 1 )
# clf = svm.SVC()

# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html#sklearn.preprocessing.MaxAbsScaler
max_abs_scaler = preprocessing.MaxAbsScaler()
# Normalização dos vetores de treinamento
vetores = max_abs_scaler.fit_transform(vetores)

clf.fit( vetores, rotulos )

# Normalização dos vetores de teste
dados = max_abs_scaler.fit_transform(dados)

r = clf.predict( dados )
a = metrics.accuracy_score( template, r, normalize=True ) * 100
print( 'prediction\n', r )
print( 'accuracy: %02f%%' % a )
end = time.time()
print(f"Time: {end - start}s")
