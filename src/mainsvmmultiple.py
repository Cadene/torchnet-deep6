# -*- coding: utf-8 -*-

import os
import time
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

import lutorpy as lua

l2norm = False

# paths
path = '.'

pathdataset = path+'/data/processed/dsgqualif'
pathclasses = pathdataset+'/classes.t7'
pathclass2target = pathdataset+'/class2target.t7'

pathlog = path+'/features/dsgqualif/16_08_27_20:37:21'
pathconfig = pathlog+'/config.t7'

classes = torch.load(pathclasses)
class2target = torch.load(pathclass2target)
config = torch.load(pathconfig)

nclasses = len(classes)
nfeats = config['nfeats'] or 2048
print('#features', nfeats)

listdftrain = []
listdfval = []

listpathfeats = []
listpathfeats.append(path+'/features/dsgqualif/16_08_27_21:30:47')
listpathfeats.append(path+'/features/dsgqualif/16_08_27_20:36:08')
listpathfeats.append(path+'/features/dsgqualif/16_08_27_20:37:58')
listpathfeats.append(path+'/features/dsgqualif/16_08_27_20:37:21')
for i, pathfeats in enumerate(listpathfeats):
    pathtrain = pathfeats+'/trainextract.csv'
    pathval   = pathfeats+'/valextract.csv'
    listdftrain.append(pd.read_csv(pathtrain, sep=';'))#, nrows=1000)
    listdfval.append(pd.read_csv(pathval, sep=';'))#, nrows=1000)
    print(listdftrain[i].shape)
    print(listdfval[i].shape)

dftrain = pd.concat(listdftrain, ignore_index=True)
dfval = pd.concat(listdfval, ignore_index=True)

print(dftrain.shape)
print(dfval.shape)

def extractX(df, nfeats, l2norm):
  dims=[]
  for i in xrange(nfeats):
    dims.append('feat'+str(i+1))
  X = df[dims].values
  if l2norm:
    norm = np.linalg.norm(X,ord=None,axis=1)
    X /= norm.reshape((X.shape[0],1))
    X /= np.sqrt(X)
  return X

def extracty(df):
  y = df['gttarget'].values
  return y

Xtrain = extractX(dftrain, nfeats, l2norm)
ytrain = extracty(dftrain)

Xval = extractX(dfval, nfeats, l2norm)
yval = extracty(dfval)

#rangeC = [pow(10,i) for i in range(-5,5)]
#rangeC = [150, 250, 380, 520]
#rangeC = [610, 700, 800, 900]
#rangeC = [2, 4]
#rangeC = [6, 8]
#rangeC = [20, 40]
rangeC = [60, 80]
print(rangeC)
for C in rangeC:
    clf = LinearSVC(dual=True, C=C, loss='hinge', tol=1e-5, fit_intercept=False, max_iter=2000)
    clf.fit(Xtrain, ytrain)
    yvalpred = clf.predict(Xval)
    print('C='+str(C), str(accuracy_score(yval, yvalpred)*100)+'% accTop1')
