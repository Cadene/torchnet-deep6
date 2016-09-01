# -*- coding: utf-8 -*-

import os
import time
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

import lutorpy as lua

# config
l2norm = False
nfeats = 2048

# paths
path = '.'

pathdataset = path+'/data/processed/m2caiworkflow'
pathclasses = pathdataset+'/classes.t7'
pathclass2target = pathdataset+'/class2target.t7'

pathfeats  = path+'/features/m2caiworkflow/16_08_31_17:29:04'
pathtrain  = pathfeats+'/trainextract.csv'
pathval    = pathfeats+'/valextract.csv'
pathconfig = pathfeats+'/config.t7'
#pathtest  = pathfeats+'/testextract.csv'

classes = torch.load(pathclasses)
class2target = torch.load(pathclass2target)
config = torch.load(pathconfig)

nclasses = len(classes)
nfeats = config['nfeats'] or 2048
print('#features', nfeats)

dftrain = pd.read_csv(pathtrain, sep=';')#, nrows=1000)
dfval   = pd.read_csv(pathval, sep=';')#, nrows=1000)
#dftest  = pd.read_csv(pathtest)#, nrows=1000)

print(dftrain.shape)
print(dfval.shape)

def extractX(df, nfeats, norm, l2norm):
  dims=[]
  for i in xrange(nfeats):
    dims.append('feat'+str(i+1))
  X = df[dims].values
  if l2norm:
    norm = np.linalg.norm(X,ord=None,axis=1)
    X /= norm.reshape((X.shape[0],1))
    X /= np.sqrt(X)
  if norm:
      X = (X - X.mean(axis=0)) / (X.std(axis=0))
  return X

def extracty(df):
  y = df['gttarget'].values
  return y

Xtrain = extractX(dftrain, nfeats, True, l2norm)
ytrain = extracty(dftrain)

Xval = extractX(dfval, nfeats, True, l2norm)
yval = extracty(dfval)

print(Xtrain.shape)
print(ytrain.shape)

print(Xval.shape)
print(yval.shape)

rangeC = [pow(10,i) for i in range(0, 5)]
#rangeC = [pow(10,i) for i in range(-3, 0)]
#rangeC = [150, 250, 380, 520]
# rangeC = [610, 700, 800, 900]
# print(pathfeats)
# print('SVM primal crammer_singer')
# print(rangeC)
# for C in rangeC:
#     clf = LinearSVC(dual=False, C=C, multi_class='crammer_singer', loss='hinge', tol=1e-4, fit_intercept=True, max_iter=1000)
#     # clf = SVC(C=C, )
#     clf.fit(Xtrain, ytrain)
#     yvalpred = clf.predict(Xval)
#     print('C='+str(C), str(accuracy_score(yval, yvalpred)*100)+'% accTop1')
