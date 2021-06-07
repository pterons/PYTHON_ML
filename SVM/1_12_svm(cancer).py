# -*- coding: utf-8 -*-
"""1-12.SVM(cancer).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1luhUPDO_-7y3G12A6bP6u5VwJLMx94RG
"""

# linear-SVM으로 iris 데이터를 학습한다.
# -------------------------------------
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

# breast cancer 데이터를 가져온다.
cancer = load_breast_cancer()

# 표준화
feature_data = StandardScaler().fit_transform(cancer.data)

# Train 데이터 세트와 Test 데이터 세트를 구성한다
trainX, testX, trainY, testY = train_test_split(feature_data, cancer.target, test_size = 0.2)

# 학습 및 평가
model = SVC(kernel='rbf', gamma=1.0, C=0.5)
model.fit(trainX, trainY)
print('정확도 =', np.round(model.score(testX, testY), 3))

# gamma와 C의 조합을 바꿔가면서 학습 데이터의 정확도가 최대인 조합을 찾는다
optAcc = -999
optG = 0
optC = 0
for gamma in np.arange(0.1, 5.0, 0.1):
    for c in np.arange(0.1, 5.0, 0.1):
        model = SVC(kernel='rbf', gamma=gamma, C=c)
        model.fit(trainX, trainY)
        acc = model.score(testX, testY)
        
        if acc > optAcc:
            optG = gamma
            optC = c
            optAcc = acc

print('Optimal gamma = %.2f' % optG)
print('optimal C = %.2f' % optC)
print('optimal Accuracy = %.2f' % optAcc)

# 최적 조건으로 학습한 결과를 확인한다.
model = SVC(kernel='rbf', gamma=optG, C=optC)
model.fit(trainX, trainY)

# Test 세트의 Feature에 대한 class를 추정하고, 정확도를 계산한다
print()
print("* 학습용 데이터로 측정한 정확도 = %.2f" % model.score(trainX, trainY))
print("* 시험용 데이터로 측정한 정확도 = %.2f" % model.score(testX, testY))

