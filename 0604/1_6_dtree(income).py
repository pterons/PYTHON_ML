# -*- coding: utf-8 -*-
"""1-6.DTree(income).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rvlFIk9u4l8FO9csWKKt3QB5D0tU8TFP
"""

# Decision Tree로 income 데이터를 학습한다 (pre-pruning).
# 데이터 세트 : http://archive.ics.uci.edu/ml/datasets/Adult
# Abstract: Predict whether income exceeds $50K/yr based on 
#           census data. Also known as "Census Income" dataset
# ------------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Commented out IPython magic to ensure Python compatibility.
# %cd '/content/drive/MyDrive/Colab Notebooks'

# 데이터 파일을 읽어온다.
income = pd.read_csv("data/income.csv", index_col=False)

income.head()

cat_income = income.copy()

# categorical feature들을 숫자로 바꾼다.
cat_features = ["workclass", "education_num", "marital_status", "occupation", 
                "relationship", "race", "sex","native_country", "income"]

le = {}
for c in cat_features:
    le[c] = LabelEncoder()
    cat_income[c] = le[c].fit_transform(cat_income[c])

cat_income.head()

le['workclass'].inverse_transform([7])
le['workclass'].transform([' State-gov'])

# Train 데이터 세트와 Test 데이터 세트를 구성한다
data = np.array(cat_income)
feature_data = data[:, :-1]
target_data = data[:, -1]
trainX, testX, trainY, testY = train_test_split(feature_data, target_data, test_size = 0.2)

testGini = []
depth = []
for k in range(1, 20):
    # Gini 계수를 사용하여 학습 데이터를 학습한다.
    dt = DecisionTreeClassifier(criterion='gini', max_depth=k)
    dt.fit(trainX, trainY)
    
    # 정확도를 측정한다.
    testGini.append(dt.score(testX, testY))
   
    depth.append(k)
    print('depth = %d done.' % k)

# Gini와 Entropy, 그리고 tree depth에 따른 정확도를 비교한다.
plt.figure(figsize=(8, 5))
plt.plot(testGini, label="Gini/Test")
plt.legend()
plt.xlabel("Tree depth")
plt.ylabel("Accuracy")
plt.show()

# 정확도가 가장 큰 최적 depth를 찾는다.
nDepth = depth[np.argmax(testGini)]

# opt_alpha를 적용한 tree를 사용한다.
dt = DecisionTreeClassifier(max_depth = nDepth)
dt.fit(trainX, trainY)
print('시험 데이터의 정확도 = %.4f' % dt.score(testX, testY))
print('최적 트리의 depth = %d' % nDepth)

# feature별 중요도를 파악한다.
feat_impo = dt.feature_importances_
feat_name = income.columns

# 중요도가 높은 feature 5개를 확인한다.
idx = np.argsort(feat_impo)[::-1][:5]
np.array(feat_name)[idx]

