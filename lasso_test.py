from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
import numpy as np
import csv
from sklearn.linear_model import LassoCV
raw_data = csv.reader(open('H:/Pattern/test3/data_fix3.csv', 'r'))
a_data = []
a_label = []
for i in raw_data:
    a_data.append(i[1:])
    a_label.append(i[0])
a1_data = np.array(a_data)
a1_label =np.array(a_label)
a2_data = a1_data.astype(np.float32)
a2_label = a1_label.astype(np.float32)
# print(a2_data[:, 13])
# print(a2_label)
lassocv = LassoCV()
lassocv.fit(a2_data, a2_label)
# lassocv.fit(reg_data, reg_target)
LassoCV(alphas=None, copy_X=True, cv=None, eps=0.001, fit_intercept=True,
    max_iter=10000, n_alphas=100, n_jobs=1, normalize=False, positive=False,
    precompute='auto', random_state=None, selection='cyclic', tol=0.0001,
    verbose=False)
mask = lassocv.coef_ != 0
new_reg_data = a2_data[:, mask]
print(new_reg_data.shape)
with open('H:/Pattern/test3/test3_attributes.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(new_reg_data)