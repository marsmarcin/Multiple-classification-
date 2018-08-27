import numpy as np
import xgboost as xgb
data = np.loadtxt('K:/Pattern/test1_exp1/xgb_1.data', delimiter=',')
sz = data.shape
train = data[:int(sz[0] * 0.7), :]
test = data[int(sz[0] * 0.7):, :]
train_X = train[:, 0:279]
train_Y = train[:, 279]
# print(train_X)
# print(train_Y)
test_X = test[:, 0:279]
test_Y = test[:, 279]
xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)
param = {}
param['objective'] = 'multi:softmax'
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 6
watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 1000
bst = xgb.train(param, xg_train, num_round, watchlist)
pred = bst.predict(xg_test)
param['objective'] = 'multi:softprob'
bst = xgb.train(param, xg_train, num_round, watchlist)
yprob = bst.predict(xg_test).reshape(test_Y.shape[0], 6)
ylabel = np.argmax(yprob, axis=1)
test_error = sum(int(ylabel[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y))
accuracy = 1 - test_error
print(test_error)
print(accuracy)
