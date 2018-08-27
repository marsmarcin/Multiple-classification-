import csv
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
ops.reset_default_graph()
sess = tf.Session()
m_data = csv.reader(open('K:/Pattern/test0/data_two_train.csv', 'r'))
x_va = []
for i in m_data:
    x_va.append(i)
x_vb = np.array(x_va)
x_vc = []
n_c = []
for n_nu in x_vb:

    n_c = np.array([0.0 if y == '' else float(y) for y in n_nu])
    x_vc.append(n_c)
x_vd = np.array(x_vc)
x_1 = []
for d in x_vd:
    x_1.append(d[1:])
x_11 = np.array(x_1)
x_csv = np.array(x_11)
y_1 = np.array([[x[0]] for x in x_va])
y_t1 = np.array([1 if y == '0' else -1 for y in y_1])
y_t2 = np.array([1 if y == '1' else -1 for y in y_1])
y_t3 = np.array([1 if y == '2' else -1 for y in y_1])
y_csv = np.array([y_t1, y_t2, y_t3])
y_csv = y_csv.astype(np.float32)
batch_size = 100
x_data = tf.placeholder(shape=[None, 279], dtype=tf.float32)
y_target = tf.placeholder(shape=[3, None], dtype=tf.float32)
prediction_grid = tf.placeholder(shape=[None, 279], dtype=tf.float32)
b = tf.Variable(tf.random_normal(shape=[3, batch_size]))
gamma = tf.constant(-10.0)
dist = tf.reduce_sum(tf.square(x_data), 1)
dist = tf.reshape(dist, [-1, 1])
squ_dists = tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))
m_kernel = tf.exp(tf.multiply(gamma, tf.abs(squ_dists)))

def reshape_matmul(mat):
    v1 = tf.expand_dims(mat, 1)
    v2 = tf.reshape(v1, [3, batch_size, 1])
    return (tf.matmul(v2, v1))
term_1 = tf.reduce_sum(b)
b_v_cross = tf.matmul(tf.transpose(b), b)
y_tar_cross = reshape_matmul(y_target)
term_2 = tf.reduce_sum(tf.multiply(m_kernel, tf.multiply(b_v_cross, y_tar_cross)), [1, 2])
loss = tf.reduce_sum(tf.negative(tf.subtract(term_1, term_2)))
r_1 = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
r_2 = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1, 1])
pre_sq_dist = tf.add(tf.subtract(r_1, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))), tf.transpose(r_2))
pre_kernel = tf.exp(tf.multiply(gamma, tf.abs(pre_sq_dist)))
prediction_out = tf.matmul(tf.multiply(y_target, b), pre_kernel)
prediction = tf.argmax(prediction_out-tf.expand_dims(tf.reduce_mean(prediction_out, 1), 1), 0)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_target, 0)), tf.float32))
m_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = m_opt.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
loss_vec = []
batch_accuracy = []
for i in range(100):
    rand_index = np.random.choice(len(x_csv), size=batch_size)
    rand_x = x_csv[rand_index]
    rand_y = y_csv[:, rand_index]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid: rand_x})
    batch_accuracy.append(acc_temp)
saver.save(sess, "./model_1/model_1.ckpt")
plt.plot(batch_accuracy, 'k-', label='Accuracy')
plt.title('Batch Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
raw_test_data = csv.reader(open('K:/Pattern/test0/data_two_test.csv', 'r'))
ra_data = []
for i in raw_test_data:
    ra_data.append(i)
rb_data = np.array(ra_data)
rc_data = []
temp_data = []
for j in rb_data:
    temp_data = np.array([0.0 if a == '' else float(a) for a in j])
    rc_data.append(temp_data)
x_test_data = []
for b in rc_data:
    x_test_data.append(b[1:])
x_test = np.array(x_test_data)
y_test_data = np.array([[x[0]] for x in ra_data])
y_t1 = np.array([1 if y == '0' else -1 for y in y_test_data])
y_t2 = np.array([1 if y == '1' else -1 for y in y_test_data])
y_t3 = np.array([1 if y == '2' else -1 for y in y_test_data])
y_test = np.array([y_t1, y_t2, y_t3])
y_test = y_test.astype(np.float32)
saver = tf.train.Saver()
saver.restore(sess, "K:/Pattern/test0/model_1/model_1.ckpt")
test_accuracy = []
for i in range(10):
    rand_test = np.random.choice(len(x_test), size=batch_size)
    rand_x_test = x_test[rand_test]
    rand_y_test = y_test[:, rand_test]
    acc_test = sess.run(accuracy, feed_dict={x_data: rand_x_test, y_target: rand_y_test, prediction_grid: rand_x_test})
    test_accuracy.append(acc_test)
print(test_accuracy)
print('平均为', np.mean(test_accuracy))