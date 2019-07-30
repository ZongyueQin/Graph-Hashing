from metrics import MSE_Loss
import tensorflow as tf
import pickle
import numpy as np

f = open('SavedModel/inverted_index_rank.pkl','rb')
D=pickle.load(f)
codes = []
for key in D.keys():
    lis = D[key]
    for pair in lis:
        codes.append(list(pair[1]))
codes = tf.constant(codes)
labels = 8*np.ones((5,5))
for i in range(5):
    labels[i,i]=0
#loss = MSE_Loss(codes, labels, None)
sess = tf.Session()
#a = sess.run(loss)
bs = 5
k = 0
A1, A2 = tf.split(codes, [bs, bs*k])
    # Handle first part of loss
M1 = tf.matmul(A1, tf.transpose(A1))
diag = tf.squeeze(tf.matrix_diag_part(M1))
M2 = tf.stack([diag for i in range(bs)])
    # l2_mat_{i,j} = ||d_i - d_j||^2
l2_mat_1 = (M2 + tf.transpose(M2) - 2*M1)
loss_mat_1 = tf.matrix_band_part((l2_mat_1 - labels)**2, 0, -1)
loss_1 = tf.reduce_sum(loss_mat_1)

lis = sess.run([A1,A2,M1,diag,M2,l2_mat_1,loss_mat_1,loss_1])
print(lis)
    
