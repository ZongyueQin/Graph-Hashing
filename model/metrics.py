import tensorflow as tf
from config import FLAGS

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

def l1_loss(codes):
     return FLAGS.l1_loss_w * tf.reduce_sum(tf.abs(codes))

def binary_regularizer(codes):
    """ regularizer to force codes to be binary """
#    distance_to_1 = tf.abs(tf.abs(codes)-1)
#    l1_norm = tf.reduce_sum(distance_to_1, axis=1)
#    return tf.reduce_mean(l1_norm)
    closer2one = tf.sign(tf.nn.relu(codes-0.5))
    loss_1 = tf.reduce_sum(closer2one * tf.abs(codes-1), axis=1)
    loss_2 = tf.reduce_sum((1-closer2one)*tf.abs(codes), axis=1)
    loss = loss_1 + loss_2
    return tf.reduce_mean(loss)

def DSH_loss(codes, labels, m):
    """ make similar graphs close, dissimilar graphs distant """
    # First split sampled data and generated data
    bs = FLAGS.batchsize
    k = FLAGS.k
    A1, A2 = tf.split(codes, [bs, bs*k])
    # Handle first part of loss
    M1 = tf.matmul(A1, tf.transpose(A1))
    diag = tf.squeeze(tf.matrix_diag_part(M1))
    M2 = tf.stack([diag for i in range(bs)])
    l2_mat = tf.matrix_band_part((M2 + tf.transpose(M2) - 2*M1), 0, -1)
    loss_mat = labels * l2_mat + (1-labels) * tf.nn.relu(m-l2_mat)
    loss_1 = tf.reduce_mean(loss_mat)
    
    # Handle second part of loss
    A2 = tf.reshape(A2, [bs, k, -1])
    A3 = tf.stack([A1 for i in range(k)], axis=1)
    loss_2 = tf.reduce_mean(tf.reduce_sum((A2-A3)**2, axis=2)) 
    
    return loss_1 + loss_2

def MSE_Loss(codes, label_1, label_2):
    bs = FLAGS.batchsize
    k = FLAGS.k
    A1, A2 = tf.split(codes, [bs, bs*k])
    # Handle first part of loss
    M1 = tf.matmul(A1, tf.transpose(A1))
    diag = tf.squeeze(tf.matrix_diag_part(M1))
    M2 = tf.stack([diag for i in range(bs)])
    # l2_mat_{i,j} = ||d_i - d_j||^2
    l2_mat_1 = (M2 + tf.transpose(M2) - 2*M1)
    loss_mat_1 = tf.matrix_band_part((l2_mat_1 - label_1)**2, 0, -1)
    loss_1 = tf.reduce_sum(loss_mat_1)
    # Handle second part of loss
    grad = tf.gradients(loss_1, codes)
    loss_2 = 0
    if k > 0:
      A2 = tf.reshape(A2, [bs, k, -1])
      A3 = tf.stack([A1 for i in range(k)], axis=1)
      l2_mat_2 = tf.reduce_sum((A2-A3)**2, axis=2) 
      loss_mat_2 = (label_2 - l2_mat_2)**2
      loss_2 = tf.reduce_sum(loss_mat_2)
    
    return FLAGS.real_data_loss_weight * loss_1 +\
           FLAGS.syn_data_loss_weight * loss_2, l2_mat_1, label_1

    
def pair_accuracy(codes, labels):
    pass  
    
    
    
