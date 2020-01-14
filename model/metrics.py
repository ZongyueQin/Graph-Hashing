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

#    closer2one = tf.sign(tf.nn.relu(codes-0.5))
#    loss_1 = tf.reduce_sum(closer2one * tf.abs(codes-1), axis=1)
#    loss_2 = tf.reduce_sum((1-closer2one)*tf.abs(codes), axis=1)
#    loss = loss_1 + loss_2

    """ make code close to +- 0.5 """
    
    loss_mat = tf.abs(codes**2-0.25)
    #loss = tf.reduce_sum(loss_mat, axis=1)
    return tf.reduce_mean(loss_mat)

def DSH_Loss(codes, label, m = FLAGS.hash_code_len/2):
    binary_label = label < FLAGS.GED_threshold
    binary_label = tf.cast(binary_label, tf.float32)
    """ make similar graphs close, dissimilar graphs distant """
    bs = FLAGS.batchsize
    k = FLAGS.k
    A1, A2 = tf.split(codes, [bs, bs*k])
    
    
    # Handle first part of loss
    # Some intermediate results
    M1 = tf.matmul(A1, tf.transpose(A1))
    diag = tf.squeeze(tf.matrix_diag_part(M1))
    M2 = tf.stack([diag for i in range(bs)])

    # pred{i,j} = ||d_i - d_j||^2
    pred_1 = (M2 + tf.transpose(M2) - 2*M1)
#    pred_1 = tf.clip_by_value(pred_1, 0, FLAGS.GED_threshold)
    #loss_mat_1 = tf.matrix_band_part((pred_1 - label_1)**2, 0, -1)
    loss_mat_1 = binary_label * pred_1 + (1 - binary_label) * tf.nn.relu(m-pred_1)
#    loss_mat_1 = tf.matrix_band_part(loss_mat_1, 0, -1)
    #loss_1 = tf.reduce_sum(loss_mat_1)
    loss_1 = tf.reduce_mean(loss_mat_1)
    
    # Handle second part of loss
    loss_2 = 0
    if k > 0:
        # intermediate results
        A2 = tf.reshape(A2, [bs, k, -1])
        A3 = tf.stack([A1 for i in range(k)], axis=1)
        
        pred_2 = tf.reduce_sum((A2-A3)**2, axis=2) 
        loss_mat_2 = pred_2
#        pred_2 = tf.clip_by_value(pred_2, 0, FLAGS.GED_threshold)
 #       loss_mat_2 = (label_2 - pred_2)**2
#        loss_2 = tf.reduce_sum(loss_mat_2)
        loss_2 = tf.reduce_mean(loss_mat_2)
        
    return FLAGS.real_data_loss_weight * loss_1 +\
           FLAGS.syn_data_loss_weight * loss_2, pred_1, loss_mat_1

def MSE_Loss_SimGNN(preds, label_1, label_2):
    bs = FLAGS.batchsize
    k = FLAGS.k

    label = tf.reshape(tf.concat([label_1, label_2], axis=1), 
                       [bs*(bs+k)])
    #label = tf.reshape(label_2, [bs*k])
    preds = tf.squeeze(preds)
    preds = tf.clip_by_value(preds, 0, FLAGS.GED_threshold)
    loss_vec = (preds-label)**2
    
    w_mask_1 = FLAGS.real_data_loss_weight * tf.ones([bs, bs], tf.float32)
    w_mask_2 = FLAGS.syn_data_loss_weight * tf.ones([bs, k], tf.float32)
    w_mask = tf.reshape(tf.concat([w_mask_1, w_mask_2], axis=1),
                        [bs*(bs+k)])
    
    loss = tf.reduce_mean(loss_vec*w_mask)
#    loss = tf.reduce_mean(loss_vec)    
    return loss, preds, label
    
    

def MSE_Loss(codes, label_1, label_2, bit_weights=None):

    a = FLAGS.exp_a

    bs = FLAGS.batchsize
    k = FLAGS.k
    A1, A2 = tf.split(codes, [bs, bs*k])
    
    
    # Handle first part of loss
    # Some intermediate results
    if FLAGS.bit_weight_type == 'const' or bit_weights is None:
        M1 = tf.matmul(A1, tf.transpose(A1))
        diag = tf.squeeze(tf.matrix_diag_part(M1))
        M2 = tf.stack([diag for i in range(bs)])
    else:
        D = tf.nn.relu(tf.matrix_diag(bit_weights))
        M1 = tf.matmul(tf.matmul(A1, D), tf.transpose(A1)) # M1 = A1@D@A1.T
        diag = tf.squeeze(tf.matrix_diag_part(M1))
        M2 = tf.stack([diag for i in range(bs)])

    # pred{i,j} = ||d_i - d_j||^2
    pred_1 = (M2 + tf.transpose(M2) - 2*M1)
    pred_1 = tf.clip_by_value(pred_1, 0, FLAGS.GED_threshold)
    loss_mat_1 = tf.matrix_band_part(tf.exp(a*(pred_1-label_1))*(pred_1 - label_1)**2, 0, -1)
    #loss_1 = tf.reduce_sum(loss_mat_1)
    loss_1 = tf.reduce_mean(loss_mat_1)
    
    # Handle second part of loss
    loss_2 = 0
    if k > 0:
        # intermediate results
        A2 = tf.reshape(A2, [bs, k, -1])
        A3 = tf.stack([A1 for i in range(k)], axis=1)
        if FLAGS.bit_weight_type == 'const' or bit_weights is None:
            pred_2 = tf.reduce_sum((A2-A3)**2, axis=2) 
        else:
            W_1 = tf.stack([bit_weights for i in range(k)], axis=0)
            W = tf.nn.relu(tf.stack([W_1 for i in range(bs)], axis=0))
            pred_2 = tf.reduce_sum(W*(A2-A3)**2, axis=2)

        pred_2 = tf.clip_by_value(pred_2, 0, FLAGS.GED_threshold)
        loss_mat_2 = tf.exp(a*(pred_2-label_2))*(label_2 - pred_2)**2
#        loss_2 = tf.reduce_sum(loss_mat_2)
        loss_2 = tf.reduce_mean(loss_mat_2)
        
    return FLAGS.real_data_loss_weight * loss_1 +\
           FLAGS.syn_data_loss_weight * loss_2, pred_1, label_1

    
def pair_accuracy(codes, labels):
    pass  
    
    
   
