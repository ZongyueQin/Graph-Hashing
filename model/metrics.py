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

def binary_regularizer(codes):
    """ regularizer to force codes to be binary """
    distance_to_1 = tf.abs(codes)-1
    l1_norm = tf.reduce_sum(distance_to_1, axis=1)
    return tf.reduce_mean(l1_norm)

def DSH_loss(codes, labels, m):
    """ make similar graphs close, dissimilar graphs distant """
    code_1, code_2 = tf.split(codes, 2)
    distance = tf.reduce_sum((code_1 - code_2)**2, axis=1)
    loss = 0.5 * labels * distance + 0.5 * (1-labels) * tf.nn.relu(m-distance)
    return tf.reduce_mean(loss)

def pair_accuracy(codes, labels):
    pass  
    
    
    