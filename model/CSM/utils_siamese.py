from CSM_config import CSM_FLAGS
import sys
from os.path import dirname, abspath, join
import tensorflow as tf
import numpy as np
from math import exp

cur_folder = dirname(abspath(__file__))
#sys.path.insert(0, '{}/../../src'.format(cur_folder))

from CSM_utils import sorted_nicely, get_ts


def solve_parent_dir():
    pass


def check_flags():
    if CSM_FLAGS.csm_node_feat_name:
        assert (CSM_FLAGS.csm_node_feat_encoder == 'onehot')
    else:
        assert ('constant_' in CSM_FLAGS.csm_node_feat_encoder)
    assert (0 < CSM_FLAGS.csm_valid_percentage < 1)
    assert (CSM_FLAGS.csm_layer_num >= 1)
    assert (CSM_FLAGS.csm_batch_size >= 1)
    assert (CSM_FLAGS.csm_iters >= 0)
    assert (CSM_FLAGS.csm_iters_val_start >= 1)
    assert (CSM_FLAGS.csm_iters_val_every >= 1)
    assert (CSM_FLAGS.csm_gpu >= -1)
    d = CSM_FLAGS.flag_values_dict()
    ln = d['csm_layer_num']
    ls = [False] * ln
    for k in d.keys():
        print(k)
        if 'layer_' in k and 'gc' not in k and 'branch' not in k and 'id' not in k:
            lt = k.split('_')[2]
            print(lt)
            if lt != 'num':
                i = int(lt) - 1
                if not (0 <= i < len(ls)):
                    raise RuntimeError('Wrong spec {}'.format(k))
                ls[i] = True
    for i, x in enumerate(ls):
        if not x:
            raise RuntimeError('layer {} not specified'.format(i + 1))
    if is_transductive():
        assert (CSM_FLAGS.csm_layer_num == 1)  # can only have one layer
        assert (CSM_FLAGS.csm_gemb_dim >= 1)
        assert (not CSM_FLAGS.csm_dataset_super_large)
    if CSM_FLAGS.csm_need_gc:
        assert (CSM_FLAGS.csm_gc_bert_or_semi in ['bert', 'semi'])
    assert (CSM_FLAGS.csm_train_real_percent >= 0 and CSM_FLAGS.csm_train_real_percent <= 1)
    # TODO: finish.


def get_flags(k, check=False):
    if hasattr(CSM_FLAGS, k):
        return getattr(CSM_FLAGS, k)
    else:
        if check:
            raise RuntimeError('Need flag {} which does not exist'.format(k))
        return None


def extract_config_code():
    with open(join(get_siamese_dir(), 'CSM_config.py')) as f:
        return f.read()


def convert_msec_to_sec_str(sec):
    return '{:.2f}msec'.format(sec * 1000)


def convert_long_time_to_str(sec):
    day = sec // (24 * 3600)
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    minutes = sec // 60
    sec %= 60
    seconds = sec
    return '{} days {} hours {} mins {:.1f} secs'.format(
        int(day), int(hour), int(minutes), seconds)


def get_siamese_dir():
    return cur_folder


def get_coarsen_level():
    if CSM_FLAGS.csm_coarsening:
        return int(CSM_FLAGS.csm_coarsening[6:])
    else:
        return 1


def is_transductive():
    return 'transductive' in CSM_FLAGS.csm_model


def get_model_info_as_str(model_info_table=None):
    rtn = []
    d = CSM_FLAGS.flag_values_dict()
    for k in sorted_nicely(d.keys()):
        v = d[k]
        s = '{0:26} : {1}'.format(k, v)
        rtn.append(s)
        if model_info_table:
            model_info_table.append([k, '**{}**'.format(v)])
    rtn.append('{0:26} : {1}'.format('ts', get_ts()))
    return '\n'.join(rtn)


def need_val(iter):
    return False
    """
    assert (iter != 0)  # 1-based iter
    if CSM_FLAGS.csm_dataset_super_large or CSM_FLAGS.csm_dataset_val_test == 'nci109':
        return False  # no val for super large dataset
    return (iter >= CSM_FLAGS.csm_iters_val_start and
            (iter - CSM_FLAGS.csm_iters_val_start) % CSM_FLAGS.csm_iters_val_every == 0)
    """

def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def create_activation(act, ds_kernel=None, use_tf=True):
    if act == 'relu':
        return tf.nn.relu if use_tf else relu_np
    elif act == 'identity':
        return tf.identity if use_tf else identity_np
    elif act == 'sigmoid':
        return tf.sigmoid if use_tf else sigmoid_np
    elif act == 'tanh':
        return tf.tanh if use_tf else np.tanh
    elif act == 'ds_kernel':
        return ds_kernel.dist_to_sim_tf if use_tf else \
            ds_kernel.dist_to_sim_np
    else:
        raise RuntimeError('Unknown activation function {}'.format(act))


def relu_np(x):
    return np.maximum(x, 0)


def identity_np(x):
    return x


def sigmoid_np(x):
    try:
        ans = exp(-x)
    except OverflowError:  # TODO: fix
        ans = float('inf')
    return 1 / (1 + ans)


def truth_is_dist_sim():
    if CSM_FLAGS.csm_ds_metric == 'ged':
        sim_or_dist = 'dist'
    else:
        assert (CSM_FLAGS.csm_ds_metric == 'mcs')
        sim_or_dist = 'sim'
    return sim_or_dist


def reset_flag(func, str, v):
    delattr(CSM_FLAGS, str)
    func(str, v, '')


def clear_all_flags():
    for k in CSM_FLAGS.flag_values_dict():
        delattr(CSM_FLAGS, k)
