""" Integrate all parts """
# Below are moduels from outside
from __future__ import division
from __future__ import print_function
import time
import tensorflow as tf
from random import randint, sample, seed
import numpy as np
import os
import subprocess
from tensorflow.python.tools import inspect_checkpoint as chkp
import sys

cur_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '{}/CSM'.format(cur_folder))

# Below are modules implemented by our own
from utils import *
from graphHashFunctions import GraphHash_Emb_Code
import numpy as np
from config import FLAGS
from DataFetcher import DataFetcher
import pickle
from train import train_model, train_GH_CSM
from query import computeTrainingMSE, computeTestMSEWithoutGroundTruth
from query import computeTestMSEWithGroundTruth
from query import topKQuery, rangeQuery
from query import computeCSMTestMSEWithGroundTruth
from query import rangeQueryCSM

from CSM import CSM
from saver import Saver as CSM_Saver
from CSMDataFetcher import CSMDataFetcher
from CSM_config import CSM_FLAGS

""" environment configuration """
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

""" Specific which metrics to test """
use_csm = False
test_top_k = False
test_range_query = True
train_mse = False
test_mse = False
test_csm_mse = False
test_range_query_csm = False

""" train the model or load existing model """
train = True
model_name = "0902:code_emb_ours_"+FLAGS.dataset
model_path = "SavedModel/"+model_name+".ckpt"
saved_files_dir = "SavedModel"

""" Set random seed """
random_seed = 123
np.random.seed(random_seed)
tf.set_random_seed(random_seed)
seed(random_seed)


""" Create data fetcher """

if use_csm:
    assert(FLAGS.dataset == CSM_FLAGS.csm_dataset)
#data_fetcher = DataFetcher(dataset = FLAGS.dataset, exact_ged = True)
data_fetcher = DataFetcher(dataset=FLAGS.dataset, exact_ged=True)
# wrap the data fetcher with tensorflow.dataset.prefetch to accelerate training
dataset = tf.data.Dataset.from_generator(data_fetcher.get_train_data, 
                                         (tf.int64, tf.float32, tf.int64,
                                          tf.int64, tf.float32, tf.int64,
                                          tf.int32, tf.float32, tf.float32), 
                                          (tf.TensorShape([None,2]),#feature, sparse index
                                           tf.TensorShape([None]), # feature sparse value
                                           tf.TensorShape([2]), # feature shape
                                           tf.TensorShape([None,2]),# laplacian sparse index
                                           tf.TensorShape([None]), #laplacian sparse value
                                           tf.TensorShape([2]), # laplacian sparse shape
                                           tf.TensorShape([(1+FLAGS.k)*FLAGS.batchsize]), #shape
                                           #tf.TensorShape([None]), # shape
                                           tf.TensorShape([FLAGS.batchsize, FLAGS.batchsize]),#label
                                           tf.TensorShape([FLAGS.batchsize, FLAGS.k]),# gen_label
                                           ))
dataset = dataset.prefetch(buffer_size=1)
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
next_element = construct_input(one_element)
if use_csm:
    csm_data_fetcher = CSMDataFetcher(dataset=CSM_FLAGS.csm_dataset, exact_ged=False)


# Define placeholders
placeholders = {
    'support': tf.sparse_placeholder(tf.float32),
    'features': tf.sparse_placeholder(tf.float32, shape=(None, data_fetcher.get_node_feature_dim())),
    'labels': tf.placeholder(tf.float32, shape=(FLAGS.batchsize, FLAGS.batchsize)),
    'dropout': tf.placeholder_with_default(0., shape=()),
#    'graph_sizes': tf.placeholder(tf.int32, shape=(FLAGS.batchsize*(1+FLAGS.k))),
#    'graph_sizes': tf.placeholder(tf.int32, shape=(None)),
    'graph_sizes': tf.placeholder(tf.int32, shape=(1)),
    'generated_labels':tf.placeholder(tf.float32, shape=(FLAGS.batchsize, FLAGS.k)),
    'thres':tf.placeholder(tf.float32, shape=(FLAGS.hash_code_len))
}



# Create model
model = GraphHash_Emb_Code(placeholders, 
                           input_dim=data_fetcher.get_node_feature_dim(),
                           next_ele = next_element,
                           logging=True)

if use_csm:
    csm = CSM(csm_data_fetcher)

# Initialize session
sess = tf.Session(config=config)

# Init variables
saver = tf.train.Saver()

if use_csm:
    csm_saver = CSM_Saver(sess)

cost_val = []

if train:
    if use_csm:
        train_GH_CSM(sess, model, saver, placeholders, data_fetcher, 
                     csm, csm_saver, csm_data_fetcher, save_path = model_path)
    else:
        train_model(sess, model, saver, placeholders, data_fetcher, save_path=model_path)
#    csm.train(sess, csm_saver)
else:
    saver.restore(sess, model_path)
    print("Model restored from", model_path)


inverted_index, id2emb, id2code = encodeTrainingData(sess, model, 
                                                     data_fetcher, 
                                                     placeholders,
                                                     True, True)

index_file = open('SavedModel/inverted_index_'+model_name+'.pkl', 'wb')
pickle.dump(inverted_index, index_file)
index_file.close()
writeInvertedIndex(os.path.join(saved_files_dir, 'inverted_index_'+model_name+'.txt'),
                   inverted_index, 
                   FLAGS.embedding_dim)
# turn inverted index into format that can be processed by query.cpp or 
# topKQuery.cpp 
inv_idx_txt = os.path.join(saved_files_dir, 'inverted_index_'+model_name+'.txt')
inv_idx_idx = os.path.join(saved_files_dir, 'inverted_index_'+model_name+'.index')
inv_idx_val = os.path.join(saved_files_dir, 'inverted_index_'+model_name+'.value')
subprocess.check_output(['./processInvertedIndex', 
                        inv_idx_txt,
                        inv_idx_idx, 
                        inv_idx_val])

print('Saved index to ', 
      os.path.join(saved_files_dir, 'inverted_index_'+model_name+'.pkl'))
 
thres = np.zeros(FLAGS.hash_code_len)
if train_mse == True:
    computeTrainingMSE(sess, model, thres, data_fetcher, placeholders)


total_query_num = data_fetcher.get_test_graphs_num()

# Read Ground Truth

ground_truth_path = os.path.join('..','data',
                                 FLAGS.dataset,
                                 'test',
                                 FLAGS.ground_truth_file)
try:
    f = open(ground_truth_path, 'r')
    ground_truth, ged_cnt = readGroundTruth(f)
    has_GT = True
except IOError:
    print('Groundtruth file doesn\'t exist, ignore top-k and range query')
    has_GT = False



# Start testing


if not has_GT:
    if test_mse == True:
        computeTestMSEWithoutGroundTruth(model, sess, thres, 
                             data_fetcher, 
                             placeholders,
                             encode_batchsize = 1)

else:

    if test_mse == True:
        computeTestMSEWithGroundTruth(sess, model, thres, placeholders, 
                                      data_fetcher, ground_truth,
                                      id2emb, id2code, encode_batchsize=1)
    
    if test_top_k == True:
        topKQuery(sess, model, data_fetcher, ground_truth,
              inverted_index,
              placeholders,
              index_index_fname=inv_idx_idx,
              index_value_fname=inv_idx_val)
     
    if test_csm_mse == True:
        computeCSMTestMSEWithGroundTruth(sess, csm_saver, csm,
                                         csm_data_fetcher, 
                                         ground_truth)
   
    if test_range_query_csm:
        rangeQueryCSM(sess, 
                      csm,
                      csm_data_fetcher,
                      csm_saver,
                      ground_truth,
                      t_min = 1, t_max=6)

    if test_range_query == True:
        if not use_csm:
            rangeQuery(sess, model, data_fetcher, ground_truth,
              placeholders, inverted_index, 
              t_min=1, t_max=3,
              index_index_fname=inv_idx_idx,
              index_value_fname=inv_idx_val)

        else:
            rangeQuery(sess, model, data_fetcher, ground_truth,
              placeholders, inverted_index, 
              csm=csm, csm_data_fetcher = csm_data_fetcher,
              csm_saver = csm_saver,
              t_min=1, t_max=6,
              index_index_fname=inv_idx_idx,
              index_value_fname=inv_idx_val, use_code=True, use_emb=True)
        

