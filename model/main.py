""" Integrate all parts """
# Below are moduels from outside
from __future__ import division
from __future__ import print_function
import time
import tensorflow as tf
from random import randint, sample
import numpy as np
import os
import subprocess
from tensorflow.python.tools import inspect_checkpoint as chkp
import sys

# Below are modules implemented by our own
from utils import *
from graphHashFunctions import GraphHash_Emb_Code
import numpy as np
from config import FLAGS
from DataFetcher import DataFetcher
import pickle
from train import train_model
from query import computeTrainingMSE, computeTestMSEWithoutGroundTruth
from query import computeTestMSEWithGroundTruth
from query import topKQuery, rangeQuery

""" environment configuration """
os.environ['CUDA_VISIBLE_DEVICES']='4,5'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

""" Specific which metrics to test """
test_top_k = True
test_range_query = True
train_mse = True
test_mse = True

""" train the model or load existing model """
train = True
model_path = ""
saved_files_dir = "SavedModel"

""" Set random seed """
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)


""" Create data fetcher """
data_fetcher = DataFetcher(dataset = FLAGS.dataset, exact_ged = True)
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


# Define placeholders
placeholders = {
    'support': tf.sparse_placeholder(tf.float32),
    'features': tf.sparse_placeholder(tf.float32, shape=(None, data_fetcher.get_node_feature_dim())),
    'labels': tf.placeholder(tf.float32, shape=(FLAGS.batchsize, FLAGS.batchsize)),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'graph_sizes': tf.placeholder(tf.int32, shape=((1+FLAGS.k)*FLAGS.batchsize)),
#    'graph_sizes': tf.placeholder(tf.int32, shape=(None)),
    'generated_labels':tf.placeholder(tf.float32, shape=(FLAGS.batchsize, FLAGS.k)),
    'thres':tf.placeholder(tf.float32, shape=(FLAGS.hash_code_len))
}



# Create model
model = GraphHash_Emb_Code(placeholders, 
                           input_dim=data_fetcher.get_node_feature_dim(),
                           next_ele = next_element,
                           logging=True)


# Initialize session
sess = tf.Session(config=config)

# Init variables
saver = tf.train.Saver()

cost_val = []

if train:
    train_model(sess, model, saver, placeholders, data_fetcher)

else:
    saver.restore(sess, model_path)
    print("Model restored from", model_path)


inverted_index, id2emb, id2code = encodeTrainingData(sess, model, 
                                                     data_fetcher, 
                                                     placeholders)

index_file = open('SavedModel/inverted_index_rank.pkl', 'wb')
pickle.dump(inverted_index, index_file)
index_file.close()
writeInvertedIndex(os.path.join(saved_files_dir, 'inverted_index.txt'),
                   inverted_index, 
                   FLAGS.embedding_dim)
# turn inverted index into format that can be processed by query.cpp or 
# topKQuery.cpp 
subprocess.check_output(['./processInvertedIndex', 
                        os.path.join(saved_files_dir, 'inverted_index.txt'),
                        os.path.join(saved_files_dir, 'inverted_index.index'), 
                        os.path.join(saved_files_dir, 'inverted_index.value')])

print('Saved index to ', 
      os.path.join(saved_files_dir, 'inverted_index_rank.pkl'))
 
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
                             placeholders)

else:
    if test_mse == True:
        computeTestMSEWithGroundTruth(sess, model, thres, placeholders, 
                                      data_fetcher, ground_truth,
                                      id2emb, id2code)
    
    if test_top_k == True:
        topKQuery(sess, model, data_fetcher, ground_truth,
              inverted_index,
              placeholders)
        
    if test_range_query == True:
        rangeQuery(sess, model, data_fetcher, ground_truth,
              placeholders,
              inverted_index)
        

