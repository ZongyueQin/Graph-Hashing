import time
import pickle
import tensorflow as tf
import numpy as np
import os
import sys

import random

from utils import *
from graphHashFunctions import GraphHash_Emb_Code
from config import FLAGS
from DataFetcher import DataFetcher

os.environ['CUDA_VISIBLE_DEVICES']='4'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

MinGED = 0
MaxGED = 11
MaxGraphNum = 999900

if len(sys.argv) != 3:
    print('parameters are model_path, output_name')
    os._exit(0)
model_path = str(sys.argv[1])
output_fname = str(sys.argv[2])


""" Load datafetcher and model"""
print('restoring model...')
data_fetcher = DataFetcher(dataset=FLAGS.dataset, exact_ged=True, max_graph_num=MaxGraphNum)
node_feature_dim = data_fetcher.get_node_feature_dim()
# Define placeholders
placeholders = {
    'support': tf.sparse_placeholder(tf.float32),
    'features': tf.sparse_placeholder(tf.float32, shape=(None, node_feature_dim)),
#    'labels': tf.placeholder(tf.float32, shape=(FLAGS.batchsize, FLAGS.batchsize)),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'graph_sizes': tf.placeholder(tf.int32, shape=(FLAGS.ecd_batchsize)),
#    'graph_sizes': tf.placeholder(tf.int32, shape=(FLAGS.batchsize*(1+FLAGS.k))),
#    'generated_labels':tf.placeholder(tf.float32, shape=(FLAGS.batchsize, FLAGS.k)),
    'thres':tf.placeholder(tf.float32, shape=(FLAGS.hash_code_len))
}

thres = np.zeros(FLAGS.hash_code_len)

# Create Model
#model = GraphHash_Emb_Code_Mapper(placeholders, 
model = GraphHash_Emb_Code(placeholders,
                           input_dim=node_feature_dim,
                           next_ele = None,
#                           mapper=None,
                           logging=True)

# Initialize session
sess = tf.Session(config=config)
# Init variables
saver = tf.train.Saver()
saver.restore(sess, model_path)
print("Model restored from", model_path)

# encoding training data
start_time = time.time()
index, bit_weights = encodeTrainingData(sess, model, 
                                                     data_fetcher, 
                                                     placeholders,
                                                     True, True)
#inverted_index, id2emb, id2code, bit_weights = encodeDataInDir(sess, model,
#                                                  data_fetcher,
#                                                  data_fetcher.train_data_dir,
#                                                  placeholders)
print('encoding data, cost %.5f s'%(time.time()-start_time))

index_file = open('SavedModel/Index_'+output_fname+'.pkl', 'wb')
pickle.dump(index, index_file)
index_file.close()
saved_files_dir = "SavedModel"
writeIndex(os.path.join(saved_files_dir, 'Index_'+output_fname+'.txt'),
                   index, 
                   FLAGS.embedding_dim)
print('Bit Weights:')
print(bit_weights)
writeBitWeights(os.path.join(saved_files_dir, 'bit_weights_'+output_fname+'.txt'),
        bit_weights)


