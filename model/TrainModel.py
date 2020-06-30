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
#from query import computeTrainingMSE, computeTestMSEWithoutGroundTruth
#from query import computeTestMSEWithGroundTruth
#from query import topKQuery, rangeQuery
#from query import computeCSMTestMSEWithGroundTruth
#from query import rangeQueryCSM

""" environment configuration """
#os.environ['CUDA_VISIBLE_DEVICES']='2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


model_name = "0618_NoAug"+FLAGS.dataset
model_path = "SavedModel/"+model_name+".ckpt"
saved_files_dir = "SavedModel"

""" Set random seed """
random_seed = 123
np.random.seed(random_seed)
tf.set_random_seed(random_seed)
seed(random_seed)



""" Create data fetcher """

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


# Define placeholders
placeholders = {
    'support': tf.sparse_placeholder(tf.float32),
    'features': tf.sparse_placeholder(tf.float32, shape=(None, data_fetcher.get_node_feature_dim())),
    'labels': tf.placeholder(tf.float32, shape=(FLAGS.batchsize, FLAGS.batchsize)),
    'dropout': tf.placeholder_with_default(0., shape=()),
#    'graph_sizes': tf.placeholder(tf.int32, shape=(FLAGS.batchsize*(1+FLAGS.k))),
#    'graph_sizes': tf.placeholder(tf.int32, shape=(None)),
    'graph_sizes': tf.placeholder(tf.int32, shape=(FLAGS.ecd_batchsize)),
    'generated_labels':tf.placeholder(tf.float32, shape=(FLAGS.batchsize, FLAGS.k)),
    'thres':tf.placeholder(tf.float32, shape=(FLAGS.hash_code_len))
}



# Create model
mapper_dict = {0:[0], 1:[1], 2:[1], 3:[1], 4:[1], 5:[1], 6:[1], 7:[1], 8:[2], 9:[3,4,5], 10:[6,7,8,9,10]}

def mapper(x):
    return sum(mapper_dict[x])/len(mapper_dict[x])

model = GraphHash_Emb_Code(placeholders, 
                           input_dim=data_fetcher.get_node_feature_dim(),
                           next_ele = next_element,
#                           mapper = lambda x:x,
#                           mapper = mapper,
                           logging=True)


# Initialize session
sess = tf.Session(config=config)

# Init variables
saver = tf.train.Saver()


cost_val = []
#sess.run(tf.global_variables_initializer())
train_model(sess, model, saver, placeholders, data_fetcher, save_path=model_path)
#    csm.train(sess, csm_saver)
#model_path = "SavedModel/"+model_name+".ckpt"
#saver.restore(sess, model_path)
 
start_time = time.time()
inverted_index, bit_weights = encodeTrainingData(sess, model, 
                                                     data_fetcher, 
                                                     placeholders,
                                                     True, True)
#inverted_index, id2emb, id2code, bit_weights = encodeDataInDir(sess, model,
#                                                  data_fetcher,
#                                                  data_fetcher.train_data_dir,
#                                                  placeholders)
print('encoding data, cost %.5f s'%(time.time()-start_time))

index_file = open('SavedModel/inverted_index_'+model_name+'.pkl', 'wb')
pickle.dump(inverted_index, index_file)
index_file.close()
#writeInvertedIndex(os.path.join(saved_files_dir, 'inverted_index_'+model_name+'.txt'),
writeIndex(os.path.join(saved_files_dir, 'inverted_index_'+model_name+'.txt'), 
            inverted_index, 
            FLAGS.embedding_dim)
print('Bit Weights:')
print(bit_weights)
writeBitWeights(os.path.join(saved_files_dir, 'bit_weights_'+model_name+'.txt'),
        bit_weights)

#writeMapperDict(model.mapper, os.path.join(saved_files_dir, 'mapper_dict_'+model_name+'.txt'))

print('model training finish')
