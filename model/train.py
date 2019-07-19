from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import construct_feed_dict_for_train, construct_feed_dict_for_encode
from utils import construct_feed_dict_for_query
from utils import get_similar_graphs_gid
from graphHashFunctions import GraphHash_Naive
import numpy as np
from config import FLAGS
from DataFetcher import DataFetcher
import pickle
# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Load data
data_fetcher = DataFetcher('PTC')

# Some preprocessing

# Define placeholders
placeholders = {
    'support': tf.sparse_placeholder(tf.float32),
    'features': tf.sparse_placeholder(tf.float32, shape=(None, data_fetcher.get_node_feature_dim())),
    'labels': tf.placeholder(tf.float32, shape=(None)),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    'graph_sizes': tf.placeholder(tf.int32, shape=(2*FLAGS.batchsize))
}

# Create model
model = GraphHash_Naive(placeholders, input_dim=data_fetcher.get_node_feature_dim(), 
                        logging=True)

# Initialize session
sess = tf.Session()
# Init variables
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
cost_val = []

print('start optimization...')
# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict_for_train(data_fetcher, placeholders)

    # Training step
    outs = sess.run([model.opt_op, model.loss], feed_dict=feed_dict)

    # Validation

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]), 
          "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

save_path = saver.save(sess, "SavedModel/model.ckpt")
print("Model saved in path: {}".format(save_path))

print('start encoding training data...')
size = data_fetcher.get_train_graphs_num()
inverted_index = {}
encode_batchsize = 2 * FLAGS.batchsize
for i in range(0,size, encode_batchsize):
    end = i + encode_batchsize
    if end > size:
        end = size
    idx_list = list(range(i,end))
    # To adjust to the size of placeholders, we add some graphs for padding
    while (len(idx_list) < encode_batchsize):
        idx_list.append(0)
    feed_dict = construct_feed_dict_for_encode(data_fetcher, placeholders, idx_list)
    codes = sess.run(model.codes, feed_dict = feed_dict)
    for j, code in enumerate(codes):
        # ignore all padding graphs
        if i + j >= size:
            break
        tuple_code = tuple(code)
        gid = data_fetcher.get_train_graph_gid(i+j)
        inverted_index.setdefault(tuple_code, [])
        inverted_index[tuple_code].append(gid)

index_file = open('SavedModel/inverted_index.pkl', 'wb')
pickle.dump(inverted_index, index_file)
index_file.close()
print('finish encoding, saved index to SavedModel/inverted_index.pkl')

print('Start testing')
total_query_num = data_fetcher.get_test_graphs_num()
for i in range(0, total_query_num, encode_batchsize):
    
    end = i + encode_batchsize
    if end > total_query_num:
        end = total_query_num
    idx_list = list(range(i,end))
    # To adjust to the size of placeholders, we add some graphs for padding
    while (len(idx_list) < encode_batchsize):
        idx_list.append(0)
    feed_dict = construct_feed_dict_for_query(data_fetcher, placeholders, idx_list)
    codes = sess.run(model.codes, feed_dict = feed_dict)
    for j, code in enumerate(codes):
        # ignore all padding graphs
        if i + j >= size:
            break
        tuple_code = tuple(code)
        similar_sets = get_similar_graphs_gid(inverted_index, tuple_code)
        # TODO compute precision of similar_sets and groud_truth sets

