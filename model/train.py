""" This file contains necessary things to train a model """

import tensorflow as tf
import time
from config import FLAGS
from utils import *

def train_model(sess, model, saver, placeholders, data_fetcher, 
                save_path = "SavedModel/model_rank.ckpt"):
    print('start optimization...')
    sess.run(tf.global_variables_initializer())
    train_start = time.time()
    for epoch in range(FLAGS.epochs):
    
        t = time.time()
    
        # Construct feed dictionary
        feed_dict = construct_feed_dict_prefetch(data_fetcher, placeholders)
        # Training step
        outs = sess.run([model.opt_op, model.loss], feed_dict=feed_dict)
    
        if (epoch+1) % 100 == 0:
            pred,lab = sess.run([model.pred, model.lab], feed_dict=feed_dict)
            print(pred)
            print(lab)
    
        # No Validation For Now

        # Print loss
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]), 
              "time=", "{:.5f}".format(time.time() - t))

#    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
#        print("Early stopping...")
#        break

    print("Optimization Finished, tiem cost {:.5f} s"\
          .format(time.time()-train_start))

    save_path = saver.save(sess, save_path)
    print("Model saved in path: {}".format(save_path))
