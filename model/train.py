""" This file contains necessary things to train a model """

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from random import shuffle
import numpy as np

import tensorflow as tf
import time
from config import FLAGS
#from CSM_config import CSM_FLAGS
from utils import *
#from CSM_train import run_tf
#from utils_siamese import convert_msec_to_sec_str

def initProximityMat(sess, model, placeholders, data_fetcher):
    if len(data_fetcher.train_graphs) < FLAGS.sample_pool_size:
        raise RuntimeError('FLAGS.sample_pool size is larger than total graph number')
    if FLAGS.sample_pool_size % FLAGS.ecd_batchsize != 0:
        raise RuntimeError('sample_pool_size % ecd_batchsize has to be 0')

    shuffle(data_fetcher.train_graphs)

    sample_embs = np.zeros((FLAGS.sample_pool_size, FLAGS.embedding_dim))
    bs = FLAGS.ecd_batchsize
    for i in range(0,FLAGS.sample_pool_size, bs):
        idx_list = list(range(i,i+bs))
        feed_dict = construct_feed_dict_for_encode(data_fetcher, 
                                                   placeholders, 
                                                   idx_list,
                                                   'train')
        embs = sess.run(model.ecd_embeddings, feed_dict = feed_dict)
        sample_embs[i:i+bs,:]=np.array(embs)

    M1 = sample_embs@sample_embs.T
    diag = np.squeeze(np.diag(M1))
    M2 = np.stack([diag for i in range(FLAGS.sample_pool_size)])
    proximityMat= M2+M2.T-2*M1

    data_fetcher.positive_cum_p = np.exp(-proximityMat)
    data_fetcher.negative_cum_p = np.exp(proximityMat)
    data_fetcher.sample_ptr = 0
    data_fetcher.sample_bias = 0
    return sample_embs

def updateProximityMat(sess, model, placeholders, data_fetcher, sample_embs):
    bs = FLAGS.ecd_batchsize
    sample_embs[0:-bs,:]=sample_embs[bs:,:]

    i = data_fetcher.sample_ptr + data_fetcher.sample_bias
    end = i + bs
    N = len(data_fetcher.train_graphs)
    if end > N:
        end = N
    idx_list = list(range(i,end))+list(range(0,bs-(end-i)))
    feed_dict = construct_feed_dict_for_encode(data_fetcher,
                                               placeholders,
                                               idx_list,
                                               'train')
    embs = sess.run(model.ecd_embeddings, feed_dict = feed_dict)
    sample_embs[-bs:,:]=np.array(embs)

    M1 = sample_embs@sample_embs.T
    diag = np.squeeze(np.diag(M1))
    M2 = np.stack([diag for i in range(FLAGS.sample_pool_size)])
    proximityMat= M2+M2.T-2*M1

    data_fetcher.positive_cum_p = np.exp(-proximityMat)
    data_fetcher.negative_cum_p = proximityMat
#    data_fetcher.sample_ptr = 0
#    data_fetcher.sample_bias = data_fetcher.sample_bias + bs
 



def train_model(sess, model, saver, placeholders, data_fetcher, 
                save_path = "SavedModel/model_rank.ckpt"):
    print('start optimization...')
    sess.run(tf.global_variables_initializer())
    if FLAGS.sample_by_proximity == True:
        sample_embs = initProximityMat(sess, model, placeholders, data_fetcher)

    train_start = time.time()
    last_large_range_losses = []
    last_small_range_losses = []
    train_losses = []
    for epoch in range(FLAGS.epochs):
    
        t = time.time()
    
        # Construct feed dictionary
        feed_dict = construct_feed_dict_prefetch(data_fetcher, placeholders)
        # Training step
        outs = sess.run([model.opt_op, model.loss, model.pred, model.lab], feed_dict=feed_dict)
        train_losses.append(outs[1])
        if epoch % 100 == 0:
            print(outs[2])
            print(outs[3])

        if FLAGS.sample_by_proximity == True and (epoch+1) % FLAGS.update_iter_num == 0:
            data_fetcher.sample_ptr = 0
            data_fetcher.sample_bias = data_fetcher.sample_bias + FLAGS.ecd_batchsize
            if data_fetcher.sample_bias > len(data_fetcher.train_graphs):
                print('Reinitialize proximity matrix...')
                sample_embs = initProximityMat(sess, model, placeholders, data_fetcher)
            else:
                print('update proximity matrix...')
                updateProximityMat(sess, model, placeholders, data_fetcher, sample_embs)



        last_large_range_losses.append(outs[1])
        if len(last_large_range_losses) > FLAGS.early_stopping_large_range:
            last_large_range_losses.pop(0)
        last_small_range_losses.append(outs[1])
        if len(last_small_range_losses) > FLAGS.early_stopping_small_range:
            last_small_range_losses.pop(0)

#        if (epoch+1) % 100 == 0:
#            pred,lab = sess.run([model.pred, model.lab], feed_dict=feed_dict)
#            print(pred)
#            print(lab)
    
        # No Validation For Now

        # Print loss
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]), 
              "time=", "{:.5f}".format(time.time() - t))

        if epoch > FLAGS.early_stopping_large_range and epoch % FLAGS.early_stopping_check_frequency == 0:
            if np.mean(last_large_range_losses)-np.mean(last_small_range_losses) < FLAGS.early_stopping_thres:
                print("Early stopping...")
                break
    plt.plot(train_losses)
    plt.xlabel('iterations')
    plt.ylabel('training loss')
    plt.savefig('Graph-Hashing-train-loss.eps')
    
    print("Optimization Finished, tiem cost {:.5f} s"\
          .format(time.time()-train_start))

    save_path = saver.save(sess, save_path)
    print("Model saved in path: {}".format(save_path))

def train_GH_CSM(sess, model, saver, plhdr, data_fetcher, 
                 csm, csm_saver, csm_data_fetcher,
                 save_path="SavedModel/model_csm.ckpt"):
    print('start optimization...')
    sess.run(tf.global_variables_initializer())
    train_start = time.time()
    GH_train_losses = []
    CSM_train_costs, train_times = [], []

    max_iter = max([FLAGS.epochs,CSM_FLAGS.csm_iters])
    
    need_gc = False
    tvt = 'train'
    for Iter in range(1, max_iter+1):
    
        t = time.time()
    
        # Construct feed dictionary
        feed_dict = {}
        target = []
        if Iter <= FLAGS.epochs:    
            GH_feed_dict = construct_feed_dict_prefetch(data_fetcher, plhdr)
            target = target + [model.opt_op, model.loss]
            feed_dict.update(GH_feed_dict)

        csm_feed_dict = {}
        if Iter <= CSM_FLAGS.csm_iters:
            csm_feed_dict = csm.model.get_feed_dict_for_train(need_gc)
            feed_dict.update(csm_feed_dict)
            objs = [csm.model.opt_op, csm.model.train_loss]
            objs = csm_saver.proc_objs(objs, 'train', Iter)  # may become [loss_related_obj, objs...]
            target = target + objs 
        print('after get feed dict, %f s'%(time.time()-t))

        # Training step
        t = time.time()
        outs = sess.run(target, 
                       feed_dict=feed_dict)

        time_rtn = time.time() - t

        if Iter <= FLAGS.epochs:
            # Print loss
            print("Iter:", '%04d' % Iter, "GH train_loss=", "{:.5f}".format(outs[1]), 
                  "time=", "{:.5f}".format(time_rtn))
            GH_train_losses.append(outs[1])

        if Iter <= CSM_FLAGS.csm_iters: 
            train_cost = outs[-1]
            train_time = time_rtn
            CSM_train_costs.append(train_cost)
            s = 'Iter:{:04n}  CSM train_loss={:.5f} time={}'.format(
                Iter, train_cost, convert_msec_to_sec_str(train_time))
            print(s)

        if Iter % 100 == 0:
            pred,lab = sess.run([model.pred, model.lab], feed_dict=GH_feed_dict)
            print(pred)
            print(lab)
    
        # No Validation For Now


#    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
#        print("Early stopping...")
#        break
    
    print("Optimization Finished, tiem cost {:.5f} s"\
          .format(time.time()-train_start))

    plt.plot(GH_train_losses)
    plt.xlabel('iterations')
    plt.ylabel('training loss')
    plt.savefig('Graph-Hashing-train-loss.eps')

    save_path = saver.save(sess, save_path)
    print("Model saved in path: {}".format(save_path))

    plt.plot(CSM_train_costs)
    plt.xlabel('iterations')
    plt.ylabel('training loss')
    plt.savefig('CSM-train-loss.eps')
