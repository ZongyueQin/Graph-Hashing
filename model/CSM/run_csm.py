#!/usr/bin/python3
from __future__ import division
from __future__ import print_function
import sys

import os
cur_folder = os.path.dirname(os.path.abspath(__file__))
#sys.path.insert(0, '{}/../'.format(cur_folder))

from CSMDataFetcher import CSMDataFetcher
from CSM_config import CSM_FLAGS
from CSM_train import train_val_loop, test
from utils_siamese import get_model_info_as_str, \
    check_flags, extract_config_code, convert_long_time_to_str
from CSM_utils import slack_notify, get_ts
#from data_siamese import SiameseModelData
#from dist_sim_calculator import CSM_DistSimCalculator
#from models_factory import create_model
from saver import Saver
#from eval import Eval
import tensorflow as tf
from time import time
import os, traceback
from CSMDataFetcher import CSMDataFetcher
from CSM import CSM
import numpy as np

def readGroundTruth(f):
    ged_cnt = {}
    ground_truth = {}
    for line in f.readlines():
        g, q, d = line.split(' ')
        g = int(g)
        q = int(q)
        d = int(d)

        if q not in ground_truth.keys():
            ground_truth[q] = []
        ground_truth[q].append((g,d))

        ged_cnt.setdefault(d,0)
        ged_cnt[d] = ged_cnt[d] + 1

    return ground_truth, ged_cnt

def computeTestMSEWithGroundTruth(sess, saver, csm,
                                  data_fetcher, 
                                  ground_truth,
                                  per_test_cnt = 10):
    total_query_num = data_fetcher.get_test_graphs_num()
    MSE_test = 0
    
    for i in range(0, total_query_num):
        g1 = data_fetcher.test_graphs[i]
        g_list1 = [g1]

        q = data_fetcher.get_test_graph_gid(i)

        ground_truth[q] = sorted(ground_truth[q],
                                key=lambda x: x[1]*10000000 + x[0])

            # Second batch, the closest graphs
        g_list2 = []
        GT = []

        thres_scores = []        
        for pair in ground_truth[q][0:per_test_cnt]:
            true_ged = pair[1]
            gid = pair[0]
            g2 = data_fetcher.getGraphByGid(gid)
            g_list2.append(g2)
            normalzied_ged = true_ged*2/(len(g1.nxgraph.nodes())+len(g2.nxgraph.nodes()))
            thres = CSM_FLAGS.csm_GED_threshold*2/(len(g1.nxgraph.nodes())+len(g2.nxgraph.nodes()))
            thres = np.exp(-thres)
            score = np.exp(-normalzied_ged)
            GT.append(score)
            thres_scores.append(thres)
            
        pred = csm.predict(sess, saver, g_list1, g_list2)
        GT = np.array(GT)
        pred = np.squeeze(pred)
        thres_scores = np.array(thres_scores)
        GT[GT == thres_scores] = 0
        MSE_test = MSE_test + (np.mean((GT-pred)**2))/total_query_num

    print('MSE for test = {:f}'.format(MSE_test))

def main():
    t = time()
    conf_code = extract_config_code()
    check_flags()
    print(get_model_info_as_str())

    data_fetcher = CSMDataFetcher(CSM_FLAGS.csm_dataset, True)
    model = CSM(data_fetcher)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CSM_FLAGS.csm_gpu)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = Saver(sess)
    tf_saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    print('finish initialization')

    try:
        ground_truth_path = os.path.join('..', '..','data',
                                 CSM_FLAGS.csm_dataset,
                                 'test',
                                 CSM_FLAGS.csm_ground_truth_file)
        try:
            f = open(ground_truth_path, 'r')
            ground_truth, ged_cnt = readGroundTruth(f)
            has_GT = True
    #        computeTestMSEWithGroundTruth(sess, saver, model, 
    #                                      data_fetcher, ground_truth)
        except IOError:
            print('Groundtruth file doesn\'t exist, ignore top-k and range query')
            has_GT = False

        train_costs, train_times = model.train(sess, saver)
        save_path = "SavedModel/CSM_" + CSM_FLAGS.csm_dataset + ".ckpt"
        save_path = tf_saver.save(sess, save_path)
        print("Model saved in {}".format(save_path))
        #best_iter, test_results = \
        #    test(data_val_test, eval, model, saver, sess, val_results_dict)
        
        
        if has_GT:
            computeTestMSEWithGroundTruth(sess, saver, model, 
                                          data_fetcher, ground_truth)
        saver.save_conf_code(conf_code)
        overall_time = convert_long_time_to_str(time() - t)
        print(overall_time, saver.get_log_dir())
        saver.save_overall_time(overall_time)
    except:
        traceback.print_exc()
    #    slack_notify('model train {} error'.format(get_ts()))
    else:
    #    slack_notify('model train {} complete'.format(get_ts()))
        return train_costs, train_times


if __name__ == '__main__':
    main()
