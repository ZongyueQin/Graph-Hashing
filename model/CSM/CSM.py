import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from CSM_config import CSM_FLAGS
from model_regression_df import SiameseRegressionModel_DF
from CSM_train import run_tf
from utils_siamese import convert_msec_to_sec_str
from time import time
import numpy as np

class CSM:
    def __init__(self, data_fetcher):
        
        self.model = SiameseRegressionModel_DF(data_fetcher.node_feat_encoder.input_dim(), 
                                               data_fetcher)
        self.data_fetcher = data_fetcher
        
    
    def run_pairs_for_val_test(self, row_graphs, col_graphs, saver,
                           sess, val_or_test, care_about_loss=True):
        m = len(row_graphs)
        n = len(col_graphs)
        sim_dist_mat = np.zeros((m, n))
        time_list = []
        loss_list = []
        print_count = 0
        flush = True
        for i in range(m):
            for j in range(n):
                g1 = row_graphs[i]
                g2 = col_graphs[j]
                if care_about_loss:
                    ged = self.data_fetcher.getLabelForPair(g1,g2)
                    normalized_ged = ged * 2 / (len(g1.nxgraph.nodes())+len(g2.nxgraph.nodes()))
                    true_sim_dist = np.exp(-normalized_ged)
                    if true_sim_dist is None:
                        continue
                else:
                    true_sim_dist = 0  # only used for loss
                feed_dict = self.model.get_feed_dict_for_val_test(g1, g2, true_sim_dist, False)
                (loss_i_j, dist_sim_i_j), test_time = run_tf(
                        feed_dict, self.model, saver, sess, val_or_test)
                if flush:
                    (loss_i_j, dist_sim_i_j), test_time = run_tf(
                            feed_dict, self.model, saver, sess, val_or_test)
                    flush = False
            
                test_time *= 1000
                if val_or_test == 'test' and print_count < 100:
                    print('{},{},{:.2f}mec,{:.4f},{:.4f}'.format(
                            i, j, test_time, dist_sim_i_j, true_sim_dist))
                    print_count += 1
                sim_dist_mat[i][j] = dist_sim_i_j
                loss_list.append(loss_i_j)
                time_list.append(test_time)
        return sim_dist_mat, loss_list, time_list

    def predict(self, sess, saver, row_graphs, col_graphs):
        m = len(row_graphs)
        n = len(col_graphs)
        sim_dist_mat = np.zeros((m, n))
        flush = True
        for i in range(m):
            for j in range(n):
                g1 = row_graphs[i]
                g2 = col_graphs[j]
                true_sim_dist = 0
                feed_dict = self.model.get_feed_dict_for_val_test(g1, g2, true_sim_dist, False)
                (loss_i_j, dist_sim_i_j), test_time = run_tf(
                        feed_dict, self.model, saver, sess, 'test')
                if flush:
                    (loss_i_j, dist_sim_i_j), test_time = run_tf(
                            feed_dict, self.model, saver, sess, 'test')
                    flush = False
            
                sim_dist_mat[i][j] = dist_sim_i_j
                
        return sim_dist_mat
        
    
    def train(self, sess, saver):
        #train_costs, train_times, ss, val_results_dict = [], [], [], OrderedDict()
        train_costs, train_times, ss = [], [], []
        print('Optimization Started!')
        for iteration in range(1, CSM_FLAGS.csm_iters+1):
            #iteration += 1
            #need_gc, tvt = get_train_tvt(iter)
            need_gc = False
            tvt = 'train'
            # Train.
            start_time = time()
            feed_dict = self.model.get_feed_dict_for_train(need_gc)
            print('after get feed dict, %f s'%(time()-start_time))
            r, train_time = run_tf(
                    feed_dict, self.model, saver, sess, tvt, iter=iteration)
            train_s = ''
            if need_gc:
                train_cost, train_acc = r
                train_s = ' train_acc={:.5f}'.format(train_acc)
            else:
                train_cost = r
            train_costs.append(train_cost)
            train_times.append(train_time)
            """
        # Validate.
        val_s = ''
        if need_val(iter):
            t = time()
            val_results, val_s = val(data_val_test, eval, model, saver, sess,
                                     iter, need_gc)
            val_time = time() - t
            val_s += ' val_time={}'.format(convert_msec_to_sec_str(val_time))
            val_results_dict[iter] = val_results
            model.save(sess, saver, iter)
            """
            s = 'Iter:{:04n} {} train_loss={:.5f}{} time={}'.format(
                    iteration, tvt, train_cost, train_s, convert_msec_to_sec_str(train_time))
            print(s)
            ss.append(s)
        print('Optimization Finished!')
        plt.plot(train_costs)
        plt.xlabel('iterations')
        plt.ylabel('training loss')
        plt.savefig('CSM-train-loss.eps')
#        saver.save_train_val_info(train_costs, train_times, ss)
        return train_costs, train_times

