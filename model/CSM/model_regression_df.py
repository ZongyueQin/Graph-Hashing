from CSM_config import CSM_FLAGS
from CSM_model import CSM_Model
#from samplers import SelfShuffleList
from utils_siamese import get_coarsen_level, get_flags, is_transductive
from CSM_inits import CSM_glorot
#from dist_sim_kernel import create_ds_kernel
#from dist_sim_calculator import get_gs_ds_mat
import numpy as np
import tensorflow as tf
#from data_siamese import GeneratedGraphCollection


class SiameseRegressionModel_DF(CSM_Model):
    def __init__(self, input_dim, data_fetcher):
        self.input_dim = input_dim
        self.data_fetcher = data_fetcher
        print('original_input_dim', self.input_dim)
#        if is_transductive():
#            self._create_transductive_gembs_placeholders(data,
#                                                         CSM_FLAGS.csm_batch_size, CSM_FLAGS.csm_batch_size)
#        else:
        self._create_basic_placeholders(CSM_FLAGS.csm_batch_size, CSM_FLAGS.csm_batch_size,
                                        level=get_coarsen_level())
        self.train_y_true = tf.placeholder(
            tf.float32, shape=(CSM_FLAGS.csm_batch_size, 1))
        self.val_test_y_true = tf.placeholder(
            tf.float32, shape=(1, 1))
        # Build the model.
        super(SiameseRegressionModel_DF, self).__init__()
        #self.ds_kernel = create_ds_kernel(
        #    CSM_FLAGS.csm_ds_kernel, get_flags('yeta'), get_flags('scale'))
        #self.train_triples = self._load_train_triples(data, dist_sim_calculator)

    def pred_sim_without_act(self):
        return self.val_test_pred_score

    def apply_final_act_np(self, score):
        return score

    def get_feed_dict_for_train(self, need_gc):
        rtn = {}
        pairs = []
        y_true = np.zeros((CSM_FLAGS.csm_batch_size, 1))
        glabels = None
        if need_gc:
            glabels = np.zeros((CSM_FLAGS.csm_batch_size * 2, CSM_FLAGS.csm_num_glabels))
        """
        for i in range(CSM_FLAGS.csm_batch_size):
            g1, g2, true_sim_dist = self._sample_train_pair()
            pairs.append((g1, g2))
            y_true[i] = true_sim_dist
            if need_gc:
                glabels[i][g1.glabel_position] = 1
                glabels[CSM_FLAGS.csm_batch_size + i][g2.glabel_position] = 1
        """
        pair_list, true_sim_dist_list =\
            self.data_fetcher.samplePairListWithLabel(CSM_FLAGS.csm_real_pair_bs,
                                                      CSM_FLAGS.csm_batch_size-CSM_FLAGS.csm_real_pair_bs)
        for i, (pair, true_sim_dist) in enumerate(zip(pair_list, true_sim_dist_list)):
#            print(i)
#            print(true_sim_dist)
#            print(pair)
            pairs.append(pair)
            y_true[i] = true_sim_dist
            
        rtn[self.train_y_true] = y_true
        rtn[self.dropout] = CSM_FLAGS.csm_dropout
        if need_gc:
            rtn[self.train_true_glabels] = glabels
        return self._supply_laplacians_etc_to_feed_dict(rtn, pairs, 'train')

    def get_feed_dict_for_val_test(self, g1, g2, true_sim_dist, need_gc):
        rtn = {}
        pairs = [(g1, g2)]
        y_true = np.zeros((1, 1))
        y_true[0] = true_sim_dist
        rtn[self.val_test_y_true] = y_true
        if need_gc:
            glabels = np.zeros((1 * 2, CSM_FLAGS.csm_num_glabels))
            glabels[0][g1.glabel_position] = 1
            glabels[1][g2.glabel_position] = 1
            rtn[self.val_test_true_glabels] = glabels
        return self._supply_laplacians_etc_to_feed_dict(rtn, pairs, 'val_test')

    def get_true_dist_sim(self, i, j, true_result):
        assert (true_result.dist_or_sim() in ['sim', 'dist'])  # A* GED or MCS
        _, ds = true_result.dist_sim(i, j, CSM_FLAGS.csm_ds_norm)
        if CSM_FLAGS.csm_supply_sim_dist == 'sim' and true_result.dist_or_sim() == 'dist':  # e.g. A*
            assert (CSM_FLAGS.csm_ds_metric == 'ged')
            rtn = self.ds_kernel.dist_to_sim_np(ds)
        elif CSM_FLAGS.csm_supply_sim_dist == 'dist' and true_result.dist_or_sim() == 'sim':  # e.g. MCS
            assert (CSM_FLAGS.csm_ds_metric == 'mcs')
            raise NotImplementedError()  # TODO
        else:
            rtn = ds
        return rtn

    def _get_eval_metrics_for_val_without_gc(self):
        return ['loss', 'mse']

    def get_eval_metrics_for_test(self):
        # In reality, may be less than these,
        # e.g. when the model does not produce graph-level embeddings.
        return ['graph_classification',
                'mse', 'dev', 'prec@k',
                'mrr', 'kendalls_tau', 'spearmans_rho', 'time',
                # 'train_pair_ged_vis',
                'emb_vis_gradual',
                'draw_heat_hist',
                'ranking'
                # 'attention'
                ]

    def _get_determining_result_for_val(self):
        if CSM_FLAGS.csm_need_gc:
            return 'val_acc'
        else:
            return 'val_mse'

    def _val_need_max(self):
        if CSM_FLAGS.csm_dataset_super_large:
            return True  # max val_iso_top_<>_<>
        else:
            if CSM_FLAGS.csm_need_gc:
                return True  # max val_acc
            else:
                return False  # min val mse

    def _get_ins(self, layer, tvt):
        if is_transductive():
            return self._get_ins_for_transductive_model(layer, tvt)
        else:
            ins = []
            assert (layer.__class__.__name__ == 'CSM_GraphConvolution' or
                    layer.__class__.__name__ == 'CSM_GraphConvolutionAttention')
            for features in (self._get_plhdr('features_1', tvt) +
                             self._get_plhdr('features_2', tvt)):
                ins.append(features)
        return ins

    def _get_ins_for_transductive_model(self, layer, tvt):
        assert (layer.__class__.__name__ == 'CSM_Dist' or
                layer.__class__.__name__ == 'CSM_Dot')
        ids_1 = self._get_plhdr('gemb_lookup_ids_1', tvt)
        ids_2 = self._get_plhdr('gemb_lookup_ids_2', tvt)
        gembs_1 = tf.nn.embedding_lookup(self.all_gembs, ids_1)
        gembs_2 = tf.nn.embedding_lookup(self.all_gembs, ids_2)
        if tvt == 'train':
            self.graph_embeddings_train = self._stack_concat([gembs_1, gembs_2])
        elif tvt == 'val_test':
            self.graph_embeddings_val_test = self._stack_concat([gembs_1, gembs_2])  # for train.py to collect
        self.gemb_dim = CSM_FLAGS.csm_gemb_dim
        return [gembs_1, gembs_2]

    def _proc_ins_for_merging_layer(self, ins, _):
        assert (len(ins) % 2 == 0)
        proc_ins = []
        i = 0
        j = len(ins) // 2
        for _ in range(len(ins) // 2):
            proc_ins.append([ins[i], ins[j]])
            i += 1
            j += 1
        return proc_ins

    def _val_test_pred_score(self):
        self.val_test_output = self._stack_concat(self.val_test_output)
        assert self.val_test_output.get_shape().as_list() == [1, 1], \
            self.val_test_output.get_shape()
        return tf.squeeze(self.val_test_output)

    def _task_loss(self, tvt):
        rtn = {}
        for loss_lambda in self._get_loss_lambdas_flags():
            if loss_lambda == 'csm_lambda_mse_loss':
                # If the model predicts sim, ground-truth should be sim.
                # If the model predicts dist, ground-truth should be dist.
                assert (CSM_FLAGS.csm_supply_sim_dist == CSM_FLAGS.csm_pred_sim_dist)
                y_pred, y_true = self._get_y_pred_y_true(tvt)
                loss = tf.nn.l2_loss(y_true - y_pred)
                rtn['mse_loss'] = CSM_FLAGS.csm_lambda_mse_loss * loss
            elif loss_lambda == 'csm_lambda_weighted_dist_loss':
                # The model must predict dist and ground-truth must be sim.
                assert (CSM_FLAGS.csm_pred_sim_dist == 'dist')
                assert (CSM_FLAGS.csm_supply_sim_dist == 'sim')
                assert (CSM_FLAGS.csm_ds_metric == 'ged')
                y_pred, y_true_sim = self._get_y_pred_y_true(tvt)
                assert (y_true_sim.get_shape() == y_pred.get_shape())
                loss = tf.reduce_mean(y_true_sim * y_pred)
                rtn['weighted_dist_loss'] = CSM_FLAGS.csm_lambda_weighted_dist_loss * loss
            elif loss_lambda == 'csm_lambda_triv_avoid_loss':
                phi, bs_times_2, D = self._get_graph_embs_as_one_mat(tvt)
                loss = tf.reduce_mean(phi ** 2)  # tf.nn.l2_loss does not do sqrt
                rtn['triv_avoid_loss'] = CSM_FLAGS.csm_lambda_triv_avoid_loss * loss
            elif loss_lambda == 'csm_lambda_diversity_loss':
                phi, bs_times_2, D = self._get_graph_embs_as_one_mat(tvt)
                # phi /= bs_times_2
                phi_T_phi = tf.matmul(tf.transpose(phi), phi)
                should = tf.eye(D)
                assert (phi_T_phi.get_shape() == should.get_shape())
                loss = tf.reduce_mean((phi_T_phi - should) ** 2)  # tf.nn.l2_loss does not do sqrt
                rtn['diversity_loss'] = CSM_FLAGS.csm_lambda_diversity_loss * loss
            elif loss_lambda == 'csm_lambda_gc_loss':
                pass  # handled by model
            else:
                raise RuntimeError('Unknown loss lambda flag {}'.format(loss_lambda))
        return rtn

    def _get_y_pred_y_true(self, tvt):
        if tvt == 'train':
            y_pred = self._stack_concat(self.train_outputs)
            y_true = self.train_y_true
        else:
            y_pred = self._stack_concat(self.val_test_output)
            y_true = self.val_test_y_true
        assert (y_true.get_shape() == y_pred.get_shape())
        return y_pred, y_true

    def _get_graph_embs_as_one_mat(self, tvt):
        phi = self._get_graph_embds(tvt, True)
        s = phi.get_shape().as_list()
        bs_times_2, D = s[0], s[1]
        bs = CSM_FLAGS.csm_batch_size if tvt == 'train' else 1
        assert (bs_times_2 == bs * 2)
        return phi, bs_times_2, D

    def _get_graph_embds(self, tvt, need_concat):
        if tvt == 'train':
            rtn = self.graph_embeddings_train
        else:
            assert (tvt == 'val')
            rtn = self.graph_embeddings_val_test
        if need_concat:
            return self._stack_concat(rtn)
        else:
            return rtn

    def _create_basic_placeholders(self, num1, num2, level):
        self.max_node = tf.placeholder(tf.int32, shape=())
        self.laplacians_1 = \
            [[[tf.sparse_placeholder(tf.float32)]
              for _ in range(level)] for _ in range(num1)]
        self.laplacians_2 = \
            [[[tf.sparse_placeholder(tf.float32)]
              for _ in range(level)] for _ in range(num2)]
        self.features_1 = \
            [tf.sparse_placeholder(tf.float32) for _ in range(num1)]
        self.features_2 = \
            [tf.sparse_placeholder(tf.float32) for _ in range(num2)]
        self.num_nonzero_1 = \
            [tf.placeholder(tf.int32) for _ in range(num1)]
        self.num_nonzero_2 = \
            [tf.placeholder(tf.int32) for _ in range(num2)]
        self.edge_index_1 = [tf.sparse_placeholder(tf.int32, shape=(None, 2)) for _ in range(num1)]
        self.edge_index_2 = [tf.sparse_placeholder(tf.int32, shape=(None, 2)) for _ in range(num2)]
        self.incidence_mat_1 = [tf.sparse_placeholder(tf.int32) for _ in range(num1)]
        self.incidence_mat_2 = [tf.sparse_placeholder(tf.int32) for _ in range(num2)]
        self.dropout = tf.placeholder_with_default(0., shape=())
        self.val_test_laplacians_1 = [[[tf.sparse_placeholder(tf.float32)] for _ in range(level)]]
        self.val_test_laplacians_2 = [[[tf.sparse_placeholder(tf.float32)] for _ in range(level)]]
        self.val_test_features_1 = [tf.sparse_placeholder(tf.float32)]
        self.val_test_features_2 = [tf.sparse_placeholder(tf.float32)]
        self.val_test_num_nonzero_1 = [tf.placeholder(tf.int32)]
        self.val_test_num_nonzero_2 = [tf.placeholder(tf.int32)]
        self.val_test_edge_index_1 = [tf.sparse_placeholder(tf.int32, shape=(None, 2))]
        self.val_test_edge_index_2 = [tf.sparse_placeholder(tf.int32, shape=(None, 2))]
        self.val_test_incidence_mat_1 = [tf.sparse_placeholder(tf.int32)]
        self.val_test_incidence_mat_2 = [tf.sparse_placeholder(tf.int32)]


    def _create_transductive_gembs_placeholders(self, data, num1, num2):
        # Create the dataset-level graph-level embeddings table for later look up.
        self.all_gembs = CSM_glorot([data.num_graphs(), CSM_FLAGS.csm_gemb_dim],
                                name='all_graph_embeddings')
        self.gemb_lookup_ids_1 = tf.placeholder(tf.int32, shape=(num1))
        self.gemb_lookup_ids_2 = tf.placeholder(tf.int32, shape=(num2))
        self.val_test_gemb_lookup_ids_1 = tf.placeholder(tf.int32, shape=(1))  # 1 graph pair per val/test
        self.val_test_gemb_lookup_ids_2 = tf.placeholder(tf.int32, shape=(1))
        self.dropout = tf.placeholder_with_default(0., shape=())

    """
    def _load_train_triples(self, data, ds_calc):
        triples = self._load_create_fake_pairs_if_needed(data)
        # if CSM_FLAGS.csm_dataset_super_large:  # no real pairs and no need to load TODO: may have some real pairs
        #     assert (CSM_FLAGS.csm_train_fake_from is not None and CSM_FLAGS.csm_train_fake_gen)  # must have fake
            # return SelfShuffleList(triples)
        triples = self._load_real_pairs(data.train_gs, data.train_gs,
                                        'train', 'train', triples, ds_calc)
        if is_transductive():
            # Load more pairs to better train the model,
            # since it directly optimizes over the embeddings.
            triples = self._load_real_pairs(data.val_gs, data.train_gs,
                                            'val', 'train', triples, ds_calc)
            triples = self._load_real_pairs(data.val_gs, data.val_gs,
                                            'val', 'val', triples, ds_calc)
            triples = self._load_real_pairs(data.test_gs, data.train_gs,
                                            'test', 'train', triples, ds_calc)
            triples = self._load_real_pairs(data.test_gs, data.val_gs,
                                            'test', 'val', triples, ds_calc)
        return SelfShuffleList(triples)
     
    def _load_real_pairs(self, gs1, gs2, tvt1, tvt2, triples, ds_calc):
        rtn = []
        nx_gs1 = self._get_nxgraph_list(gs1)
        nx_gs2 = self._get_nxgraph_list(gs2)
        ds_mat = get_gs_ds_mat(
            nx_gs1, nx_gs2, ds_calc, tvt1, tvt2,
            CSM_FLAGS.csm_dataset_train, CSM_FLAGS.csm_ds_metric, CSM_FLAGS.csm_ds_algo, CSM_FLAGS.csm_ds_norm,
            dec_gsize=CSM_FLAGS.csm_supersource, return_neg1=True)
        m, n = ds_mat.shape
        # assert (m == n)
        # ds_mat_idx = np.argsort(ds_mat, axis=1)
        valid_count, skip_count = 0, 0
        for i in range(m):
            g1 = gs1[i]
            for j in range(n):
                col = j
                g2, ds = gs2[col], ds_mat[i][col]
                if ds < 0:
                    skip_count += 1
                    if tvt1 == 'test':
                        print('@@@', i, j, g1.nxgraph.graph['gid'], g2.nxgraph.graph['gid'], len(gs1), len(gs2))
                        exit()
                    continue
                valid_count += 1
                if CSM_FLAGS.csm_ds_metric == 'mcs' or CSM_FLAGS.csm_ds_metric == 'glet':
                    assert (CSM_FLAGS.csm_supply_sim_dist == 'sim')  # TODO: transform mcs to dist if needed
                    need = ds
                else:
                    assert (CSM_FLAGS.csm_ds_metric == 'ged')
                    if CSM_FLAGS.csm_supply_sim_dist == 'sim':  # supply for train --> consistent with supply_sim_dist
                        need = self.ds_kernel.dist_to_sim_np(ds)
                    else:
                        need = ds
                rtn.append((g1, g2, need))
        if CSM_FLAGS.csm_train_real_percent < 1:
            sp = int(len(rtn) * CSM_FLAGS.csm_train_real_percent)
            rtn_new = rtn[0:sp]
            print('Only take {} from {} due to {} percent'.format(
                len(rtn_new), len(rtn), CSM_FLAGS.csm_train_real_percent))
            rtn = rtn_new
        triples += rtn
        print('{} valid pairs; {} pairs with dist or sim < 0; {} total triples'.format(
            valid_count, skip_count, len(triples)))
        return triples

    def _load_create_fake_pairs_if_needed(self, data):
        if CSM_FLAGS.csm_train_fake_from:
            gc = GeneratedGraphCollection(
                CSM_FLAGS.csm_train_fake_from, CSM_FLAGS.csm_train_fake_gen, data)
            return gc.get_train_pairs(self.ds_kernel)
        else:
            return []
    """
    def _sample_train_pair(self):
        x, y, true_sim_dist = self.data_fetcher.getNextTrainPair()
        return x, y, true_sim_dist

    def _supply_laplacians_etc_to_feed_dict(self, feed_dict, pairs, tvt):
        if is_transductive():
            gemb_lookup_ids_1 = []
            gemb_lookup_ids_2 = []
            for (g1, g2) in pairs:
                gemb_lookup_ids_1.append(g1.global_id)
                gemb_lookup_ids_2.append(g2.global_id)
            feed_dict[self._get_plhdr('gemb_lookup_ids_1', tvt)] = \
                gemb_lookup_ids_1
            feed_dict[self._get_plhdr('gemb_lookup_ids_2', tvt)] = \
                gemb_lookup_ids_2
        else:
            cur_max_node = 1
        #    print(pairs)
            for i, (g1, g2) in enumerate(pairs):
         #       print(g1)
                if len(g1.nxgraph.nodes()) > cur_max_node:
                    cur_max_node = len(g1.nxgraph.nodes())
                if len(g2.nxgraph.nodes()) > cur_max_node:
                    cur_max_node = len(g2.nxgraph.nodes())
                    
                feed_dict[self._get_plhdr('features_1', tvt)[i]] = \
                    g1.get_node_inputs()
                feed_dict[self._get_plhdr('features_2', tvt)[i]] = \
                    g2.get_node_inputs()
                feed_dict[self._get_plhdr('num_nonzero_1', tvt)[i]] = \
                    g1.get_node_inputs_num_nonzero()
                feed_dict[self._get_plhdr('num_nonzero_2', tvt)[i]] = \
                    g2.get_node_inputs_num_nonzero()
                feed_dict[self._get_plhdr('edge_index_1', tvt)[i]] = \
                    g1.edge_index
                feed_dict[self._get_plhdr('edge_index_2', tvt)[i]] = \
                    g2.edge_index
                feed_dict[self._get_plhdr('incidence_mat_1', tvt)[i]] = \
                    g1.incidence_mat
                feed_dict[self._get_plhdr('incidence_mat_2', tvt)[i]] = \
                    g2.incidence_mat
                assert g1.incidence_mat is not None
                num_laplacians = 1
                for j in range(get_coarsen_level()):
                    for k in range(num_laplacians):
                        feed_dict[
                            self._get_plhdr('laplacians_1', tvt)[i][j][k]] = \
                            g1.get_laplacians(j)[k]
                        feed_dict[
                            self._get_plhdr('laplacians_2', tvt)[i][j][k]] = \
                            g2.get_laplacians(j)[k]
            feed_dict[self._get_plhdr('max_node', tvt)] = cur_max_node
        return feed_dict

    def _get_loss_lambdas_flags(self):
        rtn = []
        d = CSM_FLAGS.flag_values_dict()
        for k in d.keys():
            if 'lambda_' in k:
                flag_split = k.split('_')
                assert (flag_split[1] == 'lambda')
                assert (flag_split[-1] == 'loss')
                rtn.append(k)
        return rtn

    def _get_nxgraph_list(self, model_graphs):
        return [g.nxgraph for g in model_graphs]
