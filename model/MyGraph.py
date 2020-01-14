import scipy.sparse as sp
import networkx as nx
from config import FLAGS
from sklearn.preprocessing import OneHotEncoder

import numpy as np 

""" My wrapper object for networkx graph 

    Parts of the code refer to https://github.com/tkipf/gcn """

class MyGraph(object):
    def __init__(self, graph, max_label):

        self.ori_graph = graph
        assert(len(graph['nodes'])==graph['adj_mat'].shape[0])
        adj_mat = sp.csr_matrix(graph['adj_mat'])
#        adj_mat = nx.to_scipy_sparse_matrix(nxgraph)
        dense_node_inputs = self.encode(graph, max_label)
        #dense_node_inputs = node_feat_encoder.encode(nxgraph)
        # TODO probably add ordering
        self.sparse_node_inputs = self._preprocess_inputs(
                sp.csr_matrix(dense_node_inputs))
        self.laplacian = self._preprocess_adj(adj_mat)

    def encode(self, graph, max_label):
        n = len(graph['nodes'])
        if FLAGS.node_feat_encoder == 'onehot':
            feature = np.zeros((n, max_label+1))
            for i,l in enumerate(graph['nodes']):
                feature[i,l] = 1
        elif 'constant' in FLAGS.node_feat_encoder:
            input_dim_ = int(FLAGS.node_feat_encoder.split('_')[1])
            const = float(2.0)
            feature = np.full((n, input_dim_), const)
        else:
            raise RuntimeError('Unknown node_feat_encoder {}'.format(FLAGS.node_feat_encoder))
        return feature
        

    def _preprocess_inputs(self, inputs):

        """Row-normalize feature matrix and convert to tuple representation"""

        rowsum = np.array(inputs.sum(1))

        r_inv = np.power(rowsum, -1).flatten()

        r_inv[np.isinf(r_inv)] = 0.

        r_mat_inv = sp.diags(r_inv)

        inputs = r_mat_inv.dot(inputs)

        return inputs

#        return self._sparse_to_tuple(inputs)



    

    """ compute laplacian matrix, return a sparse matrix in form of tuple """

    def _preprocess_adj(self, adj_mat):

        adj_proc = sp.coo_matrix(adj_mat+sp.eye(adj_mat.shape[0]))

        if FLAGS.laplacian == 'gcn': # what's other options ?

            adj_proc = self._normalize_adj(adj_proc)

        

        return adj_proc

#        return self._sparse_to_tuple(adj_proc)

    

    """ Symmetrically normalize matrix """

    def _normalize_adj(self, adj):

        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()

        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.

        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()





class NodeFeatureEncoder(object):

    def encode(self, g):

        raise NotImplementedError()



    def input_dim(self):

        raise NotImplementedError()





class NodeFeatureOneHotEncoder(NodeFeatureEncoder):

    def __init__(self, gs, node_feat_name):

        self.node_feat_name = node_feat_name

        # Go through all the graphs in the entire dataset

        # and create a set of all possible

        # labels so we can one-hot encode them.

        inputs_set = set()

        for g in gs:

            inputs_set = inputs_set | set(self._node_feat_dic(g).values())

        self.feat_idx_dic = {feat: idx for idx, feat in enumerate(sorted(list(inputs_set)))}

        self._fit_onehotencoder()



    def _fit_onehotencoder(self):

        self.oe = OneHotEncoder().fit(

            np.array(list(self.feat_idx_dic.values())).reshape(-1, 1))



    def add_new_feature(self, feat_name):

        """Use this function if a new feature was added to the graph set."""

        # Add the new feature to the dictionary

        # as a unique feature and reinit the encoder.

        new_idx = len(self.feat_idx_dic)

        self.feat_idx_dic[feat_name] = new_idx

        self._fit_onehotencoder()



    def encode(self, g):

        node_feat_dic = self._node_feat_dic(g)

        temp = [self.feat_idx_dic[node_feat_dic[n]] for n in g.nodes()]

        return self.oe.transform(np.array(temp).reshape(-1, 1)).toarray()



    def input_dim(self):

        return self.oe.transform([[0]]).shape[1]



    def _node_feat_dic(self, g):

        return nx.get_node_attributes(g, self.node_feat_name)





class NodeFeatureConstantEncoder(NodeFeatureEncoder):

    def __init__(self, _, node_feat_name):

        self.input_dim_ = int(FLAGS.node_feat_encoder.split('_')[1])

        self.const = float(2.0)

        assert (node_feat_name is None)



    def encode(self, g):

        rtn = np.full((g.number_of_nodes(), self.input_dim_), self.const)

        return rtn



    def input_dim(self):

        return self.input_dim_
