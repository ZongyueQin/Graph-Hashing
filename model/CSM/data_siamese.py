#from data import Data
from CSM_config import CSM_FLAGS
#from coarsening import coarsen, perm_data
from utils_siamese import get_coarsen_level, is_transductive
from CSM_utils import load_data, exec_turnoff_print
from node_ordering import node_ordering
#from random_walk_generator import generate_random_walks
#from supersource_generator import generate_supersource
#from super_large_dataset_handler import gen_data
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder
import networkx as nx
import numpy as np

exec_turnoff_print()


class ModelGraph(object):
    """Defines all relevant graph properties required for training a Siamese model.

    This is a data model representation of a graph
    for use during the training stage.
    Each ModelGraph has precomputed parameters
    (Laplacian, inputs, adj matrix, etc) as needed by the
    network during training.
    """

    def __init__(self, nxgraph, node_feat_encoder, graph_label_encoder=None):
        #self.glabel_position = graph_label_encoder.encode(nxgraph)
        #self.glabel_position = 0
        # Check flag compatibility.
        self._error_if_incompatible_flags()

        self.nxgraph = nxgraph
        last_order = []  # Overriden by supersource flag if used.

        # Overrides nxgraph to DiGraph if supersource needed.
        # Also edits the node_feat_encoder
        # because we add a supersource type for labeled graphs,
        # which means that the encoder
        # needs to be aware of a new one-hot encoding.
        if CSM_FLAGS.csm_supersource:
            nxgraph, supersource_id, supersource_label = generate_supersource(
                nxgraph, CSM_FLAGS.csm_node_feat_name)
            # If we have a labeled graph,
            # we need to add the supersource's new label to the encoder
            # so it properly one-hot encodes it.
            if CSM_FLAGS.csm_node_feat_name:
                node_feat_encoder.add_new_feature(supersource_label)
            self.nxgraph = nxgraph
            last_order = [supersource_id]

        # Generates random walks with parameters determined by the flags.
        # Walks are defined
        # by the ground truth node ids, so they do not depend on ordering,
        # but if a supersource
        # node is used it should be generated before the random walk.
        if CSM_FLAGS.csm_random_walk:
            if CSM_FLAGS.csm_supersource and type(nxgraph) != nx.DiGraph:
                raise RuntimeError(
                    'The input graph must be a DiGraph '
                    'in order to use random walks with '
                    'a supersource node so it is not used as a shortcut')
            params = CSM_FLAGS.csm_random_walk.split('_')
            num_walks = int(params[0])
            walk_length = int(params[1])
            self.random_walk_data = generate_random_walks(nxgraph, num_walks,
                                                          walk_length)

        # Encode features.
        dense_node_inputs = node_feat_encoder.encode(nxgraph)

        # Determine ordering and reorder the dense inputs
        # based on the desired ordering.
        if CSM_FLAGS.csm_ordering:
            if CSM_FLAGS.csm_ordering == 'bfs':
                self.order, self.mapping = node_ordering(
                    nxgraph, 'bfs', CSM_FLAGS.csm_node_feat_name, last_order)

            elif CSM_FLAGS.csm_ordering == 'degree':
                self.order, self.mapping = node_ordering(
                    nxgraph, 'degree', CSM_FLAGS.csm_node_feat_name, last_order)
            else:
                raise RuntimeError('Unknown ordering mode {}'.format(self.order))
            assert (len(self.order) == len(nxgraph.nodes()))
            # Apply the ordering.
            dense_node_inputs = dense_node_inputs[self.order, :]

        # Save matrix properties after reordering the nodes.
        self.sparse_node_inputs = self._preprocess_inputs(
            sp.csr_matrix(dense_node_inputs))
        # Only one laplacian.
        self.num_laplacians = 1
        if nxgraph.number_of_nodes() < 500000:
            adj = nx.to_numpy_matrix(nxgraph)
        else:
            adj = nx.to_scipy_sparse_matrix(nxgraph)
        # Fix ordering for adj.
        if CSM_FLAGS.csm_ordering:
            # Reorders the adj matrix using the order provided earlier.
            adj = adj[np.ix_(self.order, self.order)]

        # Special handling for coarsening because it is
        # incompatible with other flags.
        if CSM_FLAGS.csm_coarsening:
            self._coarsen(dense_node_inputs, adj)
        else:
            self.laplacians = [self._preprocess_adj(adj)]

    def _error_if_incompatible_flags(self):
        """Check flags and error for unhandled flag combinations.
            CSM_FLAGS.csm_coarsening
            CSM_FLAGS.csm_ordering
            CSM_FLAGS.csm_supersource
            CSM_FLAGS.csm_random_walk
        """
        if CSM_FLAGS.csm_coarsening:
            if CSM_FLAGS.csm_ordering or CSM_FLAGS.csm_supersource or CSM_FLAGS.csm_random_walk:
                raise RuntimeError(
                    'Cannot use coarsening with any of the following: ordering, '
                    'supersource, random_walk')
        else:
            if CSM_FLAGS.csm_supersource and CSM_FLAGS.csm_train_fake_from:
                raise RuntimeError(
                    'Cannot use supersource with fake generation right now because'
                    ' fake_generation doesnt support digraphs and labeled graphs '
                    'will break since the supersource_type could be duplicated')

    def get_nxgraph(self):
        return self.nxgraph

    def get_node_inputs(self):
        if CSM_FLAGS.csm_coarsening:
            return self.sparse_permuted_padded_dense_node_inputs
        else:
            return self.sparse_node_inputs

    def get_node_inputs_num_nonzero(self):
        return self.get_node_inputs()[1].shape

    def get_laplacians(self, gcn_id):
        if CSM_FLAGS.csm_coarsening:
            return self.coarsened_laplacians[gcn_id]
        else:
            return self.laplacians

    def _coarsen(self, dense_node_inputs, adj):
        assert ('metis_' in CSM_FLAGS.csm_coarsening)
        self.num_level = get_coarsen_level()
        assert (self.num_level >= 1)
        graphs, perm = coarsen(sp.csr_matrix(adj), levels=self.num_level,
                               self_connections=False)
        permuted_padded_dense_node_inputs = perm_data(
            dense_node_inputs.T, perm).T
        self.sparse_permuted_padded_dense_node_inputs = self._preprocess_inputs(
            sp.csr_matrix(permuted_padded_dense_node_inputs))
        self.coarsened_laplacians = []
        for g in graphs:
            self.coarsened_laplacians.append([self._preprocess_adj(g.todense())])
        assert (len(self.coarsened_laplacians) == self.num_laplacians * self.num_level + 1)

    def _preprocess_inputs(self, inputs):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(inputs.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        inputs = r_mat_inv.dot(inputs)
        return self._sparse_to_tuple(inputs)

    def _preprocess_adj(self, adj):
        """Preprocessing of adjacency matrix and conversion to tuple representation."""
        adj_normalized = self._normalize_adj(adj + sp.eye(adj.shape[0]))
        self._edge_list_incidence_mat(adj)
        return self._sparse_to_tuple(adj_normalized)

    def _edge_list_incidence_mat(self, adj):
        tmp_G = nx.from_numpy_matrix(adj)
        edge_list = tmp_G.edges()
        edge_list_full = []
        for (i, j) in edge_list:
            edge_list_full.append((i, j))
            if i != j:
                edge_list_full.append((j, i))
        self.edge_index = self._sparse_to_tuple(sp.csr_matrix(edge_list_full))
        incidence_mat = self._our_incidence_mat(tmp_G, edgelist=edge_list_full)
        self.incidence_mat = self._sparse_to_tuple(incidence_mat)

    def _our_incidence_mat(self, G, edgelist):
        nodelist = G.nodes()
        A = sp.lil_matrix((len(nodelist), len(edgelist)))
        node_index = dict((node, i) for i, node in enumerate(nodelist))
        for ei, e in enumerate(edgelist):
            (u, v) = e[:2]
            if u == v: continue  # self loops give zero column
            ui = node_index[u]
            # vi = node_index[v]
            A[ui, ei] = 1
        return A.asformat('csc')

    def _normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def _sparse_to_tuple(self, sparse_mx):
        """Convert sparse matrix to tuple representation."""

        def to_tuple(mx):
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
            return coords, values, shape

        if isinstance(sparse_mx, list):
            for i in range(len(sparse_mx)):
                sparse_mx[i] = to_tuple(sparse_mx[i])
        else:
            sparse_mx = to_tuple(sparse_mx)

        return sparse_mx


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
        self.feat_idx_dic = {feat: idx for idx, feat in enumerate(inputs_set)}
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
        self.input_dim_ = int(CSM_FLAGS.csm_node_feat_encoder.split('_')[1])
        self.const = float(2.0)
        assert (node_feat_name is None)

    def encode(self, g):
        rtn = np.full((g.number_of_nodes(), self.input_dim_), self.const)
        return rtn

    def input_dim(self):
        return self.input_dim_


class GraphLabelOneHotEncoder(object):
    def __init__(self, gs):
        self.glabel_map = {}
        for g in gs:
            glabel = g.graph['glabel']
            if glabel not in self.glabel_map:
                self.glabel_map[glabel] = len(self.glabel_map)

    def encode(self, g):
        return self.glabel_map[g.graph['glabel']]
