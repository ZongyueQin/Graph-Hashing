from CSM_config import CSM_FLAGS
from utils_siamese import convert_msec_to_sec_str, get_model_info_as_str, \
    need_val, get_flags
#from data_siamese import GeneratedGraphCollection
from time import time
import numpy as np
from collections import OrderedDict, defaultdict


def train_val_loop(data, data_val_test, eval, model, saver, sess):
    train_costs, train_times, ss, val_results_dict = [], [], [], OrderedDict()
    print('Optimization Started!')
    for iter in range(CSM_FLAGS.csm_iters):
        iter += 1
        need_gc, tvt = get_train_tvt(iter)
        # Train.
        feed_dict = model.get_feed_dict_for_train(data, need_gc)
        r, train_time = run_tf(
            feed_dict, model, saver, sess, tvt, iter=iter)
        train_s = ''
        if need_gc:
            train_cost, train_acc = r
            train_s = ' train_acc={:.5f}'.format(train_acc)
        else:
            train_cost = r
        train_costs.append(train_cost)
        train_times.append(train_time)
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
        s = 'Iter:{:04n} {} train_loss={:.5f}{} time={} {}'.format(
            iter, tvt, train_cost, train_s, convert_msec_to_sec_str(train_time), val_s)
        print(s)
        ss.append(s)
    print('Optimization Finished!')
    saver.save_train_val_info(train_costs, train_times, ss, val_results_dict)
    return train_costs, train_times, val_results_dict


def val(data, eval, model, saver, sess, iter, need_gc):
    assert (not CSM_FLAGS.csm_dataset_super_large)
    val_gs, train_gs = eval.get_val_gs_as_tuple(data)
    val_results = OrderedDict()
    val_s = ''
    if need_gc:
        sim_dist_mat, loss_list, time_list = None, None, None
        total_loss_with_gc, gc_acc = run_gc(val_gs, model, saver, sess)
        val_s += 'val_loss_with_gc={:.5f} val_acc={:.5f}'.format(
            total_loss_with_gc, gc_acc)
        val_results['val_loss_with_gc'] = total_loss_with_gc
        val_results['val_acc'] = gc_acc
        # Collect all the graph-level embeddings for tsne plots.
        gs1, gs2 = eval.get_test_gs_as_tuple(data)
        _, graph_embs_mat, _ = collect_embeddings(
            gs1, gs2, model, saver, sess, gemb_only=True)
    elif eval.need_val:
        sim_dist_mat, loss_list, time_list = run_pairs_for_val_test(
            val_gs, train_gs, eval, model, saver, sess, 'val')
        graph_embs_mat = None
    val_results, val_s = eval.eval_for_val(val_results, val_s,
                                           sim_dist_mat, loss_list, time_list,
                                           graph_embs_mat,
                                           model.get_eval_metrics_for_val(need_gc),
                                           saver, iter)
    saver.log_val_results(val_results, iter)
    return val_results, val_s


def test(data, eval, model, saver, sess, val_results_dict):
    if CSM_FLAGS.csm_dataset_super_large:
        return test_for_super_large_dataset(data, eval, model, saver, sess,
                                            iter, 'test')
    best_iter = model.find_load_best_model(sess, saver, val_results_dict)
    saver.clean_up_saved_models(best_iter)
    gs1, gs2 = eval.get_test_gs_as_tuple(data)
    sim_dist_mat, loss_list, time_list = run_pairs_for_val_test(
        gs1, gs2, eval, model, saver, sess, 'test')
    node_embs_dict, graph_embs_mat, emb_time = collect_embeddings(
        gs1, gs2, model, saver, sess)
    attentions = collect_attentions(gs1, gs2, model, saver, sess)
    saver.save_test_info(
        sim_dist_mat, time_list, best_iter, node_embs_dict, graph_embs_mat,
        emb_time, attentions)
    print('Evaluating...')
    results = eval.eval_for_test(
        sim_dist_mat, model.get_eval_metrics_for_test(), saver,
        loss_list, time_list, node_embs_dict, graph_embs_mat, attentions,
        model, data)
    if CSM_FLAGS.csm_need_gc:
        total_loss_with_gc, gc_acc = run_gc(gs1, model, saver, sess)
        results.update({'total_test_loss_with_gc': total_loss_with_gc})
        results.update({'test_gc_acc': gc_acc})
    if not CSM_FLAGS.csm_plot_results:
        pretty_print_dict(results)
    print('Results generated with {} metrics; collecting embeddings'.format(
        len(results)))
    print(get_model_info_as_str())
    saver.save_test_result(results)
    return best_iter, results


def test_for_super_large_dataset(data, eval, model, saver, sess, iter, tvt):
    assert (tvt == 'test')
    test_slt_collecs = get_super_large_dataset_test_collections(data)
    all_results = OrderedDict()
    gs1, gs2 = eval.get_test_gs_as_tuple(data)
    _, graph_embs_mat, _ = collect_embeddings(
        gs1, gs2, model, saver, sess, gemb_only=True)
    results = eval.eval_for_test(
        None, model.get_eval_metrics_for_test(),
        saver,
        graph_embs_mat=graph_embs_mat)
    all_results.update({'base_results': results})
    for i, slt_collec in enumerate(test_slt_collecs):
        sim_dist_mat, time_list = run_super_large_dataset_test_collection(
            slt_collec, model, saver, sess, tvt)
        results = eval.eval_for_test(
            sim_dist_mat, model.get_eval_metrics_for_test(),
            saver, time_list=time_list, slt_collec=slt_collec)
        all_results.update({slt_collec.short_name: results})
    best_iter = iter  # no val --> best iter equal to the last iter
    saver.clean_up_saved_models(best_iter)
    saver.save_test_info(
        None, None, best_iter, None, graph_embs_mat,
        None, None)
    saver.save_test_result(all_results)
    print(get_model_info_as_str())
    return best_iter, all_results


def run_pairs_for_val_test(row_graphs, col_graphs, eval, model, saver,
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
                true_sim_dist = eval.get_true_dist_sim(i, j, val_or_test, model)
                if true_sim_dist is None:
                    continue
            else:
                true_sim_dist = 0  # only used for loss
            feed_dict = model.get_feed_dict_for_val_test(g1, g2, true_sim_dist, False)
            (loss_i_j, dist_sim_i_j), test_time = run_tf(
                feed_dict, model, saver, sess, val_or_test)
            if flush:
                (loss_i_j, dist_sim_i_j), test_time = run_tf(
                    feed_dict, model, saver, sess, val_or_test)
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


def run_tf(feed_dict, model, saver, sess, tvt, iter=None):
    if tvt == 'train':
        objs = [model.opt_op, model.train_loss]
    elif tvt == 'train_gc_bert':
        objs = [model.opt_op_gc_only, model.train_gc_loss, model.train_gc_acc]
    elif tvt == 'train_gc_semi':
        objs = [model.opt_op_with_gc, model.train_total_loss_with_gc, model.train_gc_acc]
    elif tvt == 'val':
        objs = [model.val_test_loss, model.pred_sim_without_act()]
    elif tvt == 'test':
        objs = [model.pred_sim_without_act()]
    elif tvt == 'test_node_emb':
        objs = [model.node_embeddings]
    elif tvt == 'test_graph_emb':
        objs = [model.graph_embeddings_val_test]
    elif tvt == 'test_att':
        objs = [model.attentions]
    elif tvt == 'val_test_gc':
        objs = [model.val_test_total_loss_with_gc, model.val_test_gc_acc]
    else:
        raise RuntimeError('Unknown train_val_test {}'.format(tvt))
    objs = saver.proc_objs(objs, tvt, iter)  # may become [loss_related_obj, objs...]
    t = time()
    outs = sess.run(objs, feed_dict=feed_dict)
    time_rtn = time() - t
    saver.proc_outs(outs, tvt, iter)
    if tvt == 'train':
        rtn = outs[-1]
    elif 'train_gc' in tvt or tvt == 'val_test_gc':
        rtn = (outs[-2], outs[-1])
    elif tvt == 'val' or tvt == 'test':
        np_result = model.apply_final_act_np(outs[-1])
        if tvt == 'val':
            rtn = (outs[-2], np_result)
        else:
            rtn = (0, np_result)
    else:
        rtn = outs[-1]
    return rtn, time_rtn


def collect_embeddings(test_gs, train_gs, model, saver, sess, gemb_only=False):
    assert (hasattr(model, 'node_embeddings'))
    # if not hasattr(model, 'graph_embeddings_val_test'):
    #     return None, None, None
    # [train + val ... test]
    all_gs = train_gs + test_gs
    node_embs_dict = defaultdict(list)  # {0: [], 1: [], ...}
    graph_embs_mat, emb_time = None, None
    if hasattr(model, 'graph_embeddings_val_test'):
        graph_embs_mat = np.zeros((len(all_gs), model.gemb_dim))
    emb_time_list = []
    for i, g in enumerate(all_gs):
        feed_dict = model.get_feed_dict_for_val_test(g, g, 1.0, False)
        if not gemb_only:
            node_embs, t = run_tf(
                feed_dict, model, saver, sess, 'test_node_emb')
            t *= 1000
            emb_time_list.append(t)
            for gcn_id, node_embs_list_length_two in enumerate(node_embs):
                assert (len(node_embs_list_length_two) == 2)
                node_embs_dict[gcn_id].append(node_embs_list_length_two[0])
        # Only collect graph-level embeddings when the model produces them.
        if hasattr(model, 'graph_embeddings_val_test'):
            graph_embs, _ = run_tf(
                feed_dict, model, saver, sess, 'test_graph_emb')
            assert (len(graph_embs) == 2)
            graph_embs_mat[i] = graph_embs[0]
    if emb_time_list:
        emb_time = np.mean(emb_time_list)
        print('node embedding time {:.5f}msec'.format(emb_time))
    if hasattr(model, 'graph_embeddings_val_test') and not gemb_only:
        print(graph_embs_mat[0:2])
    return node_embs_dict, graph_embs_mat, emb_time


def collect_attentions(test_gs, train_gs, model, saver, sess):
    if not hasattr(model, 'attentions'):
        return None
    # [train + val ... test]
    all_gs = train_gs + test_gs
    rtn = []
    for i, g in enumerate(all_gs):
        feed_dict = model.get_feed_dict_for_val_test(g, g, 1.0, False)
        atts, _ = run_tf(
            feed_dict, model, saver, sess, 'test_att')
        assert (atts.shape[1] == 1)
        rtn.append(atts)
    print('attention')
    print(rtn[0])
    return rtn


def run_gc(val_or_test_gs, model, saver, sess):
    assert hasattr(model, 'val_test_gc_loss')
    losses, accs = [], []
    for i, g in enumerate(val_or_test_gs):
        feed_dict = model.get_feed_dict_for_val_test(g, g, 1.0, True)
        (total_loss_with_gc, gc_acc), _ = run_tf(
            feed_dict, model, saver, sess, 'val_test_gc')
        losses.append(total_loss_with_gc)
        accs.append(gc_acc)
    return np.mean(losses), np.mean(accs)

"""
def get_super_large_dataset_test_collections(model_data):
    rtn = []
    assert (CSM_FLAGS.csm_slt_cat_num >= 0)
    for i in range(CSM_FLAGS.csm_slt_cat_num):
        from_info = get_flags('slt_cat_{}_from'.format(i + 1), True)
        gen_info = get_flags('slt_cat_{}_gen'.format(i + 1), True)
        rtn.append(GeneratedGraphCollection(from_info, gen_info, model_data))
    return rtn
"""

def run_super_large_dataset_test_collection(slt_collec, model, saver, sess, tvt):
    sim_dist_mat = np.zeros(slt_collec.true_ds_mat.shape)
    time_list = np.zeros(slt_collec.true_ds_mat.shape)
    for i, row_g in enumerate(slt_collec.row_gs):
        sim_dist_mat_for_i, _, time_list_for_i = run_pairs_for_val_test(
            [row_g], slt_collec.col_gs_list[i], eval, model, saver, sess, tvt,
            care_about_loss=False)
        sim_dist_mat[i] = sim_dist_mat_for_i
        time_list[i] = time_list_for_i
    return sim_dist_mat, time_list


def get_train_tvt(iter):
    need_gc = CSM_FLAGS.csm_need_gc and iter >= CSM_FLAGS.csm_iters_gc_start
    if need_gc:
        if CSM_FLAGS.csm_gc_bert_or_semi == 'bert':
            tf_opt = 'train_gc_bert'
        elif CSM_FLAGS.csm_gc_bert_or_semi == 'semi':
            tf_opt = 'train_gc_semi'
        else:
            assert (False)
    else:
        tf_opt = 'train'
    return need_gc, tf_opt


def pretty_print_dict(d, indent=0):
    for key, value in sorted(d.items()):
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty_print_dict(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))
