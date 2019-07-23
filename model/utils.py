from config import FLAGS

def construct_feed_dict_for_train(data_fetcher, placeholders):
    feed_dict = dict()
    features, laplacians, sizes, labels = \
        data_fetcher.sample_train_data(FLAGS.batchsize)
    nfn = data_fetcher.get_node_feature_dim()

    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support']: laplacians})
    feed_dict.update({placeholders['num_features_nonzero']: [nfn]})
    feed_dict.update({placeholders['graph_sizes']: sizes})
    return feed_dict


def construct_feed_dict_for_encode(data_fetcher, placeholders, idx_list, tvt):
    feed_dict = dict()
    features, laplacians, sizes= \
        data_fetcher.get_data_without_label(idx_list, tvt)
    nfn = data_fetcher.get_node_feature_dim()
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support']: laplacians})
    feed_dict.update({placeholders['num_features_nonzero']: [nfn]})
    feed_dict.update({placeholders['graph_sizes']: sizes})
    return feed_dict


def construct_feed_dict_for_query(data_fetcher, placeholders, idx_list, tvt):
    feed_dict = dict()
    features, laplacians, sizes= data_fetcher.get_data_without_label(idx_list, 
                                                                     tvt)
    nfn = data_fetcher.get_node_feature_dim()
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support']: laplacians})
    feed_dict.update({placeholders['num_features_nonzero']: [nfn]})
    feed_dict.update({placeholders['graph_sizes']: sizes})
    return feed_dict



def node_match(n1, n2):
    return n1[FLAGS.node_feat_name] == n2[FLAGS.node_feat_name]

def sorted_nicely(l):
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    import re
    def alphanum_key(s):
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    return sorted(l, key=alphanum_key)


def get_similar_graphs_gid(inverted_index, code):
    """ use bfs to find all similar codes """
    sets = []
    frontier = [(code, 0, -1)]
    while len(frontier) > 0:
        cur_code, dist, last_flip_pos = frontier[0]
        frontier.pop(0)
        if cur_code in inverted_index.keys():
            sets = sets + inverted_index[cur_code]
        if dist < FLAGS.hamming_dist_thres:
            for j in range(last_flip_pos+1, len(code)):
                temp_code = list(cur_code)
                temp_code[j] = bool(1-temp_code[j])
                frontier.append((tuple(temp_code), dist+1, j))
            
    return sets
"""
def load_data(name, train):
    if name == 'syn':
        from data import SynData
        return SynData(train)
    elif name == 'imdbmulti':
        from data import IMDBMultiData
        return IMDBMultiData(train)
    elif name == 'nci109':
        from data import NCI109Data
        return NCI109Data(train)
    elif name == 'webeasy':
        from data import WebEasyData
        return WebEasyData(train)
    elif name == 'reddit5k':
        from data import Reddit5kData
        return Reddit5kData(train)
    elif name == 'reddit10k':
        from data import Reddit10kData
        return Reddit10kData(train)
    elif name == 'reddit10ksmall':
        from data import Reddit10kSmallData
        return Reddit10kSmallData(train)
    elif name == 'ptc':
        from data import PTCData
        return PTCData(train)
    else:
        raise RuntimeError('Not recognized data %s' % name)


def get_train_str(train_bool):
    if train_bool == True:
        return 'train'
    elif train_bool == False:
        return 'test'
    else:
        assert (False)


def get_root_path():
    from os.path import dirname, abspath
    return dirname(dirname(abspath(__file__)))


def get_data_path():
    return get_root_path() + '/data'


def get_save_path():
    return get_root_path() + '/save'


def get_src_path():
    return get_root_path() + '/src'


def get_model_path():
    return get_root_path() + '/model'


def create_dir_if_not_exists(dir):
    import os
    if not os.path.exists(dir):
        os.makedirs(dir)


def draw_graph(g, file):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = plt.figure()
    import networkx as nx
    nx.draw(g, ax=f.add_subplot(111))
    f.savefig(file)
    print('Saved graph to {}'.format(file))


exec_print = True


def exec_turnoff_print():
    global exec_print
    exec_print = False


def exec_turnon_print():
    global exec_print
    exec_print = True


def global_turnoff_print():
    import sys, os
    sys.stdout = open(os.devnull, 'w')


def global_turnon_print():
    import sys
    sys.stdout = sys.__stdout__


def exec_cmd(cmd, timeout=None):
    global exec_print
    if not timeout:
        from os import system
        if exec_print:
            print(cmd)
        else:
            cmd += ' > /dev/null'
        system(cmd)
        return True  # finished
    else:
        import subprocess, shlex
        from threading import Timer

        def kill_proc(proc, timeout_dict):
            timeout_dict["value"] = True
            proc.kill()

        def run(cmd, timeout_sec):
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            timeout_dict = {"value": False}
            timer = Timer(timeout_sec, kill_proc, [proc, timeout_dict])
            timer.start()
            stdout, stderr = proc.communicate()
            timer.cancel()
            return proc.returncode, stdout.decode("utf-8"), \
                   stderr.decode("utf-8"), timeout_dict["value"]

        if exec_print:
            print('Timed cmd {} sec(s) {}'.format(timeout, cmd))
        _, _, _, timeout_happened = run(cmd, timeout)
        if exec_print:
            print('timeout_happened?', timeout_happened)
        return not timeout_happened


tstamp = None


def get_ts():
    global tstamp
    if not tstamp:
        tstamp = get_current_ts()
    return tstamp


def get_current_ts():
    import datetime, pytz
    return datetime.datetime.now(pytz.timezone('US/Pacific')).strftime('%Y-%m-%dT%H:%M:%S.%f')


def get_file_base_id(file):
    return int(file.split('/')[-1].split('.')[0])

def save_as_dict(filepath, *args, **kwargs):
    '''
    Warn: To use this function, make sure to call it in ONE line, e.g.
    save_as_dict('some_path', some_object, another_object)
    Moreover, comma (',') is not allowed in the filepath.
    '''
    import inspect
    from collections import OrderedDict
    frames = inspect.getouterframes(inspect.currentframe())
    frame = frames[1]
    string = inspect.getframeinfo(frame[0]).code_context[0].strip()
    dict_to_save = OrderedDict()
    all_args_strs = string[string.find('(') + 1:-1].split(',')
    if 1 + len(args) + len(kwargs) != len(all_args_strs):
        msgs = ['Did you call this function in one line?',
                'Did the arguments have comma "," in the middle?']
        raise RuntimeError('\n'.join(msgs))
    for i, name in enumerate(all_args_strs[1:]):
        if name.find('=') != -1:
            name = name.split('=')[1]
        name = name.strip()
        if i >= 0 and i < len(args):
            dict_to_save[name] = args[i]
        else:
            break
    dict_to_save.update(kwargs)
    print('Saving a dictionary as pickle to {}'.format(filepath))
    save(filepath, dict_to_save)


def load_as_dict(filepath):
    print('Loading a dictionary as pickle from {}'.format(filepath))
    if 'pickle' in filepath:
        use_klepto = False
    else:
        use_klepto = True
    return load(filepath, use_klepto=use_klepto)


def save(filepath, obj):
    from collections import OrderedDict
    import sys
    use_klepto = (type(obj) is dict or type(obj) is OrderedDict) and len(
        obj) < 100  # too many entries --> not good for klepto TODO: figure out a better way to switch between pickle and klepto
    if sys.version_info.major < 3:  # python 2
        use_klepto = False
        filepath += '_' + 'py2'
    fp = proc_filepath(filepath, use_klepto)
    from os.path import dirname
    create_dir_if_not_exists(dirname((filepath)))
    if use_klepto:
        save_klepto(obj, fp)
        return
    with open(fp, 'wb') as handle:
        if not save_pkl(obj, handle):
            print('Cannot pickle save!')
            exec_cmd('rm {}'.format(fp))


def load(filepath, use_klepto=False):
    from os.path import isfile, isdir
    import sys
    if sys.version_info.major < 3:  # python 2
        use_klepto = False
        filepath += '_' + 'py2'
    else:
        if filepath.endswith('.klepto'):
            use_klepto = True
    fp = proc_filepath(filepath, use_klepto=use_klepto)
    if use_klepto:
        if isdir(fp):
            return load_klepto(fp)
        else:
            print('No dir {}'.format(fp))
    else:
        if isfile(fp):
            with open(fp, 'rb') as handle:
                return load_pkl(handle)
        else:
            print('No file {}'.format(fp))


def save_pkl(obj, handle):
    import pickle, sys
    if sys.version_info.major < 3:  # python 2
        pickle.dump(obj, handle)
        return True
    if sys.version_info >= (3, 4):  # qilin & feilong --> 3.4
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    return False


def load_pkl(handle):
    import pickle, sys
    if sys.version_info >= (2, 4):  # qilin & feilong --> 3.4, add python 2
        try:
            if sys.version_info >= (3, 4):  # somehow try except does not work on my desktop 3.5
                pickle_data = pickle.load(handle, encoding='latin1')
            else:
                pickle_data = pickle.load(handle)
        except UnicodeDecodeError as e:
            pickle_data = pickle.load(handle, encoding='latin1')
        except Exception as e:
            print('Unable to load data ', 'pickle_file', ':', e)
            raise
        return pickle_data
    else:
        return None


def save_klepto(dic, filepath):
    import klepto
    klepto.archives.dir_archive(filepath, dict=dic, cached=True, serialized=True).dump()


def load_klepto(filepath):
    import klepto
    rtn = klepto.archives.dir_archive(filepath, cached=True, serialized=True)
    rtn.load()
    return rtn


def proc_filepath(filepath, use_klepto):
    if type(filepath) is not str:
        raise RuntimeError('Did you pass a file path to this function?')
    ext = '.pickle' if not use_klepto else '.klepto'
    return append_ext_to_filepath(ext, filepath)


def append_ext_to_filepath(ext, fp):
    if ext not in fp:
        fp += ext
    return fp


def prompt(str, options=None):
    while True:
        t = input(str + ' ')
        if options:
            if t in options:
                return t
        else:
            return t


def prompt_get_cpu():
    from os import cpu_count
    while True:
        num_cpu = prompt(
            '{} cpus available. How many do you want?'.format( \
                cpu_count()))
        num_cpu = parse_as_int(num_cpu)
        if num_cpu and num_cpu <= cpu_count():
            return num_cpu


def parse_as_int(s):
    try:
        rtn = int(s)
        return rtn
    except ValueError:
        return None


computer_name = None


def prompt_get_computer_name():
    global computer_name
    if not computer_name:
        computer_name = prompt('What is the computer name?')
    return computer_name


def check_nx_version():
    import networkx as nx
    nxvg = '1.10'
    nxva = nx.__version__
    if nxvg != nxva:
        raise RuntimeError(
            'Wrong networkx version! Need {} instead of {}'.format(nxvg, nxva))


def format_float(f, multiply_by=None):
    if multiply_by:
        from math import pow
        return '{:.3f}e-{}'.format(f * pow(10, multiply_by), multiply_by)
    else:
        return '{:.3f}'.format(f)


def get_norm_str(norm):
    if norm is None:
        return ''
    elif norm:
        return '_norm'
    else:
        return '_nonorm'


def convert_csv_to_quoted(csvfile):
    # Use to convert an old csv data file to a new one where every field is in quotations for easier import.
    # The new file name is <old file name>-quoted<old file extension>
    import os

    assert os.path.isfile(csvfile)
    old_file_name, old_file_ext = os.path.splitext(os.path.realpath(csvfile))
    new_filepath = '{}-quoted{}'.format(old_file_name, old_file_ext)
    with open(new_filepath, 'w') as writefile:
        with open(csvfile, 'r') as readfile:
            for read_row_idx, line in enumerate(readfile):
                if read_row_idx % 10000 == 0:
                    print('Completed row: {}'.format(read_row_idx))
                if read_row_idx == 0:
                    # Handle the earlier bug case where header didn't get fixed.
                    if 'mcs' in old_file_name and 'ged' in line:
                        writeline = 'i,j,i_gid,j_gid,i_node,j_node,i_edge,j_edge,mcs,node_mapping,edge_mapping,time(msec)'
                    else:
                        writeline = line.strip()
                else:
                    line_json = eval('[' + line.strip() + ']')
                    escaped_items = ['"{}"'.format(item) for item in line_json]
                    writeline = ','.join(escaped_items)
                # print(writeline, file=writefile)


def get_bad_axes_count(data):
    import numpy as np
    row_bad_sums = np.sum(data < 0, axis=1)
    col_bad_sums = np.sum(data < 0, axis=0)

    worst_row_idx = np.argmax(row_bad_sums)
    worst_row_cnt = row_bad_sums[worst_row_idx]
    worst_col_idx = np.argmax(col_bad_sums)
    worst_col_cnt = col_bad_sums[worst_col_idx]

    return worst_row_idx, worst_row_cnt, worst_col_idx, worst_col_cnt


def prune_invalid_data(data, debug=False):
    # Given a test/train matrix npy matrix, remove rows/cols of the test train that have
    # invalid data (value < 0). Also returns the new i, j list mapping
    import numpy as np

    final_rows = list(range(data.shape[0]))
    final_cols = list(range(data.shape[1]))
    data_mutable = np.copy(data)
    while True:
        worst_row_idx, worst_row_cnt, worst_col_idx, worst_col_cnt = get_bad_axes_count(
            data_mutable)
        if worst_row_cnt == 0 and worst_col_cnt == 0:
            break

        current_rows = data_mutable.shape[0]
        current_cols = data_mutable.shape[1]
        bad_row_pct = worst_row_cnt / current_cols * 100
        bad_col_pct = worst_col_cnt / current_rows * 100
        if debug:
            print('Bad rows: {} ({:.1f}%), bad cols: {} ({:.1f}%), '.format(
                worst_row_cnt, bad_row_pct, worst_col_cnt, bad_col_pct))

        if bad_row_pct > bad_col_pct:
            data_mutable = np.delete(data_mutable, worst_row_idx, axis=0)
            del final_rows[worst_row_idx]
            if debug:
                print('Deleting row: {}'.format(worst_row_idx))
        else:
            data_mutable = np.delete(data_mutable, worst_col_idx, axis=1)
            del final_cols[worst_col_idx]
            if debug:
                print('Deleting col: {}'.format(worst_col_idx))

    return data_mutable, final_rows, final_cols


def run_data_pruning():
    import numpy as np
    import matplotlib.pyplot as plt
    filename = 'mcs_mcs_mat_imdbmulti_kCombu_cMCES_2018-10-11T22:33:32.551060_scai1_20cpus.npy'
    full_path = get_result_path() + '/imdbmulti/mcs/' + filename
    data = np.load(full_path)

    pruned_data, new_rows, new_cols = prune_invalid_data(data)

    # Naive pruning stats.
    naive_good_rows = np.sum(data >= 0, axis=1)
    naive_good_cols = np.sum(data >= 0, axis=0)
    naive_good_rows_count = np.sum(naive_good_rows == data.shape[1])
    naive_good_cols_count = np.sum(naive_good_cols == data.shape[0])
    naive_good_total = naive_good_rows_count * naive_good_cols_count

    # True data stats.
    good_data = data >= 0
    good_data_cnt = np.sum(good_data)

    # Pruning strategy stats.
    print('Initial data: {}/{}, ({:.1f})%'.format(good_data_cnt, data.size,
                                                  good_data_cnt / data.size * 100))
    print('Naive pruned: {}/{}, ({:.1f})%'.format(naive_good_total, data.size,
                                                  naive_good_total / data.size * 100))
    print('Pruned data: {}/{}, ({:.1f})%'.format(pruned_data.size, data.size,
                                                 pruned_data.size / data.size * 100))
    # Debug make sure that the new rows and new cols properly map to the correct spots
    new_data = data[new_rows, :]
    new_data = new_data[:, new_cols]
    print('Arrays equal: {}'.format(np.array_equal(new_data, pruned_data)))
    plt.imshow(new_data >= 0)
    plt.show()
    # Plot final data as binary output.
    plt.imshow(pruned_data >= 0)
    plt.show()


def compare_npy_results(data_arr):
    # Use to compare the npy results of multiple exact MCS algorithms.
    import numpy as np

    # Make sure inputs are same sizes.
    assert len(data_arr) > 1
    shape = data_arr[0].shape
    for data in data_arr:
        assert shape == data.shape

    # Compare the first to all of them and and the result.
    final_data = np.copy(data_arr[0])
    final_mask = np.full(shape, True)

    for data in data_arr[1:]:
        mask = final_data == data
        final_mask = np.logical_and(final_mask, mask)

    # Mask all invalid runs with negative value.
    final_data[final_mask == False] = -3

    return final_data


def node_has_type_attrib(g):
    for (n, d) in g.nodes(data=True):
        if 'type' in d:
            return True
    return False


if __name__ == '__main__':
    load('/home/yba/Documents/GraphEmbedding/save/IMDB1kFineData_test.pickle')
"""