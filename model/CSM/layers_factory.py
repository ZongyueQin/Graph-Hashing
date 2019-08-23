from CSM_config import CSM_FLAGS
from CSM_layers import CSM_GraphConvolution, CSM_GraphConvolutionAttention, CSM_Coarsening, Average, \
    CSM_Attention, CSM_Dot, CSM_Dist, CSM_NTN, CSM_SLM, CSM_Dense, CSM_Padding, CSM_MNE, CSM_MNEMatch, CSM_CNN, CSM_ANPM, CSM_ANPMD, CSM_ANNH, \
    CSM_GraphConvolutionCollector, CSM_MNEResize, CSM_PadandTruncate, CSM_Supersource, CSM_GMNAverage, \
    CSM_JumpingKnowledge, CSM_DeepSets
from utils_siamese import create_activation


def create_layers(model, pattern, num_layers, max_node):
    layers = []
    for i in range(1, num_layers + 1):  # 1-indexed
        sp = CSM_FLAGS.flag_values_dict()['{}_{}'.format(pattern, i)].split(':')
        name = sp[0]
        layer_info = {}
        if len(sp) > 1:
            assert (len(sp) == 2)
            for spec in sp[1].split(','):
                ssp = spec.split('=')
                layer_info[ssp[0]] = ssp[1]
        if name == 'CSM_GraphConvolution':
            layers.append(create_CSM_GraphConvolution_layer(layer_info, model, i))
        elif name == 'CSM_GraphConvolutionAttention':
            layers.append(create_CSM_GraphConvolutionAttention_layer(layer_info, model, i))
        elif name == 'CSM_GraphConvolutionCollector':
            layers.append(create_CSM_GraphConvolutionCollector_layer(layer_info, max_node))
        elif name == 'CSM_Coarsening':
            layers.append(create_CSM_Coarsening_layer(layer_info))
        elif name == 'Average':
            layers.append(create_Average_layer(layer_info))
        elif name == 'CSM_Attention':
            layers.append(create_CSM_Attention_layer(layer_info))
        elif name == 'CSM_Supersource':
            layers.append(create_CSM_Supersource_layer(layer_info))
        elif name == 'CSM_GMNAverage':
            layers.append(create_CSM_GMNAverage_layer(layer_info))
        elif name == 'CSM_JumpingKnowledge':
            layers.append(create_CSM_JumpingKnowledge_layer(layer_info))
        elif name == 'CSM_Dot':
            layers.append(create_CSM_Dot_layer(layer_info))
        elif name == 'CSM_Dist':
            layers.append(create_CSM_Dist_layer(layer_info))
        elif name == 'CSM_SLM':
            layers.append(create_CSM_SLM_layer(layer_info))
        elif name == 'CSM_NTN':
            layers.append(create_CSM_NTN_layer(layer_info))
        elif name == 'CSM_ANPM':
            layers.append(create_CSM_ANPM_layer(layer_info))
        elif name == 'CSM_ANPMD':
            layers.append(create_CSM_ANPMD_layer(layer_info))
        elif name == 'CSM_ANNH':
            layers.append(create_CSM_ANNH_layer(layer_info))
        elif name == 'CSM_Dense':
            layers.append(create_CSM_Dense_layer(layer_info))
        elif name == 'CSM_Padding':
            layers.append(create_CSM_Padding_layer(layer_info))
        elif name == 'CSM_PadandTruncate':
            layers.append(create_CSM_PadandTruncate_layer(layer_info))
        elif name == 'CSM_MNE':
            layers.append(create_CSM_MNE_layer(layer_info))
        elif name == 'CSM_MNEMatch':
            layers.append(create_CSM_MNEMatch_layer(layer_info))
        elif name == 'CSM_MNEResize':
            layers.append(create_CSM_MNEResize_layer(layer_info, max_node))
        elif name == 'CSM_CNN':
            layers.append(create_CSM_CNN_layer(layer_info))
        elif name == 'CSM_DeepSets':
            layers.append(create_CSM_DeepSets_layer(layer_info))
        else:
            raise RuntimeError('Unknown layer {}'.format(name))
    return layers


def create_CSM_GraphConvolution_layer(layer_info, model, layer_id):
    if not 5 <= len(layer_info) <= 7:
        raise RuntimeError('CSM_GraphConvolution layer must have 5-7 specs')
    input_dim = layer_info.get('input_dim')
    if not input_dim:
        if layer_id != 1:
            raise RuntimeError(
                'The input dim for layer must be specified'.format(layer_id))
        input_dim = model.input_dim
    else:
        input_dim = int(input_dim)
    return CSM_GraphConvolution(
        input_dim=input_dim,
        output_dim=int(layer_info['output_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        sparse_inputs=parse_as_bool(layer_info['sparse_inputs']),
        act=create_activation(layer_info['act']),
        bias=parse_as_bool(layer_info['bias']),
        featureless=False,
        num_supports=1,
        type=layer_info['type'] if 'type' in layer_info else 'gcn')


def create_CSM_GraphConvolutionAttention_layer(layer_info, model, layer_id):
    if not 5 <= len(layer_info) <= 6:
        raise RuntimeError('CSM_GraphConvolution layer must have 3-4 specs')
    input_dim = layer_info.get('input_dim')
    if not input_dim:
        if layer_id != 1:
            raise RuntimeError(
                'The input dim for layer must be specified'.format(layer_id))
        input_dim = model.input_dim
    else:
        input_dim = int(input_dim)
    return CSM_GraphConvolutionAttention(
        input_dim=input_dim,
        output_dim=int(layer_info['output_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        sparse_inputs=parse_as_bool(layer_info['sparse_inputs']),
        act=create_activation(layer_info['act']),
        bias=parse_as_bool(layer_info['bias']),
        featureless=False,
        num_supports=1)


def create_CSM_GraphConvolutionCollector_layer(layer_info, max_node):
    if not len(layer_info) == 6:
        raise RuntimeError('CSM_GraphConvolutionCollector layer must have 6 spec')
    return CSM_GraphConvolutionCollector(gcn_num=int(layer_info['gcn_num']),
                                     fix_size=int(layer_info['fix_size']),
                                     mode=int(layer_info['mode']),
                                     padding_value=int(layer_info['padding_value']),
                                     align_corners=parse_as_bool(layer_info['align_corners']),
                                     plhdr_max_node = max_node,
                                     perturb=layer_info['perturb'])


def create_CSM_Coarsening_layer(layer_info):
    if not len(layer_info) == 1:
        raise RuntimeError('CSM_Coarsening layer must have 1 spec')
    return CSM_Coarsening(pool_style=layer_info['pool_style'])


def create_Average_layer(layer_info):
    if not len(layer_info) == 0:
        raise RuntimeError('Average layer must have 0 specs')
    return Average()


def create_CSM_Attention_layer(layer_info):
    if not len(layer_info) == 5:
        raise RuntimeError('CSM_Attention layer must have 5 specs')
    return CSM_Attention(input_dim=int(layer_info['input_dim']),
                     att_times=int(layer_info['att_times']),
                     att_num=int(layer_info['att_num']),
                     att_style=layer_info['att_style'],
                     att_weight=parse_as_bool(layer_info['att_weight']))


def create_CSM_Supersource_layer(layer_info):
    if not len(layer_info) == 0:
        raise RuntimeError('CSM_Supersource layer must have 0 specs')
    return CSM_Supersource()


def create_CSM_GMNAverage_layer(layer_info):
    if not len(layer_info) == 2:
        raise RuntimeError('CSM_GMNAverage layer must have 2 specs')
    return CSM_GMNAverage(input_dim=int(layer_info['input_dim']),
                      output_dim=int(layer_info['output_dim']))


def create_CSM_JumpingKnowledge_layer(layer_info):
    if not len(layer_info) == 8:
        raise RuntimeError('CSM_JumpingKnowledge layer must have 8 specs')
    return CSM_JumpingKnowledge(gcn_num=int(layer_info['gcn_num']),
                            gcn_layer_ids=parse_as_int_list(
                                layer_info['gcn_layer_ids']),
                            input_dims=parse_as_int_list(layer_info['input_dims']),
                            att_times=int(layer_info['att_times']),
                            att_num=int(layer_info['att_num']),
                            att_style=layer_info['att_style'],
                            att_weight=parse_as_bool(layer_info['att_weight']),
                            combine_method=layer_info['combine_method'])


def create_CSM_Dot_layer(layer_info):
    if not len(layer_info) == 2:
        raise RuntimeError('CSM_Dot layer must have 2 specs')
    return CSM_Dot(output_dim=int(layer_info['output_dim']),
               act=create_activation(layer_info['act']))


def create_CSM_Dist_layer(layer_info):
    if not len(layer_info) == 1:
        raise RuntimeError('CSM_Dot layer must have 1 specs')
    return CSM_Dist(norm=layer_info['norm'])


def create_CSM_SLM_layer(layer_info):
    if not len(layer_info) == 5:
        raise RuntimeError('CSM_SLM layer must have 5 specs')
    return CSM_SLM(
        input_dim=int(layer_info['input_dim']),
        output_dim=int(layer_info['output_dim']),
        act=create_activation(layer_info['act']),
        dropout=parse_as_bool(layer_info['dropout']),
        bias=parse_as_bool(layer_info['bias']))


def create_CSM_NTN_layer(layer_info):
    if not len(layer_info) == 6:
        raise RuntimeError('CSM_NTN layer must have 6 specs')
    return CSM_NTN(
        input_dim=int(layer_info['input_dim']),
        feature_map_dim=int(layer_info['feature_map_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        inneract=create_activation(layer_info['inneract']),
        apply_u=parse_as_bool(layer_info['apply_u']),
        bias=parse_as_bool(layer_info['bias']))


def create_CSM_ANPM_layer(layer_info):
    if not len(layer_info) == 14:
        raise RuntimeError('CSM_ANPM layer must have 14 specs')
    return CSM_ANPM(
        input_dim=int(layer_info['input_dim']),
        att_times=int(layer_info['att_times']),
        att_num=int(layer_info['att_num']),
        att_style=layer_info['att_style'],
        att_weight=parse_as_bool(layer_info['att_weight']),
        feature_map_dim=int(layer_info['feature_map_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        bias=parse_as_bool(layer_info['bias']),
        ntn_inneract=create_activation(layer_info['ntn_inneract']),
        apply_u=parse_as_bool(layer_info['apply_u']),
        padding_value=int(layer_info['padding_value']),
        mne_inneract=create_activation(layer_info['mne_inneract']),
        # num_bins=int(layer_info['num_bins'])
        mne_method=layer_info['mne_method'],
        branch_style=layer_info['branch_style'])


def create_CSM_ANPMD_layer(layer_info):
    if not len(layer_info) == 22:
        raise RuntimeError('CSM_ANPMD layer must have 22 specs')
    return CSM_ANPMD(
        input_dim=int(layer_info['input_dim']),
        att_times=int(layer_info['att_times']),
        att_num=int(layer_info['att_num']),
        att_style=layer_info['att_style'],
        att_weight=parse_as_bool(layer_info['att_weight']),
        feature_map_dim=int(layer_info['feature_map_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        bias=parse_as_bool(layer_info['bias']),
        ntn_inneract=create_activation(layer_info['ntn_inneract']),
        apply_u=parse_as_bool(layer_info['apply_u']),
        padding_value=int(layer_info['padding_value']),
        mne_inneract=create_activation(layer_info['mne_inneract']),
        mne_method=layer_info['mne_method'],
        branch_style=layer_info['branch_style'],
        dense1_dropout=parse_as_bool(layer_info['dense1_dropout']),
        dense1_act=create_activation(layer_info['dense1_act']),
        dense1_bias=parse_as_bool(layer_info['dense1_bias']),
        dense1_output_dim=int(layer_info['dense1_output_dim']),
        dense2_dropout=parse_as_bool(layer_info['dense2_dropout']),
        dense2_act=create_activation(layer_info['dense2_act']),
        dense2_bias=parse_as_bool(layer_info['dense2_bias']),
        dense2_output_dim=int(layer_info['dense2_output_dim']))


def create_CSM_ANNH_layer(layer_info):
    if not len(layer_info) == 14:
        raise RuntimeError('CSM_ANNH layer must have 14 specs')
    return CSM_ANNH(
        input_dim=int(layer_info['input_dim']),
        att_times=int(layer_info['att_times']),
        att_num=int(layer_info['att_num']),
        att_style=layer_info['att_style'],
        att_weight=parse_as_bool(layer_info['att_weight']),
        feature_map_dim=int(layer_info['feature_map_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        bias=parse_as_bool(layer_info['bias']),
        ntn_inneract=create_activation(layer_info['ntn_inneract']),
        apply_u=parse_as_bool(layer_info['apply_u']),
        padding_value=int(layer_info['padding_value']),
        mne_inneract=create_activation(layer_info['mne_inneract']),
        # num_bins=int(layer_info['num_bins'])
        mne_method=layer_info['mne_method'],
        branch_style=layer_info['branch_style'])


def create_CSM_Dense_layer(layer_info):
    if not len(layer_info) == 5:
        raise RuntimeError('CSM_Dense layer must have 5 specs')
    return CSM_Dense(
        input_dim=int(layer_info['input_dim']),
        output_dim=int(layer_info['output_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        act=create_activation(layer_info['act']),
        bias=parse_as_bool(layer_info['bias']))


def create_CSM_Padding_layer(layer_info):
    if not len(layer_info) == 1:
        raise RuntimeError('CSM_Padding layer must have 1 specs')
    return CSM_Padding(
        padding_value=int(layer_info['padding_value']))


def create_CSM_PadandTruncate_layer(layer_info):
    if not len(layer_info) == 1:
        raise RuntimeError('CSM_PadandTruncate layer must have 1 specs')
    return CSM_PadandTruncate(
        padding_value=int(layer_info['padding_value']))


def create_CSM_MNE_layer(layer_info):
    if not len(layer_info) == 3:
        raise RuntimeError('CSM_MNE layer must have 3 specs')
    return CSM_MNE(
        input_dim=int(layer_info['input_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        inneract=create_activation(layer_info['inneract']))


def create_CSM_MNEMatch_layer(layer_info):
    if not len(layer_info) == 2:
        raise RuntimeError('CSM_MNEMatch layer must have 2 specs')
    return CSM_MNEMatch(
        input_dim=int(layer_info['input_dim']),
        inneract=create_activation(layer_info['inneract']))


def create_CSM_MNEResize_layer(layer_info, max_node):
    if not len(layer_info) == 6:
        raise RuntimeError('CSM_MNEResize layer must have 6 specs')
    return CSM_MNEResize(
        plhdr_max_node = max_node,
        dropout=parse_as_bool(layer_info['dropout']),
        inneract=create_activation(layer_info['inneract']),
        fix_size=int(layer_info['fix_size']),
        mode=int(layer_info['mode']),
        padding_value=int(layer_info['padding_value']),
        align_corners=parse_as_bool(layer_info['align_corners']))


def create_CSM_CNN_layer(layer_info):
    if not 11 <= len(layer_info) <= 13:
        raise RuntimeError('CSM_CNN layer must have 11-13 specs')
    gcn_num = layer_info.get('gcn_num')
    mode = layer_info.get('mode')
    if not gcn_num:
        if layer_info['mode'] != 'merge':
            raise RuntimeError('The gcn_num for layer must be specified')
        gcn_num = None
    else:
        gcn_num = int(gcn_num)
    mode = 'merge' if not mode else mode

    return CSM_CNN(
        start_cnn=parse_as_bool(layer_info['start_cnn']),
        end_cnn=parse_as_bool(layer_info['end_cnn']),
        window_size=int(layer_info['window_size']),
        kernel_stride=int(layer_info['kernel_stride']),
        in_channel=int(layer_info['in_channel']),
        out_channel=int(layer_info['out_channel']),
        padding=layer_info['padding'],
        pool_size=int(layer_info['pool_size']),
        dropout=parse_as_bool(layer_info['dropout']),
        act=create_activation(layer_info['act']),
        bias=parse_as_bool(layer_info['bias']),
        mode=mode,
        gcn_num=gcn_num)


def create_CSM_DeepSets_layer(layer_info):
    if not len(layer_info) == 2:
        raise RuntimeError('CSM_DeepSets layer must have 2 specs')
    return CSM_DeepSets(style=layer_info['style'],
                    gcn_num=int(layer_info['gcn_num']))


def parse_as_bool(b):
    if b == 'True':
        return True
    elif b == 'False':
        return False
    else:
        raise RuntimeError('Unknown bool string {}'.format(b))


def parse_as_int_list(il):
    rtn = []
    for x in il.split('_'):
        x = int(x)
        rtn.append(x)
    return rtn
