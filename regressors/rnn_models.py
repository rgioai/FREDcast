from utils import get_data

import tflearn


def valid_layer_param_dict(param_dict, architecture):
    assert isinstance(param_dict, dict)
    assert isinstance(architecture, str)
    default = default_layer_param_dict(architecture)
    for k in default:
        if k not in param_dict:
            param_dict[k] = default[k]
        else:
            assert isinstance(param_dict[k], type(default[k]))
    return param_dict


def default_layer_param_dict(architecture):
    assert isinstance(architecture, str)
    if architecture == 'rnn':
        return {
            'activation': 'sigmoid',
            'dropout': None,
            'bias': True,
            'weights_init': None,
            'return_seq': False,
            'return_state': False,
            'initial_state': None,
            'dynamic': False,
            'trainable': True,
            'restore': True,
            'reuse': False,
            'scope': None
        }
    elif architecture == 'lstm':
        return {
            'activation': 'tanh',
            'inner_activation': 'sigmoid',
            'dropout': None,
            'bias': True,
            'weights_init': None,
            'forget_bias': 1.0,
            'return_seq': False,
            'return_state': False,
            'initial_state': None,
            'dynamic': False,
            'trainable': True,
            'restore': True,
            'reuse': False,
            'scope': None
        }
    elif architecture == 'gru':
        return {
            'activation': 'tanh',
            'inner_activation': 'sigmoid',
            'dropout': None,
            'bias': True,
            'weights_init': None,
            'return_seq': False,
            'return_state': False,
            'initial_state': None,
            'dynamic': False,
            'trainable': True,
            'restore': True,
            'reuse': False,
            'scope': None
        }
    else:
        raise ValueError('Unsupported architecture: %s' % architecture)


def valid_model_param_dict(param_dict):
    assert isinstance(param_dict, dict)
    default = default_model_param_dict()
    for k in default:
        if k not in param_dict:
            param_dict[k] = default[k]
        else:
            assert isinstance(param_dict[k], type(default[k]))
    return param_dict


def default_model_param_dict():
    """
    incoming, placeholder=None, optimizer='adam', loss='categorical_crossentropy',
    metric='default', learning_rate=0.001, dtype=tf.float32, batch_size=64,
    shuffle_batches=True, to_one_hot=False, n_classes=None,
    trainable_vars=None, restore=True, op_name=None,
    validation_monitors=None, validation_batch_size=None, name=None
    :return:
    """
    return {'optimizer': 'adam',
            'loss': 'categorical_crossentropy',
            'metric': 'r2',  # TODO verify this works
            'learning_rate': 0.001,
            'dtype': tf.float32,
            'batch_size': 64,
            'shuffle_batches': True,
            'to_one_hot': False,
            'n_classes': None,
            'trainable_vars': None,
            'restore': True,
            'op_name': None,
            'validation_monitors': None,
            'validation_batch_size': None,
            'name': None
        }


def rnn_model(data_width, scaling_factor, architecture='lstm', model_params=None, layer_params=None):
    data_width = int(data_width)
    assert scaling_factor in range(2, 10)
    assert architecture in ['rnn', 'lstm', 'gru']
    if architecture == 'rnn':
        layer_function = simple_rnn
    elif architecture == 'lstm':
        layer_function = lstm_layer
    elif architecture == 'gru':
        layer_function = gru_layer
    else:
        raise ValueError('Unsupported architecture: %s' % architecture)
    if layer_params is None:
        layer_params = default_layer_param_dict(architecture)
    layer_params = valid_layer_param_dict(layer_params, architecture)

    net = tflearn.input_data(shape=[None, data_width])
    for i in range(1, scaling_factor):
        this_layer_width = int((1 - (i/scaling_factor)) * data_width)
        net = layer_function(network=net, n_units=this_layer_width, param_dict=layer_params)
    net = tflearn.fully_connected(net, 1, activation=model_params['activation'])
    net = tflearn.regression(net)  # TODO Add model params
    return tflearn.DNN(net)


def gru_layer(network, n_units, param_dict):
    return tflearn.lstm(network,
                        n_units=n_units,
                        activation=param_dict['activation'],
                        inner_activation=param_dict['inner_activation'],
                        dropout=param_dict['dropout'],
                        bias=param_dict['bias'],
                        weights_init=param_dict['weights_init'],
                        return_seq=param_dict['return_seq'],
                        return_state=param_dict['return_state'],
                        initial_state=param_dict['initial_state'],
                        dynamic=param_dict['dynamic'],
                        trainable=param_dict['trainable'],
                        restore=param_dict['restore'],
                        reuse=param_dict['reuse'],
                        scope=param_dict['scope'])


def lstm_layer(network, n_units, param_dict):
    return tflearn.lstm(network,
                        n_units=n_units,
                        activation=param_dict['activation'],
                        inner_activation=param_dict['inner_activation'],
                        dropout=param_dict['dropout'],
                        bias=param_dict['bias'],
                        weights_init=param_dict['weights_init'],
                        forget_bias=param_dict['forget_bias'],
                        return_seq=param_dict['return_seq'],
                        return_state=param_dict['return_state'],
                        initial_state=param_dict['initial_state'],
                        dynamic=param_dict['dynamic'],
                        trainable=param_dict['trainable'],
                        restore=param_dict['restore'],
                        reuse=param_dict['reuse'],
                        scope=param_dict['scope'])


def simple_rnn(network, n_units, param_dict):
    return tflearn.simple_rnn(network,
                              n_units=n_units,
                              activation=param_dict['activation'],
                              dropout=param_dict['dropout'],
                              bias=param_dict['bias'],
                              weights_init=param_dict['weights_init'],
                              return_seq=param_dict['return_seq'],
                              return_state=param_dict['return_state'],
                              initial_state=param_dict['initial_state'],
                              dynamic=param_dict['dynamic'],
                              trainable=param_dict['trainable'],
                              restore=param_dict['restore'],
                              reuse=param_dict['reuse'],
                              scope=param_dict['scope'])
