'''
Tensorflow CNN network
'''

# TODO:
#   -Batch iterators for low memory systems
#   -Other activation functions
#   -Other optimizers
#   -Other losses
#   -Fix layer name in json formatting
#   -Fix child/parent init logic

import json
import time
import tensorflow as tf
import numpy as np

from pathlib import Path
from NetworkBase import NetworkBase


class CNN(NetworkBase):
    '''
    TensorFlow implementation of an adaptable Convolutional Neural Network
    Currently, only NHWC format is supported
        *For those who don't know, NHWC means your input has dimensions
            [Num_images, img_Height, img_Width, img_Channels]*
    '''
    def __init__(self, config, config_from_file=True):
        '''
        Inherited from NetworkBase
        config: dict | has some required_keys
            *See NetworkBase pydoc for its required_keys
            *required*
            *optional*
        '''
        if config_from_file:
            assert (Path(config).is_file()), "Couldn't find path {}"\
                .format(config)
            with open(config, 'r') as f:
                print("Config loaded.")
                config = json.load(f)

        self.iterations_passed = 0
        super().__init__(config=config)

    def pre_child_init(self):
        '''
        Before parsing config file in child
        '''
        self.layer_settings = {
            'conv2d': {
                'layer_input': None,
                'name': 'conv2d',
                'filter_size': 1,
                'num_filters': None,
                'stride': 1,
                'padding': 'SAME',
                'activation': 'relu',
                'use_pooling': True,
                'kernel_size': 1,
                'kernel_stride': 1
            },
            'flatten': {
                'layer_input': None,
                'name': 'flatten'
            },
            'fully_connected': {
                'layer_input': None,
                'name': 'flatten',
                'num_outputs': None,
                'use_activation': True,
                'activation': 'relu'
            },
            'prediction': {
                'layer_input': None,
                'name': 'prediction',
                'regularizer': 'softmax'
            },
        }

    def post_child_init(self):
        '''
        CNN initializer, overwriting call from NetworkBase
        '''
        self.x, self.y_true, self.y_true_cls =\
            self.placeholders(self.config['placeholders'])
        self.layers = dict()
        self.layers['x'] = self.x
        self.batch_size = self.config['training']['batch_size']

    def placeholders(self, config):
        '''
        TensorFlow placeholders to be fed during training
        settings: dict | has required keys
        *required*
            img_width: int
            img_height: int
            img_size_flat: int | flattened image size
        '''
        self.img_width = img_width = config['img_width']
        self.img_height = img_height = config['img_height']
        self.img_size_flat = img_size_flat = config['img_size_flat']
        self.num_channels = num_channels = config['num_channels']
        self.num_classes = num_classes = config['num_classes']

        with self.graph.as_default():
            x = tf.placeholder(tf.float32,
                               shape=[None, img_size_flat],
                               name='root_input_x')
            x = tf.reshape(x, [-1, img_width, img_height, num_channels])

            y_true = tf.placeholder(tf.float32,
                                    shape=[None, num_classes],
                                    name='y_true')
            y_true_cls = tf.argmax(y_true, axis=1)

            return x, y_true, y_true_cls

    def build(self, settings=None):
        '''
        Build the next from config
        '''
        layers = self.config['layers']

        with self.graph.as_default():
            print("\n\nBuilding network.\n")
            for layer_name, layer in layers.items():
                for layer_type, layer_parameters in layer.items():
                    prev_name = layer_parameters['layer_input']
                    curr_name = layer_parameters['name']
                    layer_parameters['layer_input'] = self.layers[prev_name]
                    if layer_type == 'conv2d':
                        conv_layer, conv_weights = self.conv2d(
                            layer_parameters)
                        self.layers[curr_name] = conv_layer
                    elif layer_type == 'flatten':
                        flattened, num_features = self.flatten(
                            layer_parameters)
                        self.layers[curr_name] = flattened
                    elif layer_type == 'fully_connected':
                        fc_layer = self.fully_connected(layer_parameters)
                        self.layers[curr_name] = fc_layer
                    elif layer_type == 'prediction':
                        prediction_layer = self.prediction(layer_parameters)
                        self.layers[curr_name] = prediction_layer
                    else:
                        raise Exception("Unknown layer type. \
                            Why didn't child_config_parser() catch it?")
            self.layers['y_pred_cls'] = tf.argmax(self.layers['prediction'],
                                                  axis=1)

    def train(self,
              train_size,
              batch_generator,
              save_every=-1):
        '''
        Train your TensorFlow graph

        save_every: int | default = -1. Set to -1 to not save, else
            it will save every 'save_every' epochs between 0 and max_epoch
        train_size: int | size of training data x
        batch_generator: function
            *signature*
                batch_generator(batch_start, batch_end)
                    ...
                    return x_batch, y_true_batch
        '''
        start_time = time.time()
        num_batches = int(train_size / self.batch_size)
        metric = self.config['accuracy']['accuracy_metric']
        cost_input = self.config['cost']['layer_input']
        cost = self.cost(layer_input=self.layers[cost_input],
                         y_true=self.y_true,
                         cost=self.config['cost']['cost'],
                         cost_aggregate=self.config['cost']['cost_aggregate'])
        optimizer = self.optimizer(cost=cost,
                                   optimization=self.config['optimizer']['optimization'])

        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            for epoch in range(0, self.max_epoch):
                if save_every > 0 and epoch % save_every == 0:
                    self.save()
                for batch in range(num_batches):
                    batch_time = time.time()
                    batch_start = self.batch_size * batch
                    batch_end = batch_start + self.batch_size
                    x_batch, y_true_batch = batch_generator(batch_start,
                                                            batch_end)
                    x_batch = np.reshape(x_batch, [-1,
                                                   self.img_width,
                                                   self.img_height,
                                                   self.num_channels])
                    feed_dict_train = {self.x: x_batch,
                                       self.y_true: y_true_batch}
                    self.sess.run(optimizer, feed_dict=feed_dict_train)

                    if not batch % 5:
                        accuracy = self.calculate_accuracy(
                            y_pred_cls=self.layers['y_pred_cls'],
                            accuracy_metric=metric)
                        acc = self.sess.run(accuracy,
                                            feed_dict=feed_dict_train)
                        progress = "[" + "=" * int(20*batch/num_batches) + " " * int(20 - (20*batch/num_batches)) + "]"
                        print(progress)
                        print("Accuracy: {:.3}".format(acc))
                        print("Batch Time: {:.3}".format(time.time() - batch_time))
                        print("Total Time: {:.3}".format(time.time() - start_time))

    def new_weights(self, name, shape):
        '''
        Create weights

        name: str
        shape: list | example -> [filter_size, filter_size, input_shape,
        num_filters]
        '''
        weights = tf.Variable(
            tf.truncated_normal(shape,
                                stddev=0.05,
                                dtype=tf.float32,
                                name='{}_Weights'.format(name)))
        return weights

    def new_biases(self, name, shape):
        '''
        Create bias

        name: str
        shape: list or int
        '''
        biases = tf.Variable(tf.constant(value=0.05,
                                         dtype=tf.float32,
                                         shape=shape,
                                         name='{}_Biases'.format(name)))
        return biases

    def configure_layer_settings(self,
                                 layer_type: str,
                                 user_parameters: dict):
        assert(layer_type in list(self.layer_settings.keys())),\
            "Woops. '{}' is not a valid layer_type. Can't configure \
                settings.".format(layer_type)

        layer_parameters = self.layer_settings[layer_type]
        for parameter, value in user_parameters.items():
            layer_parameters[parameter] = value
        return layer_parameters

    def conv2d(self, settings):
        '''
        Add a convolutional layer

        settings: dict
            layer_input: tf layer | previous layer
            name: str | name of this layer. Be informative, for your own sake
            filter_size: int | width and height
            num_filters: int
            stride: int
            padding: str | ['VALID', 'SAME']
            activation: str | ['relu']
            use_pooling: boolean
            kernel_size: int | width and height
            kernel_stride: int
        '''
        settings = self.configure_layer_settings(layer_type='conv2d',
                                                 user_parameters=settings)

        layer_input = settings['layer_input']
        name = settings['name']
        filter_size = settings['filter_size']
        num_filters = settings['num_filters']
        stride = settings['stride']
        padding = settings['padding']
        activation = settings['activation']
        use_pooling = settings['use_pooling']
        kernel_size = settings['kernel_size']
        kernel_stride = settings['kernel_stride']

        input_shape = layer_input.get_shape().as_list()[-1]
        shape = [filter_size, filter_size, input_shape, num_filters]

        weights = self.new_weights(name, shape=shape)
        biases = self.new_biases(name, shape=[num_filters])

        layer = tf.nn.conv2d(input=layer_input,
                             filter=weights,
                             strides=[1, stride, stride, 1],
                             padding=padding,
                             name='{}_Layer'.format(name))
        layer += biases

        if use_pooling:
            ksize_ = [1, kernel_size, kernel_size, 1]
            strides_ = [1, kernel_stride, kernel_stride, 1]
            layer = tf.nn.max_pool(value=layer,
                                   ksize=ksize_,
                                   strides=strides_,
                                   padding=padding,
                                   data_format='NHWC')

        if activation == 'relu':
            layer = tf.nn.relu(layer)
        else:
            raise Exception("Unknown activation function in convolutional \
                layer")
        return layer, weights

    def flatten(self, settings):
        '''
        Flatten layer of dimension 4 to dimension 2

        layer_input: tf layer | previous layer (to flatten)
        '''
        settings = self.configure_layer_settings(layer_type='flatten',
                                                 user_parameters=settings)
        layer_input = settings['layer_input']

        # assume [num_images, img_height, img_width, num_channels]
        input_shape = layer_input.get_shape()

        # num_features = img_height * img_width * num_channels
        num_features = input_shape[1:4].num_elements()

        # [num_images, num_features]
        flattened = tf.reshape(layer_input, [-1, num_features])

        return flattened, num_features

    def fully_connected(self, settings):
        '''
        Add a fully connected layer

        settings: dict
            layer_input: tf layer | previous layer. Input must be
                [num_images, input_shape]
            name: str | name of this layer. Be informative, for your own sake
            num_outputs: int
            use_activation: boolean
            activation: str | ['relu']
        '''
        settings = self.configure_layer_settings(layer_type='fully_connected',
                                                 user_parameters=settings)

        layer_input = settings['layer_input']
        name = settings['name']
        num_outputs = settings['num_outputs']
        use_activation = settings['use_activation']
        activation = settings['activation']

        input_shape = layer_input.get_shape().as_list()[-1]
        weights = self.new_weights(name, shape=[input_shape, num_outputs])
        biases = self.new_biases(name, shape=[num_outputs])

        layer = tf.matmul(layer_input, weights) + biases

        if use_activation:
            if activation == 'relu':
                layer = tf.nn.relu(layer)
            else:
                raise Exception("Unknown activation function in \
                    fully_connected layer")
        return layer

    def prediction(self, settings):
        '''
        Prediction layer
            returns regularized output of previous layer (usually a fc layer)

        settings: dict
            layer_input: tf layer
            regularizer: str | ['softmax']
        '''
        settings = self.configure_layer_settings(layer_type='prediction',
                                                 user_parameters=settings)

        layer_input = settings['layer_input']
        regularizer = settings['regularizer']

        if regularizer == "softmax":
            return tf.nn.softmax(layer_input)
        else:
            raise Exception("Unkown regularizer for prediction")

    def cost(self,
             layer_input,
             y_true,
             cost='cross_entropy',
             cost_aggregate='reduce_mean'):
        '''
        Cost function to be optimized

        cost: str | ['cross_entropy']
        cost_aggregate: str | ['reduce_mean']
        '''
        with self.graph.as_default():
            if cost == 'cross_entropy':
                cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_input,
                                                               labels=y_true)
            else:
                raise Exception("Unknown loss function")

            if cost_aggregate == 'reduce_mean':
                cost_aggregate = tf.reduce_mean(cost)
            else:
                raise Exception("Unknown cost function")

        return cost_aggregate

    def optimizer(self,
                  cost,
                  optimization='adam'):
        '''
        Optimization method

        cost: TensorFlow variable | cost to minimize
        optimization: str | ['adam']
        '''
        with self.graph.as_default():
            if optimization == 'adam':
                optimizer = tf.train.\
                    AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
            else:
                raise Exception("Unknown optimization method")
        return optimizer

    def calculate_accuracy(self,
                           y_pred_cls,
                           accuracy_metric='reduce_mean'):
        '''
        Performance Measures

        y_pred_cls: TensorFlow variable
        y_true_cls: TensorFlow variable
        accuracy_metric: str | ['reduce_mean']
        '''
        correct_prediction = tf.equal(y_pred_cls, self.y_true_cls)

        if accuracy_metric == 'reduce_mean':
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        else:
            raise Exception("Unkown accuracy metric")
        return accuracy

    def child_parse_config(self, config):
        '''
        Parses the config file to assert validity for CNN network.

        *Note: 'placeholders' is a required key from the parent class
            NetworkBase. It must appear here with corresponding sub keys.
        '''
        required_keys = [
            'placeholders',
            'layers',
            'cost',
            'optimizer',
            'accuracy',
            'placeholders',
            'training',
        ]

        req_sub_keys = {
            'placeholders': [
                'img_width',
                'img_height',
                'img_size_flat',
                'num_channels',
                'num_classes',
            ],
            'accuracy': [
                'accuracy_metric',
            ],
            'training': [
                'batch_size',
            ]
        }

        possible_layers = list(self.layer_settings.keys())

        # TODO: Could this be done recursively?
        for key in required_keys:
            assert (key in config), "Missing key '{}' in config".format(key)
            if key == 'layers':
                for layer_name, layer in config[key].items():
                    for layer_type, layer_parameters in layer.items():
                        assert (layer_type in possible_layers), "'{}' is not \
                            a valid layer."
                        possible_params = list(self.layer_settings[layer_type]
                                                   .keys())
                        # TODO: Undefined 'param' in assert message.
                        #   Is there a workaround?
                        # assert(all(param in possible_params
                        #            for param in layer_parameters)),\
                        #     "'{}' is not a valid layer_parameter for layer\
                        #         '{}'".format(param, layer_type)

                        for param in layer_parameters:
                            assert param in possible_params, "'{}' is not a \
                                vaid layer_parameter for layer '{}'"\
                                .format(param, layer_type)
                        assert 'layer_input' in layer_parameters,\
                            "'layer_input' is not a user defined \
                                parameter. It must be."

            if key in req_sub_keys:
                for sub_key in req_sub_keys[key]:
                    assert (sub_key in config[key]),\
                        "Missing key '{}' in config[{}]".format(sub_key, key)
