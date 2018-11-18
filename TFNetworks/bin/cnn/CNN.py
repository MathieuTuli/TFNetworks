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

import time
import tensorflow as tf
import numpy as np

from NetworkBase import NetworkBase


class CNN(NetworkBase):
    '''
    TensorFlow implementation of an adaptable Convolutional Neural Network
    Currently, only NHWC format is supported
        *For those who don't know, NHWC means your input has dimensions
            [Num_images, img_Height, img_Width, img_Channels]*
    '''
    def __init__(self, config: dict):
        '''
        Inherited from NetworkBase
        @param config: dict | has some required_keys
            *See NetworkBase pydoc for its required_keys
            *required*
            *optional*
        '''

        self.layers = dict()
        NetworkBase.__init__(config=config)

    def placeholders(self, config):
        '''
        TensorFlow placeholders to be fed during training
        @param settings: dict | has required keys
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

    def train(self,
              train_size,
              batch_generator,
              save_every=-1):
        '''
        Train your TensorFlow graph

        @param save_every: int | default = -1. Set to -1 to not save, else
            will save every 'save_every' epochs between 0 and max_epoch
        @param train_size: int | size of training data x
        @param batch_generator: function
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
                        progress = "[" + "=" * int(20*batch/num_batches) + " " *\
                            int(20 - (20*batch/num_batches)) + "]"
                        print(progress)
                        print("Accuracy: {:.3}".format(acc))
                        print("Batch Time: {:.3}".format(time.time() - batch_time))
                        print("Total Time: {:.3}".format(time.time() - start_time))

    def new_weights(self, name, shape):
        '''
        Create weights

        @param name: str
        @param shape: list | example -> [filter_size, filter_size, input_shape,
        @param num_filters]

        @return weights: TensorFlow Variable
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

        @param name: str
        @param shape: list or int

        @return biases: Tensorflow Variable
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

        @param settings: dict
            @key layer_input: Tensorflow Tensor | previous layer
            @key name: str | name of this layer. Be informative, for your own sake
            @key filter_size: int | width and height
            @key num_filters: int
            @key stride: int
            @key padding: str | ['VALID', 'SAME']
            @key activation: str | ['relu']
            @key use_pooling: boolean
            @key kernel_size: int | width and height
            @key kernel_stride: int

        @return layer: TensorFlow Tensor
        @return weights: TensorFlow Variable
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
            raise ValueError("Unknown activation function in convolutional " +
                             "layer")
        return layer, weights

    def flatten(self, settings):
        '''
        Flatten layer of dimension 4 to dimension 2

        @param layer_input: Tensorflow Tensor | previous layer (to flatten)
        @param settings: dict
            @key layer_input: TensorFlow Tensor

        @return flattened: TensorFlow Tensor | Flattend laer
        @return num_features: int
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

        @param settings: dict
            @key layer_input: TensorFlow Tensor| Previous layer. Input must be
                [num_images, input_shape]
            @key name: str | name of this layer. Be informative, for your own sake
            @key num_outputs: int
            @key use_activation: boolean
            @key activation: str | Type of activation function
                *options* -> ['relu']

        @return layer: TensorFlow Tensor | FC layer
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
                raise ValueError("Unknown activation function in " +
                                 "fully_connected layer")
        return layer

    def prediction(self, settings):
        '''
        Prediction layer
            returns regularized output of previous layer (usually a fc layer)

        @param settings: dict
            @key layer_input: TensorFlow Tensor
            @key regularizer: str | Type of refularization
                *options* -> ['softmax']

        @return regularized: TensorFlow Tensor (regularized @param layer_input)
        '''
        settings = self.configure_layer_settings(layer_type='prediction',
                                                 user_parameters=settings)

        layer_input = settings['layer_input']
        regularizer = settings['regularizer']

        if regularizer == "softmax":
            return tf.nn.softmax(layer_input)
        else:
            raise ValueError("Unkown regularizer for prediction")

    def cost(self,
             layer_input,
             y_true,
             cost='cross_entropy',
             cost_aggregate='reduce_mean'):
        '''
        Cost function to be optimized

        @para layer_input: Tensorflow Tensor
        @param y_true: Tensorflow Placeholder
        @param cost: str | Type of Tensorflow loss function
            *options* -> ['cross_entropy']
        @param cost_aggregate: str | Type of tf.math aggregate
            *options* -> ['reduce_mean']

        @return cost/loss: Tensorflow Tensor
        '''
        with self.graph.as_default():
            if cost == 'cross_entropy':
                cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_input,
                                                                  labels=y_true)
            else:
                raise ValueError("Unknown loss function")

            if cost_aggregate == 'reduce_mean':
                cost_aggregate = tf.reduce_mean(cost)
            else:
                raise ValueError("Unknown cost function")

        return cost_aggregate

    def optimizer(self,
                  cost,
                  optimization='adam'):
        '''
        Optimization method

        @param cost: TensorFlow loss | loss to minimize
        @param optimization: str | Type of Tensorflow Optimizer
            *options* -> ['adam']

        @return optimizer: TensorFlow Optimizer, defined by @param optimizer
        '''
        with self.graph.as_default():
            if optimization == 'adam':
                optimizer = tf.train.\
                    AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
            else:
                raise ValueError("Unknown optimization method")
        return optimizer

    def calculate_accuracy(self,
                           y_pred_cls,
                           accuracy_metric='reduce_mean'):
        '''
        Performance Measures

        @param y_pred_cls: TensorFlow variable
        @param y_true_cls: TensorFlow variable
        @param accuracy_metric: str | ['reduce_mean']

        @return accurac: Tensorflow Tensor, define by @param accuracy_metric
        '''
        correct_prediction = tf.equal(y_pred_cls, self.y_true_cls)

        if accuracy_metric == 'reduce_mean':
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        else:
            raise ValueError("Unkown accuracy metric")
        return accuracy
