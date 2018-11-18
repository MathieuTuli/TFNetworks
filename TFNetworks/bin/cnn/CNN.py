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

import tensorflow as tf

from NetworkBase import NetworkBase
from train import train


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
        self.img_width = None
        self.img_height = None
        self.img_size_flat = None
        self.num_channels = None
        self.num_classes = None
        NetworkBase.__init__(config=config)

    def placeholders(self,
                     img_width,
                     img_height,
                     img_size_flat,
                     num_channels,
                     num_classes):
        '''
        TensorFlow placeholders to be fed during training
        @param img_width: int
        @param img_height: int
        @param img_size_flat: int | flattened image size
        @param num_channels: int
        @param num_classes: int

        @return x: TensorFlow Placeholder
        @return y_true: TensorFlow Placeholder
        '''
        self.img_width = img_width
        self.img_height = img_height
        self.img_size_flat = img_size_flat
        self.num_channels = num_channels
        self.num_classes = num_classes

        with self.graph.as_default():
            X = tf.placeholder(self.precision,
                               shape=[None, img_size_flat],
                               name='X')
            X = tf.reshape(X, [-1, img_width, img_height, num_channels])

            Y_true = tf.placeholder(self.precision,
                                    shape=[None, num_classes],
                                    name='Y_true')
            return X, Y_true

    def build_from_cli(self):
        '''
        Opens a cli to build the netork
        '''
        print("\n\nCNN\n")

    def build(self, settings=None):
        '''
        Unneeded.
        '''
        pass

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
                                dtype=self.precision,
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
                                         dtype=self.precision,
                                         shape=shape,
                                         name='{}_Biases'.format(name)))
        return biases

    def conv2d(self,
               layer_input=None,
               name='conv2d',
               filter_size=1,
               num_filters=1,
               stride=1,
               padding='SAME',
               activation='relu',
               use_pooling=True,
               kernel_size=1,
               kernel_stride=1,
               ):
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
        @return biases: TensorFlow Variable
        '''
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
        return layer, weights, biases

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
