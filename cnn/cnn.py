'''
    Tensorflow CNN Implementation
        -> Build graphs
        -> TensorBoard (eventually)
        -> Train
        -> Train on Google Cloud ML (eventually)
'''
# import numpy as np
from termcolor import colored

print(colored('\n TensorFlow CNN Implementation', 'blue'),
      colored('\n By Mathieu Tuli', 'red'))
print(colored('\n The following lists the settings template to follow to \
    build the graph:', 'yellow'))
print(colored('\n ', 'green'))


class CNN():
    '''
    Tensorflow CNN Implementation
        -> Build graphs
        -> TensorBoard (eventually)
        -> Train
        -> Train on Google Cloud ML (eventually)
    '''
    def __init__(self, settings):
        '''
        Init
        '''
        self.layers = list()
        self.graph = None
        self.sess = None
        self.settings = settings

    def build(self, settings):
        '''
        Build graph, respective multiprocessing rules
        '''
        import tensorflow as tf
        self.graph = tf.Graph()
        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                per_process_gpu_memory_fraction=settings['gpu_frac']),
            device_count={'GPU': settings['GPU']}
        )
        self.sess = tf.Session(config=config, graph=self.graph)
        with self.graph.as_default():
            print("\n\nBuilding network.\n")

    def conv2d(self,
               name,
               layer_input,
               filter_size,
               num_filters,
               stride,
               padding,
               activation='relu',
               use_pooling=True,
               kernel_size=None,
               kernel_stride=None):
        '''
        Convolutional layer (2 dimensional)
        '''
        import tensorflow as tf

        input_shape = layer_input.get_shape().as_list()[-1]
        shape = [filter_size, filter_size, input_shape, num_filters]
        weights = tf.Variable(
            tf.truncated_normal(shape,
                                stddev=0.05,
                                dtype=tf.float64,
                                name='{}_Weights'.format(name)))
        biases = tf.Variable(tf.constant(value=0.05,
                                         dtype=tf.float64,
                                         shape=num_filters,
                                         name='{}_Biases'.format(name)))
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
                                   padding=padding)

        if activation == 'relu':
            layer = tf.nn.relu(layer)
        else:
            raise "Unknown activation function in convolutional layer"

        self.layers.append(layer)
        return layer, weights

    def flatten(self, layer_input):
        '''
        Flatten layer
        '''
        import tensorflow as tf

        # assume [num_images, img_height,     img_width, num_channels]
        input_shape = layer_input.get_shape()

        num_features = input_shape[1:4].num_elements()

        # [num_images, num_features]
        flattened = tf.reshape(layer_input, [-1, num_features])

        return flattened, num_features

    def fully_connected(self,
                        layer_input,
                        name,
                        num_outputs,
                        use_activation=True,
                        activation='relu'):
        '''
        Fully connected layer
        '''
        import tensorflow as tf

        # input_shape = layer_input.get_shape().as_list()[-1]
        shape = [num_outputs]
        weights = tf.Variable(
            tf.truncated_normal(shape,
                                stddev=0.05,
                                dtype=tf.float64,
                                name='{}_Weights'.format(name)))
        biases = tf.Variable(tf.constant(value=0.05,
                                         dtype=tf.float64,
                                         shape=num_outputs,
                                         name='{}_Biases'.format(name)))
        layer = tf.matmul(layer_input, weights) + biases

        if use_activation:
            if activation == 'relu':
                layer = tf.nn.relu(layer)
            else:
                raise "Unknown activation function in fully_connected \
                    layer"

        self.layers.append(layer)
        return layer
