import numpy as np

class FFNN():
    def __init__(self, settings):
        """
        {
        settings is dict of settings for the network
            **REQUIRED**
            'gpu_frac':     <float> gpu fraction | [0,1],
            'GPU':          <int> GPU num | [1, your machine gpu count],
            'data':         <string> path to data dir, formatted as follows:
                                -data
                                    -training
                                        -images
                                            -...
                                        -labels
                                            -...
                                    -validation
                                        -images
                                            -...
                                        -labels
                                            -...
                                    -testing
                                        -images
                                            -...
                                        -labels
                                            -...
            'layers':       <dict> sequence of layers
                            {
                                'new_layer': <dict> layer specific settings. You must enumeratue each 'new_layer'. (ie new_layer_0, new_layer_1, etc.)
                                {
                                    'output_size': <int>
                                    'weight_init': <string> | "xavier"
                                    'activation': <string> | 'relu', 'softmax'
                                    'dropout': <float> | (0,1]
                                }
                            },
            'training':     <dict> training parameters
                            {
                                'loss': <string> | "mean_squared_error" . "cross_entropy"
                                'learning_rate': <float>
                                'weight_decay': <float>
                                'batch_size': <int>
                                'num_iterations': <int>
                            }

            **OPTIONAL**
                ??
        }
        """
        self.training_parameters = settings['training']
        self.check_settings(settings)
        self.load_data(settings['data'])
        self.build(settings)

    def load_data(self):
'''
        TODO:   self.input_dimensions, from load_data
                self.output_dimensions, from load_data
                self.input_dtype
                self.output_dtype
                self.num_samples
'''
        return

    def build(self, settings):
        import tensorflow as tf
        self.graph = tf.Graph()
        config = tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = settings['gpu_frac']),
            device_count = {'GPU': settings['GPU']}
        )
        self.sess = tf.Session(config=config, graph=self.graph)

        self.input = tf.placeholder(dtype=self.input_dtype, shape=self.input_dimensions, name='input')
        self.target = tf.placeholder(dtype=self.output_dtype, shape=self.output_dimensions, name='target')
        self.layers = list()
        self.layer_names = list()
        self.network_variable_scope = 'network_layers'

        with self.graph.as_default():
            print("\n\nBuilding network.\n")
            configs = settings['layers']

            with tf.variable_scope(self.net_variable_scope):
                self.layers.append(input)
                self.layer_names.append('input')
                for layer_name, config in configs.items()
                    print('Building {}...'.format(layer_name))
                    self.layers.append(self.new_layer(self.layers[-1], layer_name, config))
                    self.layer_names.append('layer_name')
                    print('Completed building {}.\n'.format(layer_name))

            self.predictions = self.layers[-1]
            print("\n\nNetwork build complete.\n")

    def generate_for_train(self):
        return data, target

    def train(self):
        optimizer = self.training_parameters['optimizer']
        learning_rate = self.training_parameters['learning_rate']
        weight_decay = self.training_parameters['weight_decay']
        batch_size = self.training_parameters['batch_size']
        num_batches = self.num_samples / batch_size
        num_iterations = self.training_parameters['num_iterations']

        loss = self.loss(self.training_parameters['loss'])

        if optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        else:
            raise ValueError("Woops, that isn't an optimizer I know. Something went wrong.")

        train_optimizer = optimizer.minimize(loss)

        classification_accuracy =   tf.subtract(
                                        1.0,
                                        tf.reduce_mean(
                                            tf.cast(
                                                tf.equal(
                                                    tf.argmax(self.predictions, 1),
                                                    tf.cast(self.target, tf.int64)),
                                                tf.float32))) * 100

        sess.run(tf.global_variables_initializer)

        for i in range(num_iterations):
            batch_data, batch_target = self.generate_batch(num_iterations, i, batch_size)

            sess.run(train_optimizer, feed_dict={self.input:batch_data, self.target:batch_target})

            if i != 0 and i % num_batches == 0:
                self.train_statistics(loss, classification_accuracy, batch_data, batch_target)


    def new_layer(self, input, name, config):
        output_size = config['output_size']
        weight_init = config['weight_init']
        activation = config['activation']
        dropout = config['dropout']

        input_size = input.get_shape().as_list()[-1]

        if weight_init == 'xavier':
            weights = tf.Variable(tf.random_normal([input_size, output_size],
                                        stddev=(3.0 / (input_size + output_size)),
                                        dtype=tf.float64,
                                        seed=521,
                                        name='{}_Weights'.format(name)))
        bias = tf.Varible(tf.zeros(dtype=tf.float64, shape=[output_size]),
                            name='{}_Bias'.format(name))

        layer = None
        if activation == 'relu':
            layer = tf.nn.relu(tf.matmul(input, weights) + bias)
        elif activation == 'softmax':
            layer = tf.nn.softmax(tf.matmul(input, weights) + bias)

        assert layer is not None, "Layer was 'None' while building. Check your settings and assert tensorflow didn't run into any memory issues or such."
        return tf.nn.dropout(layer, dropout)

    def loss(self, loss, parameters):
        if loss == 'cross_entropy':
            wdc = self.training_parameters['weight_decay']
            target = self.target
            predictions = self.predictions

            hidden_weights = self.graph.get_tensor_by_name("{}/{}_Weights:0".format(self.net_variable_scope, self.layer_names[1]))
            output_weights = self.graph.get_tensor_by_name("{}/{}_Weights:0".format(self.net_variable_scope, self.layer_names[-1]))

            Ld = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.one_hot(indices = tf.cast(target, tf.int32), depth = 10,on_value = 1.0, off_value = 0.0, axis = -1), logits = predictons))
            Lw = (wdc / 2) * tf.reduce_sum(tf.square(hidden_weights)) * tf.reduce_sum(tf.square(output_weights))
            return Ld + Lw
        else:
            raise ValueError("Woops, that isn't a loss function I know. Something went wrong.")

    def check_settings(self, settings):
        required_keys = [
            'gpu_frac',
            'GPU',
            'layers',
            'training'
        ]
        for key in required_keys:
            assert key in settings, "{} key not found in settings. Please provide the required parameters.".format(key)

        gpu_frac = settings['gpu_frac']
        GPU = settings['GPU']

        assert gpu_frac > 0 and gpu_frac <= 1.0, "gpu_frac out of bounds. Please provide a value within the continuous interval [0,1]."
        assert GPU > 0, "GPU assignment must be >= 0."

        for layer in settings['layers']:
            for key, value in layer.items():
                if key == 'new_layer':
                    required_keys = [
                        'output_size',
                        'weight_init',
                        'activation',
                        'dropout'
                    ]
                    for setting in required_keys:
                        assert setting in value, "{} key not found in {}. Please provide the required parameters.".format(settings, key)
                        if setting == 'weight_init':
                            possible_values = [
                                'xavier'
                            ]
                            assert value[setting] in possible_values, "{} is not a valid {}".format(value[setting], setting)
                        elif setting = 'activation':
                            possible_values = [
                                'relu',
                                'softmax'
                            ]
                            assert value[setting] in possible_values, "{} is not a valid {}".format(value[setting], setting)
                        elif setting == 'dropout':
                            assert isinstance(value[setting], float), "dropout must be a float"
                            assert value[setting] > 0 and value[setting] <= 1, "dropout must be greater than '0' and less or equal to '1'"
                    continue
                else:
                    print("{} is not a valid key for any layer type.".format(key))
                    raise

        for setting in settings['training']:
            for key, value in settings.items():
                if key == 'loss':
                    accepted_losses = [
                        'mean_squared_error',
                        'cross_entropy'
                    ]
                    assert value in accepted_losses, "loss must be one of {}".format(accepted_losses)
                elif key == 'learning_rate':
                    assert isinstance(value, float), "learning_rate must be a float"
                elif key == 'weight_decay':
                    assert isinstance(value, float), "weight_decay must be a float"
                elif key == 'batch_size':
                    assert isinstance(value, int), "batch_size must be an int"
                elif key == 'num_iterations':
                    assert isinstance(value, int), "num_iterations must be an int"
                else:
                    print("{} is not a valid key for any training setting.".format(key))
                    raise
