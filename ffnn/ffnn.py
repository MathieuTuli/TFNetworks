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
        self.num_layers = 0
        self.check_settings(settings)
        self.build(settings)

    def build(self, settings):
        import tensorflow as tf
        self.graph = tf.Graph()
        config = tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = settings['gpu_frac']),
            device_count = {'GPU': settings['GPU']}
        )
        self.sess = tf.Session(config=config, graph=self.graph)
        with self.graph.as_default():
            print("\n\nBuilding network.\n")

            configs = settings['layers']
            layers = [data]
            for key, config in configs.items()
                print('Building {}...'.format(key))
                layers.append(self.new_layer(layers[-1], config))
                print('Completed building {}.\n'.format(key))

            print("\n\nNetwork build complete.\n")

    def train(self):
        loss = self.training_parameters['loss']
        learning_rate = self.training_parameters['learning_rate']
        weight_decay = self.training_parameters['weight_decay']
        batch_size = self.training_parameters['batch_size']
        num_iterations = self.training_parameters['num_iterations']

    def new_layer(self, config):
        input = config['input']
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
                                        name='{}_Weights'.format(self.num_layers)))
        bias = tf.Varible(tf.zeros(dtype=tf.float64, shape=[output_size]),
                            name='{}_Bias'.format(self.num_layers))
        self.num_layers += 1

        layer = None
        if activation == 'relu':
            layer = tf.nn.relu(tf.matmul(input, weights) + bias)
        elif activation == 'softmax':
            layer = tf.nn.softmax(tf.matmul(input, weights) + bias)

        assert layer is not None, "Layer was 'None' while building. Check your settings and assert tensorflow didn't run into any memory issues or such."
        return tf.nn.dropout(layer, dropout)

    def loss(self, config):
        loss = config['loss']

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
