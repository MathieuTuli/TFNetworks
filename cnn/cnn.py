class CNN():
    def __init__(self, settings):
        """
        {
        settings is dict of settings for the network
            **REQUIRED**
            'gpu_frac':     <float> gpu fraction | [0,1],
            'GPU':          <int> GPU num | [1, your machine gpu count],
            'data':         <dict> train, validation, test data. All numpy arrays
                            {
                                'train': <object tuple> (data, label)
                                'val': <object tuple> (data, label)
                                'test': <single object> data
                            }
            'layers':       <dict> sequence of layers
                            {
                                'conv2d': <dict> layer specific settings
                                    {
                                        'name': <string>
                                        'kern_size': <positive non-zero int tuple> (w,h)
                                        'kern_num': <postive non-zero int>
                                        'padding': <string> | "SAME" or "VALID"
                                        'stride': <positive non-zero int>
                                        'activation': <string>
                                    }
                                'pool2d': <dict> layer specific settings
                                    {
                                        'name': <string>
                                        'kern_size': <positive non-zero int tuple> (w,h)
                                        'stride': <positive non-zero int>
                                        'activation': <string>
                                        'type': <string> | "MAX"
                                    }
                                'fc': <dict> layer specific settings
                                    {
                                        'name': <string>
                                        'output_size': <positive non-zero int>
                                        'activation': <string>
                                    }
                            },
            'training':     <dict> training settings
                            {
                                'loss': <string> | "mean_squared_error"
                            }

            **OPTIONAL**
                ??
        }
        """
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

    def conv2d(self, config):
        name = config['name']
        kern_size = config['kern_size']
        kern_num = config['kern_num']
        padding = config['padding']
        stride = config['stride']
        activation = config['activation']

    def pool2d(self, config):
        name = config['name']
        kern_size = config['kern_size']
        stride = config['stride']
        activation = config['activation']
        type = config['type']

    def fc(self, config):
        name = config['name']
        output_size = config['output_size']
        activation = config['activation']

    def activation(self, config):
        activation = config['activation']

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
        num_layers = settings['layers']
        loss = settings['loss']

        assert gpu_frac > 0 and gpu_frac <= 1.0, "gpu_frac out of bounds. Please provide a value within the continuous interval [0,1]."
        assert GPU > 0, "GPU assignment must be >= 0."

        layer_keys = [
            'conv2d',
            'pool2d',
            'fc'
        ]
        assert len(settings['layers']) > 3
        for layer in settings['layers']:
            for key, value in layer.items():
                if key == 'conv2d':
                    required_keys = [
                        'name',
                        'kern_size',
                        'kern_num',
                        'padding',
                        'stride',
                        'activation'
                    ]
                    for setting in required_keys:
                        assert setting in value, "{} key not found in {}. Please provide the required parameters.".format(settings, key)
                    continue
                elif key == 'pool2d':
                    required_keys = [
                        'name',
                        'kern_size',
                        'stride',
                        'activation',
                        'type'
                    ]
                    for setting in required_keys:
                        assert setting in value, "{} key not found in {}. Please provide the required parameters.".format(settings, key)
                    continue
                elif key == 'fc':
                    required_keys = [
                        'name',
                        'output_size',
                        'activation'
                    ]
                    for setting in required_keys:
                        assert setting in value, "{} key not found in {}. Please provide the required parameters.".format(settings, key)
                    continue
                else:
                    print("{} is not a valid key for any layer type.".format(key))
                    raise

        for setting in settings['training']:
            for key, value in settings.items():
                if key == 'loss':
                    accepted_losses = [
                        'mean_squared_error'
                    ]
                    assert loss in accepted_losses, "loss must be one of {}".format(accepted_losses)
                else:
                    print("{} is not a valid key for any training setting.".format(key))
                    raise
