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

    def activation(self, config):
        activation = config['activation']
        input = config['input']

    def loss(self, config):
        loss = config['loss']
        input = config['input']

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

        layer_keys = [

        ]
        for layer in settings['layers']:
            for key, value in layer.items():
                if key == '':
                    required_keys = [
                        ''
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
