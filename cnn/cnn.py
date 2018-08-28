import tensorflow as tf

class CNN():
    def __init__(self, settings):
        """
        settings: dict of settings for the network
            **Required**
                'gpu_frac': gpu fraction
                'GPU': GPU num
                'layers': list of dict
                    Available dict keys:
                        'conv'
                        'pool'
                        'fc'

                'loss': string, loss function. | "mean_squared_error" |
            **Optional**
                ??
        """
        self.accepted_losses = ["mean_squared_error"]
        self.check_settings(settings)

        self.graph = tf.Graph()
        self.build(settings['num_layers'], settings['filters'])

        config = tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = settings['gpu_frac']),
            device_count = {'GPU': settings['GPU']}
        )
        self.sess = tf.Session(config=config, graph=self.graph)

    def build(self, num_layers, filters):
        with self.graph.as_default():
            print("Building network.\n")
        return

    def conv(self):
        return

    def pool(self):
        returns

    def fc(self):
        return

    def loss(self):
        return

    def check_settings(self, settings):
        required_keys = [
            'gpu_frac',
            'GPU',
            'layers',
            'loss'
        ]
        for key in required_keys:
            assert(key in settings, "{} key not found in settings. Please provide the required parameters.".format(key))

        gpu_frac = settings['gpu_frac']
        GPU = settings['GPU']
        num_layers = settings['num_layers']
        filters = settings['filters']
        loss = settings['loss']

        assert(gpu_frac > 0 and gpu_frac <= 1.0, "gpu_frac out of bounds. Please provide a value within the continuous interval [0,1].")
        assert(GPU > 0, "GPU must be >= 0.")

        layer_keys = [
            'conv',
            'pool',
            'fc'
        ]
        assert(len(settings['layers']>3)
        for layer in settings['layers']:
            for key, value in layer.items():
                if key == 'conv':
                    continue
                elif key == 'pool':
                    continue
                elif key == 'fc':
                    continue
                else:
                    print("{} is not a valid key for any layer type.".format(key))
                    raise

        assert(loss in self.accepted_losses, "loss must be one of {}".format(self.accepted_losses))

        return
