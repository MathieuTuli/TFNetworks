'''
TensorFlow Base Network structure all will inherit from
'''

import copy
import tensorflow as tf

from warning import warn


class NetworkBase():
    def __init__(self, config: dict):
        self.random_seed = -1
        self.parse_config(config)
        self.child_parse_config(config)

        self.config = copy.deepcopy(config)
        self.results_dir = self.config['results_dir']
        self.learning_rate = self.config['learning_rate']
        self.max_iter = self.config['max_iter']

        self.child_parameters()

        self.layers = list()
        self.graph = tf.Graph()

        gpu_settings = config['gpu_settings']
        gpu_frac = gpu_settings['gpu_frac']
        GPU = gpu_settings['GPU']

        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                per_process_gpu_memory_fraction=gpu_frac),
            device_count={'GPU': GPU}
        )

        self.sess = tf.Session(config=config, graph=self.graph)
        self.summary_writer = tf.summary.FileWriter(self.results_dir,
                                                    self.sess.graph)

    def build(self):
        raise Exception("The build() function must be overwritten completely")

    def train(self, save_every=-1):
        for epoch in range(0, self.max_iter):
            self.learn_from_epoch()
            if save_every > 0 and epoch % save_every == 0:
                self.save()

    def learn_from_epoch(self):
        raise Exception("The learn_from_epoch() function must be overwritten\
            completely")

    def save(self):
        raise Exception("Need to write this function")

    def parse_config(self, config):
        required_keys = [
            'debug',
            'gpu_settings',
            'results_dir',
            'max_iter',
            'learning_rate',
        ]

        req_sub_keys = {
            'gpu_settings': ['GPU', 'gpu_frac'],
        }

        for key in required_keys:
            assert (key in config), "Missing key '{}' in config".format(key)
            if key in req_sub_keys:
                for sub_key in req_sub_keys[key]:
                    assert (sub_key in config[key]),\
                        "Missing key {} in config[{}]".format(sub_key, key)

        if 'random_seed' not in config:
            warn("It is good practice, for repeatability, to reuse the same\
                random seed during development.")
        else:
            self.random_seed = config['random_seed']

        if config['debug']:
            print("Debug mode.", config)

    def child_parse_config(self, config);
        raise Exception("The child_parse_config() function must be overwritten\
            completely")
