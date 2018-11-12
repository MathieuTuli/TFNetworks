'''
TensorFlow Base Network structure all will inherit from
'''

import copy
import pathlib
import tensorflow as tf

from warnings import warn


class NetworkBase():
    '''
    NetworkBase class
    Common characteristics of neural networks
    '''
    def __init__(self, config: dict):
        '''
        config: dict | has some required and optional keys
            *required*
                'debug': boolean
                'gpu_settings': dict | has some required keys
                    *required*
                        'gpu_frac': float | gpu memory fraction
                        'GPU': int | GPU device number
                'results_dir': str
                'max_epoch': int | for training. epochs [0 -> max_epoch]
                'learning_rate': float
            *optional*
                'random_seed*: int
        '''
        self.random_seed = -1
        self.parse_config(config)
        self.child_parse_config(config)

        self.config = copy.deepcopy(config)
        self.results_dir = self.config['results_dir']
        pathlib.Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        self.learning_rate = self.config['learning_rate']
        self.max_epoch = self.config['max_epoch']

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

    def build(self, config):
        '''
        Child must overwrite.
        '''
        raise Exception("The build() function must be overwritten completely")

    def train(self, settings):
        '''
        Child must overwrite.
        '''
        raise Exception("The train() function must be overwritten\
            completely")

    def save(self):
        '''
        '''
        raise Exception("Need to write this function")

    def parse_config(self, config):
        '''
        Parses the config file to assert validity.
        '''
        required_keys = [
            'debug',
            'gpu_settings',
            'results_dir',
            'max_epoch',
            'learning_rate',
            'placeholders',
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

    def visualize_tensorboard(self):
        '''
        Child must overwrite
        '''
        raise Exception("The visualize_tensorboard() function must be\
            overwritten completely")

    def child_parse_config(self, config):
        '''
        Child must overwrite.
        '''
        raise Exception("The child_parse_config() function must be overwritten\
            completely")
