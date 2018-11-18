'''
TensorFlow Base Network structure all will inherit from
'''

import copy
import pathlib

try:
    import tensorflow as tf
except ImportError:
    print("Couldn't import Tensorflow. Continuing to allow pydoc parsing.")

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
            *optional*
                'random_seed*: int
        '''
        self.required_keys = [
            'debug',
            'gpu_settings',
            'results_dir',
        ]
        self.req_sub_keys = {
            'gpu_settings': ['GPU', 'gpu_frac'],
        }
        self.optional_keys = [
            'random_seed',
        ]
        self.config = copy.deepcopy(config)
        self.parse_config(config)

        self.random_seed = -1
        if 'random_seed' in self.config:
            self.random_seed = self.config['random_seed']
        self.results_dir = self.config['results_dir']
        pathlib.Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        self.learning_rate = self.config['learning_rate']
        self.max_epoch = self.config['max_epoch']

        self.graph = tf.Graph()

        gpu_settings = self.config['gpu_settings']
        gpu_frac = gpu_settings['gpu_frac']
        GPU = gpu_settings['GPU']

        gpu_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                per_process_gpu_memory_fraction=gpu_frac),
            device_count={'GPU': GPU}
        )

        self.sess = tf.Session(config=gpu_config, graph=self.graph)
        self.summary_writer = tf.summary.FileWriter(self.results_dir,
                                                    self.sess.graph)

    def build(self, settings=None):
        '''
        Child must overwrite.
        '''
        raise NotImplementedError("The build() function must be overwritten" +
                                  "completely")

    def train(self, settings):
        '''
        Child must overwrite.
        '''
        raise NotImplementedError("The train() function must be overwritten" +
                                  "completely")

    def save(self):
        '''
        Save graph using summary writer.
        '''
        raise NotImplementedError("Need to write this function")

    def visualize_tensorboard(self):
        '''
        Child must overwrite
        '''
        raise NotImplementedError("The visualize_tensorboard() function " +
                                  "must be overwritten completely")

    def child_parse_config(self, config):
        '''
        Child must overwrite.
        '''
        raise NotImplementedError("The child_parse_config() function must " +
                                  "be overwritten completely")

    def parse_config(self):
        '''
        Parses the config file to assert validity.
        '''
        for key in self.required_keys:
            assert (key in self.config), "Missing key '{}' in config"\
                    .format(key)
            if key in self.req_sub_keys:
                for sub_key in self.req_sub_keys[key]:
                    assert (sub_key in self.config[key]),\
                        "Missing key '{}' in 'config[{}]'".format(sub_key, key)

        for key in self.optional_keys:
            if key not in self.config:
                warn("Optional key '{}' not found in 'config'.".format(key))
