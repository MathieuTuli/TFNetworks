import os
import copy

class FFNN(object):
    def __init__(self, config):
        print('\n\n')
        self.config = copy.deepcopy(config)
        if config['debug']:
            print("Current Configuration\n",self.config)

        print("Checking config.\n")
        self.check_config(config)
        print("Config checked.\n")

        print("Building network.\n")
        self.graph, self.sess = self.build()
        print("Network built.\n")



    def build(self):
        import tensorflow as tf
        config = tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = settings['gpu_frac']),
            device_count = {'GPU': settings['GPU']}
        )
        self.sess = tf.Session(config=config, graph=self.graph)
        with self.graph.as_default():
            print("\n\nBuilding network.\n")



    def check_config(self, config):
        return
