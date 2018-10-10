import os
import copy

class NeuralNetwork(object):
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
        return
    def check_config(self, config):
        return
