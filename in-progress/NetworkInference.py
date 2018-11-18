'''
TensorFlow Network inferencing, will multiprocessing capabilities
'''
import os


class NetworkInference():
    def __init__(self, GPU, model):
        self.GPU = GPU
        self.sess = None
        self.graph = None

    def run(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(self.GPU)
        # import tensorflow as tf
        return
