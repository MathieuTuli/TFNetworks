import tensorflow as tf

def train(self,
          train_size,
          batch_generator,
          save_every=-1):
    '''
    Train your TensorFlow graph

    @param save_every: int | default = -1. Set to -1 to not save, else
        will save every 'save_every' epochs between 0 and max_epoch
    @param train_size: int | size of training data x
    @param batch_generator: function
        *signature*
            batch_generator(batch_start, batch_end)
                ...
                return x_batch, y_true_batch
    '''
    start_time = time.time()
    num_batches = int(train_size / self.batch_size)
    metric = self.config['accuracy']['accuracy_metric']
    cost_input = self.config['cost']['layer_input']
    cost = self.cost(layer_input=self.layers[cost_input],
                     y_true=self.y_true,
                     cost=self.config['cost']['cost'],
                     cost_aggregate=self.config['cost']['cost_aggregate'])
    optimizer = self.optimizer(cost=cost,
                               optimization=self.config['optimizer']['optimization'])

    with self.graph.as_default():
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(0, self.max_epoch):
            if save_every > 0 and epoch % save_every == 0:
                self.save()
            for batch in range(num_batches):
                batch_time = time.time()
                batch_start = self.batch_size * batch
                batch_end = batch_start + self.batch_size
                x_batch, y_true_batch = batch_generator(batch_start,
                                                        batch_end)
                x_batch = np.reshape(x_batch, [-1,
                                               self.img_width,
                                               self.img_height,
                                               self.num_channels])
                feed_dict_train = {self.x: x_batch,
                                   self.y_true: y_true_batch}
                self.sess.run(optimizer, feed_dict=feed_dict_train)

                if not batch % 5:
                    accuracy = self.calculate_accuracy(
                        y_pred_cls=self.layers['y_pred_cls'],
                        accuracy_metric=metric)
                    acc = self.sess.run(accuracy,
                                        feed_dict=feed_dict_train)
                    progress = "[" + "=" * int(20*batch/num_batches) + " " *\
                        int(20 - (20*batch/num_batches)) + "]"
                    print(progress)
                    print("Accuracy: {:.3}".format(acc))
                    print("Batch Time: {:.3}".format(time.time() - batch_time))
                    print("Total Time: {:.3}".format(time.time() - start_time))

def cost(self,
         layer_input,
         y_true,
         cost='cross_entropy',
         cost_aggregate='reduce_mean'):
    '''
    Cost function to be optimized

    @para layer_input: Tensorflow Tensor
    @param y_true: Tensorflow Placeholder
    @param cost: str | Type of Tensorflow loss function
        *options* -> ['cross_entropy']
    @param cost_aggregate: str | Type of tf.math aggregate
        *options* -> ['reduce_mean']

    @return cost/loss: Tensorflow Tensor
    '''
    with self.graph.as_default():
        if cost == 'cross_entropy':
            cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_input,
                                                              labels=y_true)
        else:
            raise ValueError("Unknown loss function")

        if cost_aggregate == 'reduce_mean':
            cost_aggregate = tf.reduce_mean(cost)
        else:
            raise ValueError("Unknown cost function")

    return cost_aggregate

def optimizer(self,
              cost,
              optimization='adam'):
    '''
    Optimization method

    @param cost: TensorFlow loss | loss to minimize
    @param optimization: str | Type of Tensorflow Optimizer
        *options* -> ['adam']

    @return optimizer: TensorFlow Optimizer, defined by @param optimizer
    '''
    with self.graph.as_default():
        if optimization == 'adam':
            optimizer = tf.train.\
                AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        else:
            raise ValueError("Unknown optimization method")
    return optimizer

def calculate_accuracy(self,
                       y_pred_cls,
                       accuracy_metric='reduce_mean'):
    '''
    Performance Measures

    @param y_pred_cls: TensorFlow variable
    @param y_true_cls: TensorFlow variable
    @param accuracy_metric: str | ['reduce_mean']

    @return accurac: Tensorflow Tensor, define by @param accuracy_metric
    '''
    correct_prediction = tf.equal(y_pred_cls, self.y_true_cls)

    if accuracy_metric == 'reduce_mean':
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, self.precision))
    else:
        raise ValueError("Unkown accuracy metric")
    return accuracy
