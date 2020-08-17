import numpy as np
import tensorflow as tf

class column_selector_layer(tf.keras.layers.Layer):
    def __init__(self,lag,embed,n_var):
        super(column_selector_layer, self).__init__()
        self.lag = lag
        self.embed = embed
        self.n_var = n_var

        indices = []
        for i in range(n_var):
            indices = indices + [np.arange(0, lag*embed*n_var, lag*n_var) + i]
        self.columns = np.sort(np.hstack(indices))

    def build(self,input_shape):
        super(column_selector_layer, self).build(input_shape)

    def compute_output_shape(self,input_shape):
        return tf.TensorShape([input_shape[0],len(self.columns)])

    def call(self,inputs):

        layer_output = tf.gather(inputs, self.columns)
        return layer_output

