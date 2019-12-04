# Reference: https://github.com/adolfocorreia/DGM and https://github.com/alialaradi/DeepGalerkinMethod

import tensorflow as tf

class DGMNet(tf.keras.Model):
    
    def __init__(self, n_layers, layer_width, input_dim):
        '''
        Args:
            layer_width: layer's width
            n_layers:    number of intermediate LSTM layers
            input_dim:   spaital dimension of input data (EXCLUDES time dimension)
        
        Returns: customized Keras model object representing DGM neural network
        '''  
        super(DGMNet, self).__init__()
        
        # define initial layer as fully connected 
        # NOTE: to account for time inputs we use input_dim+1 as the input dimensionality
        self.initial_layer = DenseLayer(input_dim+1, layer_width, activation="tanh")
        
        # define intermediate LSTM layers
        self.n_layers = n_layers
        self.LSTMLayerList = []

        for _ in range(self.n_layers):
            self.LSTMLayerList.append(LSTMLayer(input_dim+1, layer_width, activation="tanh"))
        
        # define final layer as fully connected with a single output (function value)
        self.final_layer = DenseLayer(layer_width, 1, activation=None)
    
    def call(self, x, t):
        '''
        Args:
            t: sampled time inputs 
            x: sampled space inputs
        Run the DGM model and obtain fitted function value at the inputs (t,x)                
        '''  
        # define input vector as time-space pairs
        X = tf.concat([t, x], 1)
        S = self.initial_layer.call(X)
        for i in range(self.n_layers):
            S = self.LSTMLayerList[i].call(S, X)
        result = self.final_layer.call(S)
        return result

class LSTMLayer(tf.keras.layers.Layer):
    def __init__(self, n_inputs, n_outputs, activation):
        """
        Parameters:
            - n_inputs:     number of inputs
            - n_outputs:    number of outputs
            - activation:   activation function
        """
        super(LSTMLayer, self).__init__()

        self.n_outputs = n_outputs
        self.n_inputs = n_inputs

        self.Uz = self.add_variable("Uz", shape=[self.n_inputs, self.n_outputs], dtype=tf.float64,
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Ug = self.add_variable("Ug", shape=[self.n_inputs, self.n_outputs], dtype=tf.float64,
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Ur = self.add_variable("Ur", shape=[self.n_inputs, self.n_outputs], dtype=tf.float64,
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Uh = self.add_variable("Uh", shape=[self.n_inputs, self.n_outputs], dtype=tf.float64,
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Wz = self.add_variable("Wz", shape=[self.n_outputs, self.n_outputs], dtype=tf.float64,
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Wg = self.add_variable("Wg", shape=[self.n_outputs, self.n_outputs], dtype=tf.float64,
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Wr = self.add_variable("Wr", shape=[self.n_outputs, self.n_outputs], dtype=tf.float64,
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Wh = self.add_variable("Wh", shape=[self.n_outputs, self.n_outputs], dtype=tf.float64,
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.bz = self.add_variable("bz", shape=[1, self.n_outputs], dtype=tf.float64)
        self.bg = self.add_variable("bg", shape=[1, self.n_outputs], dtype=tf.float64)
        self.br = self.add_variable("br", shape=[1, self.n_outputs], dtype=tf.float64)
        self.bh = self.add_variable("bh", shape=[1, self.n_outputs], dtype=tf.float64)

        self.activation = _get_function(activation)

    def call(self, S, X):
        Z = self.activation(tf.add(tf.add(tf.matmul(X, self.Uz), tf.matmul(S, self.Wz)), self.bz))
        G = self.activation(tf.add(tf.add(tf.matmul(X, self.Ug), tf.matmul(S, self.Wg)), self.bg))
        R = self.activation(tf.add(tf.add(tf.matmul(X, self.Ur), tf.matmul(S, self.Wr)), self.br))
        H = self.activation(tf.add(tf.add(tf.matmul(X, self.Uh), tf.matmul(tf.multiply(S, R), self.Wh)), self.bh))
        Snew = tf.add(tf.multiply(tf.subtract(tf.ones_like(G), G), H), tf.multiply(Z, S))
        return Snew

class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, n_inputs, n_outputs, activation):
        """
        Parameters:
            - n_inputs:     number of inputs
            - n_outputs:    number of outputs
            - activation:   activation function
        """
        super(DenseLayer, self).__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
        self.W = self.add_variable("W", shape=[self.n_inputs, self.n_outputs], dtype=tf.float64,
                                   initializer=tf.contrib.layers.xavier_initializer())
        self.b = self.add_variable("b", shape=[1, self.n_outputs])

        self.activation = _get_function(activation)
    
    def call(self, inputs):
        S = tf.add(tf.matmul(inputs, self.W), self.b)
        return self.activation(S)

def _get_function(name):
    f = None
    if name == "tanh":
        f = tf.nn.tanh
    elif name == "sigmoid":
        f = tf.nn.sigmoid
    elif name == "relu":
        f = tf.nn.relu
    elif not name:
        f = tf.identity
    assert f is not None
    return f