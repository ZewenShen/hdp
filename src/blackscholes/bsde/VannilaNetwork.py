import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Input, Lambda

class SingleTimeStepNetwork:

    def __init__(self, dt, layer_dim_vec, last_opt_price, ir, vol_vec):
        """
        Note that layer_dim_vec[0] == len(np.concatenate(stock_price_vec, np.array([payoff, last_opt_price])))
        """
        self.layer_num = len(layer_dim_vec) # 0, 1, ..., layer_num-1
        self.layer_dim_vec = layer_dim_vec
        self.model = Sequential()
        for i in range(self.layer_num-1):
            self.model.add(Dense(layer_dim_vec[i+1], input_dim=layer_dim_vec[i]))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
        self.model.add(Dense(1))
        self.model.add(Lambda(lambda x: x * dt))
        self.model.add(Lambda(lambda x: x + keras.backend.constant(last_opt_price)))
        self.model.add(Dense(1, use_bias=False))

        def bsde_loss(y_true, y_pred):
            dq = K.gradients(self.model.output, self.model.input)[:-2]

    def single_train(self, stock_price_vec, payoff, last_opt_price):
        concat_input = np.concatenate(stock_price_vec, np.array([payoff, last_opt_price]))




        
