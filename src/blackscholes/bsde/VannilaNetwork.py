import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Input, Lambda

class SingleTimeStepNetwork:

    def __init__(self, dt, layer_dim_vec, last_opt_price):
        """
        Note that layer_dim_vec[0] == len(np.concatenate(stock_price_vec, np.array([payoff, last_opt_price])))
        """
        self.layer_num = len(layer_dim_vec) # 0, 1, ..., layer_num-1
        self.layer_dim_vec = layer_dim_vec
        
        self.model = Sequential(Input(shape=(layer_dim_vec[0],)))
        for i in range(self.layer_num-1):
            self.model.add(Dense(layer_dim_vec[i+1]))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
        self.model.add(Dense(1))
        self.model.add(Lambda(lambda x: x * dt))
        self.model.add(Lambda(lambda x: x + keras.backend.constant(last_opt_price)))
        self.model.add(Dense(1, use_bias=False))

    def single_train(self, stock_price_vec, payoff, last_opt_price):
        concat_input = np.concatenate(stock_price_vec, np.array([payoff, last_opt_price]))




        
