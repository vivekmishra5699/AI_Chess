import tensorflow as tf
from tensorflow import keras

class ChessNet(keras.Model):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv_layers = [
            keras.layers.Conv2D(128, 3, activation='relu', padding='same', input_shape=(8, 8, 119))
        ] + [
            keras.layers.Conv2D(128, 3, activation='relu', padding='same') 
            for _ in range(8)
        ]
        self.flatten = keras.layers.Flatten()
        self.policy_dense = keras.layers.Dense(4672, activation='softmax')
        self.value_dense = keras.layers.Dense(1, activation='tanh')

    def call(self, inputs):
        x = inputs
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = self.flatten(x)
        policy = self.policy_dense(x)
        value = self.value_dense(x)
        return policy, value
