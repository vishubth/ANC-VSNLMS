import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, InputLayer # type: ignore

class ANCModel:
    """
    Neural network-based adaptive noise cancellation model.
    
    Attributes:
        model (tf.keras.Model): The neural network model.
        optimizer (tf.keras.optimizers.Optimizer): The optimizer for training the model.
    """
    
    def __init__(self, input_shape, learning_rate=0.001):
        """
        Initialize the ANCModel with the given input shape and learning rate.
        
        Args:
            input_shape (tuple): The shape of the input to the model.
            learning_rate (float): The learning rate for the optimizer.
        """
        self.model = self.build_model(input_shape)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=self.optimizer, loss='mse')

    def build_model(self, input_shape):
        """
        Build and return the neural network model.
        
        Args:
            input_shape (tuple): The shape of the input to the model.
            
        Returns:
            tf.keras.Model: The constructed neural network model.
        """
        model = Sequential()
        model.add(InputLayer(input_shape=input_shape))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='linear'))
        return model

    def train(self, x, y, epochs=10, batch_size=32):
        """
        Train the model on the given data.
        
        Args:
            x (np.array): The input data.
            y (np.array): The target data.
            epochs (int): The number of epochs to train.
            batch_size (int): The size of training batches.
        """
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2)

    def save_weights(self, filename):
        """
        Save the model weights to a file.
        
        Args:
            filename (str): The path to the file where weights will be saved.
        """
        self.model.save_weights(filename)

    def load_weights(self, filename):
        """
        Load the model weights from a file.
        
        Args:
            filename (str): The path to the file from which weights will be loaded.
        """
        self.model.load_weights(filename)

    def save_tflite(self, filename):
        """
        Save the model in TensorFlow Lite format.
        
        Args:
            filename (str): The path to the file where the TFLite model will be saved.
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        with open(filename, 'wb') as f:
            f.write(tflite_model)

    def predict(self, x):
        """
        Predict the output for the given input data.
        
        Args:
            x (np.array): The input data.
            
        Returns:
            np.array: The predicted output.
        """
        return self.model.predict(x)
