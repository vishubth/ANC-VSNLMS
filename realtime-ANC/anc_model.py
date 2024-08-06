# --------------------------------------------------------------------------------------------------------------
# 
# Author - Vishal Shrivastava
# 
# ---------------------------------------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input

class ANCModel:
    def __init__(self, input_shape, learning_rate=0.001):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.model = self.build_model(input_shape)
        self.compile_model()

    def build_model(self, input_shape):
        """
        Build and return the neural network model.
        
        Args:
            input_shape (tuple): The shape of the input to the model.
            
        Returns:
            tf.keras.Model: The constructed neural network model.
        """
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(Conv1D(64, kernel_size=5, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.3))

        model.add(Conv1D(128, kernel_size=5, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='linear'))

        return model

    def compile_model(self):
        """
        Compile the neural network model with the specified loss and optimizer.
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse')

    def train(self, x_train, y_train, epochs, batch_size, validation_data=None):
        """
        Train the model with the given training data.
        
        Args:
            x_train (np.array): Training input data.
            y_train (np.array): Training target data.
            epochs (int): Number of epochs to train.
            batch_size (int): Size of each training batch.
            validation_data (tuple): Validation data for monitoring the model's performance.
        """
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def save_weights(self, file_path):
        """
        Save the model's weights to the specified file path.
        
        Args:
            file_path (str): Path to the file where the weights will be saved.
        """
        self.model.save_weights(file_path)

    def load_weights(self, file_path):
        """
        Load the model's weights from the specified file path.
        
        Args:
            file_path (str): Path to the file where the weights are stored.
        """
        self.model.load_weights(file_path)

    def save_tflite(self, file_path):
        """
        Convert the model to TFLite format and save it to the specified file path.
        
        Args:
            file_path (str): Path to the file where the TFLite model will be saved.
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        with open(file_path, 'wb') as f:
            f.write(tflite_model)

    def predict(self, x):
        """
        Make predictions using the model.
        
        Args:
            x (np.array): Input data for making predictions.
        
        Returns:
            np.array: Model predictions.
        """
        return self.model.predict(x)
