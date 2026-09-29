import tensorflow as tf
from tensorflow.keras import layers, Model

# Define the model using subclassing
class MyModel(Model):
    def __init__(self, activation="relu", **kwargs):
        super().__init__(**kwargs) # needed to support naming the model

        # The layers will be defined in the build method

    def build(self, input_shape):
        super().build(input_shape)
        # Define layers in the build method
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(200, activation='relu')
        self.dense2 = layers.Dense(50, activation='relu')
        self.dense3 = layers.Dense(50, activation='relu')
        self.dense4 = layers.Dense(50, activation='relu')
        self.dense5 = layers.Dense(50, activation='relu')
        self.output_layer = layers.Dense(10)
        
    # Define the forward pass with a skip connection
    def call(self, inputs):
        # Flatten input
        x = self.flatten(inputs)
        
        # Pass through the first two dense layers
        x1 = self.dense1(x)
        x2 = self.dense2(x1)
        
        # Skip connection: Concatenate the original flattened input with the output of dense2
        x_skip = tf.concat([x, x2], axis=-1)
        
        # Continue through the remaining layers
        x3 = self.dense3(x_skip)
        x4 = self.dense4(x3)
        x5 = self.dense5(x4)
        
        # Output layer
        return self.output_layer(x5)

# Create the model instance
model = MyModel()

# Build the model and print the summary
model.build(input_shape=(None, 28, 28))
model.summary()

