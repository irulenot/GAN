from __future__ import absolute_import, division, print_function

import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt


from tensorflow import keras
from tensorflow.keras import layers


print(tf.__version__)  # *** Print TF version ***


# Solves TF issue on mac
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# keras.utils.get_file: Downloads a file from a URL if it not already in the cache.
dataset_path = keras.utils.get_file("auto-mpg.data",
                                    "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print(dataset_path)  # *** Print dataset path ***

# read_csv: reads csv & returns dataframe
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values="?", comment='\t',
                          sep=" ", skipinitialspace=True)

# Make copy of database & copy last couple rows
dataset = raw_dataset.copy()
print(dataset.tail())  # *** Print last couple rows ***

# Dropping NaN data (For simplicity in data)
print(dataset.isna().sum())  # *** Print out count of NaN data ***
dataset = dataset.dropna()

# Pop category Origin & Add new columns to represent true if car is from that nation
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0
print(dataset.tail())  # *** Print out new tailend of dataset ***

# Create training (80%) and testing datasets (20%)
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)  # Drops indexes found in train

# sns.pairplot: Draw scatterplots for joint relationships and histograms for univariate distributions
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")  # Produces 16 graphs 4x4
# plt.show()                                # *** Outputs generated graphs ***
plt.close('all')


# Printing out training data statistics
train_stats = train_dataset.describe()  # Returns general data in dataframe statistics
train_stats.pop("MPG")
train_stats = train_stats.transpose()  # We do this so not visually confusing
print(train_stats)


# MPG removed as is target value / label
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


# Normalizing stats data so it's easier for model to train
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# Create model by defining its layers and its optimizer
def build_model():
    model = keras.Sequential([  # Sequential model runs layers in order
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        # Model has 2 relu layers & output
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)  # 0.001 is learning rate

    # .compile: Configures the model for training
    model.compile(loss='mse',  # Loss function is mean-squared-error
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])   # Metrics for us to look at - not used for training

    return model

model = build_model()
print(model.summary()) # *** Print out simple description of model ***


# Display training progress by printing a single dot for each completed epoch
# class PrintDot(keras.callbacks.Callback): # keras.callbacks.Callback: Are used to view internal states of model during trianing
#   def on_epoch_end(self, epoch, logs):
#     if epoch % 100 == 0: print('')
#     print('.', end='')

EPOCHS = 1000

# model.fit: returns a history object
print("Training model")
history = model.fit( # fit used to train model
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0) # verbose indicates output option during training
  # callbacks=[PrintDot()])   # *** Prints out training of model ***


# Turning history obj in to dataframe
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())  # *** Prints some training history epoch data ***


# Print out two figures between validation and training errors
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.legend()
    plt.ylim([0, 5])

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.legend()
    plt.ylim([0, 20])

plot_history(history)
plt.close('all')
# plt.show()


# Recreating trained model with early stop
model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# Callbacks applied during model training, enacted during training
print("Training early stop model")
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop])

plot_history(history)
# plt.show()
plt.close('all')


# Testing model
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))


# Get predictions and graph them against true values
test_predictions = model.predict(normed_test_data).flatten()    # returns numpy array then flattens to array

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]]) # xlim sets left to 0, right to limit of right most value
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100]) # plots line
# plt.show()
plt.close("all")


# Error histogram
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()