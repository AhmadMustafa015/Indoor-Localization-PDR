# Use scikit-learn to grid search the batch size and epochs
import numpy as np
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
import pandas as pd
from keras.constraints import maxnorm
from sklearn.model_selection import KFold
seed = 15
np.random.seed(seed)
filesname_test = list()
filesname_test += ['total_acc_x_test.txt', 'total_acc_y_test.txt', 'total_acc_z_test.txt']
filesname_test += ['body_acc_x_test.txt', 'body_acc_x_test.txt', 'body_acc_x_test.txt']
filesname_test += ['body_gyro_x_test.txt', 'body_gyro_y_test.txt', 'body_gyro_z_test.txt']
dataX_testing = list()
for name in filesname_test:
    frame = pd.read_csv('Dataset/test/Inertial Signals/' + name,header=None, delim_whitespace=True) # load a single file
    dataX_testing.append(frame.values) # store data as a numpy array
# dataX_training is a 3D numpy array (samples[number of raws is any file), time steps(number of readings in a single window ex: 128)
#                               ,features[nine features 3-axis accel, 3-axis gyro, 3-axis body accel]
dataX_testing = np.dstack(dataX_testing) # To stack each of the loaded 3D arrays into a single 3D array
dataY_testingRaw = pd.read_csv('Dataset/test/y_test.txt',header=None, delim_whitespace=True)
dataY_testingRaw = dataY_testingRaw.values
print(dataX_testing.shape, dataY_testingRaw.shape)
dataY_testing = dataY_testingRaw - 1
dataY_testing = tf.keras.utils.to_categorical(dataY_testing)
def RNN_model(optimizer='adam', init_mode='uniform', activation='relu',dropout_rate=0.5, weight_constraint=0,neurons=100):
##    numTimestep, numFeatures, numOutputs = dataX_testing.shape[1], dataX_testing.shape[2], dataY_testing.shape[1] 
##    model = tf.keras.models.Sequential()
##    model.add(tf.keras.layers.LSTM(neurons, input_shape= (numTimestep, numFeatures)))
##    model.add(tf.keras.layers.Dropout(dropout_rate))
##    model.add(tf.keras.layers.Dense(neurons, kernel_initializer=init_mode ,activation=activation,kernel_constraint=maxnorm(weight_constraint))) #  kernel_constraint=maxnorm(weight_constraint)
##    model.add(tf.keras.layers.Dense(numOutputs, kernel_initializer=init_mode , activation='softmax'))
##    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
##    return model
    numTimestep, numFeatures, numOutputs = dataX_testing.shape[1], dataX_testing.shape[2], dataY_testing.shape[1] 
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(100, input_shape= (numTimestep, numFeatures)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(numOutputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=RNN_model, verbose=1)
# define the grid search parameters
batch_size = [64,128]#, 128, 192, 256]
epochs = [3]#, 20, 30, 50]
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#param_grid = dict(optimizer=optimizer)
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
#param_grid = dict(init_mode=init_mode)
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
#param_grid = dict(activation=activation)
weight_constraint = [0,1, 2, 3, 4, 5, 6]
dropout_rate = [0.0,0.1,0.2, 0.35, 0.45 ,0.5, 0.6, 0.7]
#param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
neurons = [50, 100,128,175, 200, 250]
#param_grid = dict(neurons=neurons)
param_grid = dict(batch_size=batch_size, epochs=epochs)#, optimizer=optimizer,
                  #init_mode=init_mode,activation=activation,weight_constraint = weight_constraint,dropout_rate=dropout_rate,neurons=neurons)
#kfold = KFold(n_splits=10, shuffle=True)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(dataX_testing, dataY_testing)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
