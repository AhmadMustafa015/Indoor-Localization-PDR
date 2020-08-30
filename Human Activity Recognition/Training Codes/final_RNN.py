# Load the TensorBoard notebook extension
from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['TF_ENABLE_CONTROL_FLOW_V2'] = '1'
import pandas as pd
import numpy as np
from numpy import std
from numpy import mean
from numpy import dstack
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sn
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from tensorflow.keras.regularizers import l2
import datetime
mpl.rcParams['axes.grid'] = False
num_runs = 1 # Total number of repeats
filesname_train = list()
#file name of train data
filesname_train += ['total_acc_x_train.txt', 'total_acc_y_train.txt', 'total_acc_z_train.txt']
filesname_train += ['body_acc_x_train.txt', 'body_acc_x_train.txt', 'body_acc_x_train.txt']
filesname_train += ['body_gyro_x_train.txt', 'body_gyro_y_train.txt', 'body_gyro_z_train.txt']
dataY_trainingRaw = pd.read_csv('Dataset/train/y_train.txt',header=None, delim_whitespace=True)
index1 = dataY_trainingRaw[dataY_trainingRaw[0] == 6].index
# indexStanding = dataY_trainingRaw[dataY_trainingRaw[0] == 5].index
# indexSitting = dataY_trainingRaw[dataY_trainingRaw[0] == 4].index
# #print(index1.shape)
# templist = dataY_trainingRaw.index.tolist()
# idx = templist.index(5)
# templist[idx] = 4
# dataY_trainingRaw.index = templist;
dataY_trainingRaw.drop(index= index1, inplace = True)
dataY_trainingRaw.replace(5,4,inplace = True)
print(dataY_trainingRaw.max())
dataX_training = list()
for name in filesname_train:
    frame = pd.read_csv('Dataset/train/Inertial Signals/' + name,header=None, delim_whitespace=True) # load a single file
    frame.drop(index= index1, inplace = True)
    dataX_training.append(frame.values) # store data as a numpy array
# dataX_training is a 3D numpy array (samples[number of raws is any file), time steps(number of readings in a single window ex: 128)
#                               ,features[nine features 3-axis accel, 3-axis gyro, 3-axis body accel]
dataX_training = dstack(dataX_training) # To stack each of the loaded 3D arrays into a single 3D array

dataY_trainingRaw = dataY_trainingRaw.values
#print(dataX_training.shape, dataY_trainingRaw.shape)
#file name of test data
filesname_test = list()
filesname_test += ['total_acc_x_test.txt', 'total_acc_y_test.txt', 'total_acc_z_test.txt']
filesname_test += ['body_acc_x_test.txt', 'body_acc_x_test.txt', 'body_acc_x_test.txt']
filesname_test += ['body_gyro_x_test.txt', 'body_gyro_y_test.txt', 'body_gyro_z_test.txt']
dataY_testingRaw = pd.read_csv('Dataset/test/y_test.txt',header=None, delim_whitespace=True)
index2 = dataY_testingRaw[dataY_testingRaw[0] == 6].index
# indexStandingT = dataY_testingRaw[dataY_testingRaw[0] == 5].index
# indexSittingT = dataY_testingRaw[dataY_testingRaw[0] == 4].index
# #print(index1.shape)
# dataY_testingRaw[indexStandingT,:] = 4;
dataY_testingRaw.drop(index= index2, inplace = True)
dataY_testingRaw.replace(5,4,inplace = True)

#print(index2.shape)
dataX_testing = list()
for name in filesname_test:
    frame = pd.read_csv('Dataset/test/Inertial Signals/' + name,header=None, delim_whitespace=True) # load a single file
    frame.drop(index= index2, inplace = True)
    dataX_testing.append(frame.values) # store data as a numpy array
# dataX_training is a 3D numpy array (samples[number of raws is any file), time steps(number of readings in a single window ex: 128)
#                               ,features[nine features 3-axis accel, 3-axis gyro, 3-axis body accel]
dataX_testing = dstack(dataX_testing) # To stack each of the loaded 3D arrays into a single 3D array
dataY_testingRaw = dataY_testingRaw.values
#print(dataX_testing.shape, dataY_testingRaw.shape)
dataY_training = dataY_trainingRaw - 1
dataY_testing = dataY_testingRaw - 1
dataY_training = tf.keras.utils.to_categorical(dataY_training, num_classes = 5)
dataY_testing = tf.keras.utils.to_categorical(dataY_testing, num_classes = 5)
#print(dataX_training.shape, dataY_training.shape,dataX_testing.shape, dataY_testing.shape)
# import serhat dataset
filesname_serhat = list()
filesname_serhat += ['total_acc_x.csv', 'total_acc_y.csv', 'total_acc_z.csv']
filesname_serhat += ['body_acc_x.csv', 'body_acc_x.csv', 'body_acc_x.csv']
filesname_serhat += ['body_gyro_x.csv', 'body_gyro_y.csv', 'body_gyro_z.csv']
# indexStandingS = dataY_testingRaw[dataY_testingRaw[0] == 5].index
# indexSittingS = dataY_testingRaw[dataY_testingRaw[0] == 4].index
# #print(index1.shape)
# dataY_testingRaw[indexStandingS,:] = 4;
dataX_serhat = list()
for name in filesname_serhat:
    frame = pd.read_csv('New Data/After processing/' + name, header=None) # load a single file
    dataX_serhat.append(frame.values) # store data as a numpy array
dataX_serhat = dstack(dataX_serhat)
dataY_serhat = pd.read_csv('New Data/After processing/labels.csv',header=None)
dataY_serhat.replace(6,5,inplace = True)
dataY_serhat = dataY_serhat.values
#print(dataX_serhat.shape, dataY_serhat.shape)
dataY_serhat = dataY_serhat - 1
dataY_serhat_T = tf.keras.utils.to_categorical(dataY_serhat, num_classes = 5)
#print(dataX_serhat.shape, dataY_serhat_T.shape)
LABELS = [
    "WALKING", 
    "WALKING_UPSTAIRS", 
    "WALKING_DOWNSTAIRS", 
    "SITTING_SITTING",
    "RUNNING"
]
# definetion of RNN model
def RNN_model(dataX_training,dataY_training,dataX_testing,dataY_testing):
    verbose, epochs, batch_size = 1, 1, 128
    numTimestep, numFeatures, numOutputs = dataX_training.shape[1], dataX_training.shape[2], dataY_training.shape[1] 
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(128,input_shape= (numTimestep, numFeatures)))
    #model.add(tf.keras.layers.LSTM(64))return_sequences=True ,
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(numOutputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit (dataX_training, dataY_training, epochs=epochs, batch_size=batch_size, verbose=verbose)
    _, accuracy = model.evaluate(dataX_testing, dataY_testing, batch_size=batch_size, verbose=0)
    return accuracy, model
def RNN_model_V2 (dataX_training,dataY_training,dataX_testing,dataY_testing):
    verbose, epochs, batch_size = 1, 50, 128
    numTimestep, numFeatures, numOutputs = dataX_training.shape[1], dataX_training.shape[2], dataY_training.shape[1]
    N_CLASSES = dataY_training.shape[1]
    N_HIDDEN_UNITS = 200
    L2 = 0.000001
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(N_HIDDEN_UNITS,return_sequences=True,input_shape= (numTimestep, numFeatures),
                                   kernel_initializer='orthogonal', kernel_regularizer=l2(L2), recurrent_regularizer=l2(L2),
                                   bias_regularizer=l2(L2), name="LSTM_1"))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten(name='Flatten'))
    model.add(tf.keras.layers.Dense(N_HIDDEN_UNITS, activation='relu', kernel_regularizer=l2(L2), bias_regularizer=l2(L2), name="Dense_1"))
    model.add(tf.keras.layers.Dense(N_CLASSES, activation='softmax', kernel_regularizer=l2(L2), bias_regularizer=l2(L2), name="Dense_2"))
    model.summary()
    opt = tf.keras.optimizers.Adam(lr=0.0001)

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    BATCH_SIZE = 128
    N_EPOCHS = 50
    log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(X_train, y_train,
              batch_size=BATCH_SIZE, epochs=N_EPOCHS,
              validation_data=(X_test, y_test),
              callbacks=[tensorboard_callback])
    _, accuracy = model.evaluate(dataX_testing, dataY_testing, batch_size=BATCH_SIZE, verbose=0)
    return accuracy, model
def buildLstmLayer(inputs, num_layers, num_units):
  """Build the lstm layer.

  Args:
    inputs: The input data.
    num_layers: How many LSTM layers do we want.
    num_units: The unmber of hidden units in the LSTM cell.
  """
  lstm_cells = []
  for i in range(num_layers):
    lstm_cells.append(
        tf.lite.experimental.nn.TFLiteLSTMCell(
            num_units, forget_bias=0, name='rnn{}'.format(i)))
  lstm_layers = tf.keras.layers.StackedRNNCells(lstm_cells)
  # Assume the input is sized as [batch, time, input_size], then we're going
  # to transpose to be time-majored.
  transposed_inputs = tf.transpose(
      inputs, perm=[1, 0, 2])
  outputs, _ = tf.lite.experimental.nn.dynamic_rnn(
      lstm_layers,
      transposed_inputs,
      dtype='float32',
      time_major=True)
  unstacked_outputs = tf.unstack(outputs, axis=0)
  return unstacked_outputs[-1]
def RNN_model_V3 (dataX_training,dataY_training,dataX_testing,dataY_testing):
    verbose, epochs, batch_size = 1, 2, 128
    numTimestep, numFeatures, numOutputs = dataX_training.shape[1], dataX_training.shape[2], dataY_training.shape[1]
    N_CLASSES = dataY_training.shape[1]
    N_HIDDEN_UNITS = 200
    L2 = 0.000001
    tf.reset_default_graph()
    model = tf.keras.models.Sequential([
      tf.keras.layers.Input(shape=(numTimestep, numFeatures), name='input'),
      tf.keras.layers.Lambda(buildLstmLayer, arguments={'num_layers' : 2, 'num_units' : N_HIDDEN_UNITS}),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(N_HIDDEN_UNITS, activation='relu', kernel_regularizer=l2(L2), bias_regularizer=l2(L2), name="Dense_1"),
      tf.keras.layers.Dense(numOutputs, activation=tf.nn.softmax,kernel_regularizer=l2(L2), bias_regularizer=l2(L2), name='output')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(X_train, y_train,
              batch_size=batch_size, epochs=epochs,
              validation_data=(X_test, y_test),
              callbacks=[tensorboard_callback],verbose =verbose)
    _, accuracy = model.evaluate(dataX_testing, dataY_testing, batch_size=batch_size, verbose=0)
    return accuracy, model


totalResultAccuracy = list()
inputX = np.concatenate((dataX_training ,dataX_serhat)) # serhat data + HAPT data
inputY = np.concatenate((dataY_training ,dataY_serhat_T))
inputX = np.concatenate((dataX_testing ,inputX))
inputY = np.concatenate((dataY_testing ,inputY))
print(inputX.shape)
def rms_feature(input_data):
    newFeature = input_data
    RMSA = np.sqrt(np.square(input_data[:,:,0]) + np.square(input_data[:,:,1]) + np.square(input_data[:,:,2]))
    newFeature = np.insert(newFeature,9,RMSA,axis=2)
    RMSL = np.sqrt(np.square(input_data[:,:,3]) + np.square(input_data[:,:,4]) + np.square(input_data[:,:,5]))
    newFeature = np.insert(newFeature,10,RMSL,axis=2)
    RMSG = np.sqrt(np.square(input_data[:,:,6]) + np.square(input_data[:,:,7]) + np.square(input_data[:,:,8]))
    newFeature = np.insert(newFeature,11,RMSG,axis=2)
    return newFeature
inputX = rms_feature(inputX)
print(inputX.shape)
X_train, X_test, y_train, y_test = train_test_split(inputX, inputY, test_size=0.33, random_state=7)
_, X_pred, _, y_pred = train_test_split(X_test, y_test, test_size=0.2, random_state=7)
y_pred = np.argmax(y_pred, axis=1)
def plt_conf(normalised_confusion_matrix):
    width = 8
    height = 6
    plt.figure(figsize=(width, height))
    sn.set(font_scale=0.8) # for label size
    sn.heatmap(normalised_confusion_matrix, annot=True, annot_kws={"size": 8},xticklabels =LABELS , yticklabels=LABELS,cmap='inferno') # font size
    plt.title("Confusion matrix",fontsize = 'x-large')
    #tick_marks = np.arange(6)
    #plt.xticks(tick_marks, LABELS, rotation=90,fontsize = 'x-small' )
    #plt.yticks(tick_marks, LABELS,  rotation='horizontal', fontsize = 'x-small')
    plt.tight_layout()
    plt.ylabel('True label',fontsize = 'x-large')
    plt.xlabel('Predicted label', fontsize = 'x-large')
    plt.show()
def outputmodel():
    from tensorflow.keras import backend as K
    from tensorflow.python.tools import freeze_graph
    from tensorflow.python.tools import optimize_for_inference_lib
    input_node_names= ["LSTM_1_input"]
    output_node_name = "Dense_2/Softmax"
    MODEL_NAME = "HAR"

    tf.train.write_graph(K.get_session().graph_def, 'models', \
                         MODEL_NAME + '_graph.pbtxt')
    saver = tf.train.Saver()
    saver.save(K.get_session(), 'models/' + MODEL_NAME + '.chkp')
    
    freeze_graph.freeze_graph('models/' + MODEL_NAME + '_graph.pbtxt', None, \
        False, 'models/' + MODEL_NAME + '.chkp', output_node_name, \
        "save/restore_all", "save/Const:0", \
    'models/frozen_' + MODEL_NAME + '.pb', True, "")

def tflite_SaveModel():
    sess = tf.keras.backend.get_session()
    input_tensor = sess.graph.get_tensor_by_name('input:0')
    output_tensor = sess.graph.get_tensor_by_name('output/Softmax:0')
    converter = tf.lite.TFLiteConverter.from_session(
        sess, [input_tensor], [output_tensor])
    tflite = converter.convert()
    print('Model converted successfully!')
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
X_pred = X_pred.astype(np.float32)
temp_y = np.argmax(y_test, axis=1)
temp_X = pd.DataFrame(data=X_test)
temp_y = pd.DataFrame(data=temp_y)
temp_X.to_csv(path_or_buf='test_X.csv', na_rep='NaN', columns=None,
                  header=False, index=False, mode='w', encoding='utf-8',line_terminator='\n',)
temp_y.to_csv(path_or_buf='test_y.csv', na_rep='NaN', columns=None,
                  header=False, index=False, mode='w', encoding='utf-8',line_terminator='\n',)

for i in range(num_runs):
    
    result, model = RNN_model_V3(X_train,y_train,X_test,y_test)
    result = result * 100.0
    print('>#%d: %.4f' % (i+1, result))
    totalResultAccuracy.append(result)
    prediction = model.predict_classes(X_pred)
    #tf.saved_model.save(model, "New model/mobilenet/1/")
    #converter = tf.lite.TFLiteConverter.from_keras_model(model)
    #tflite_model = converter.convert()
    #tflite_SaveModel()
    pred_conf = model.predict_classes(X_test)
    confusion_matrix = metrics.confusion_matrix(np.argmax(y_test, axis=1), pred_conf)
    normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100
    #plt_conf(normalised_confusion_matrix)
    predicAcc = 0.0
    for k in range(len(y_pred)):
        #print("True Value=%s, Predicted=%s" % (y_pred[k], prediction[k]))
        if y_pred[k] == prediction[k]:
            predicAcc = predicAcc + 1
    predicAcc = (predicAcc / len(y_pred)) * 100.0
    #print('>Prediction Accuracy = #%d: %.4f' % (i+1, predicAcc))
dataMean, dataStd = mean(totalResultAccuracy), std(totalResultAccuracy)
print('Accuracy: %.4f%% (+/-%.4f)' % (dataMean, dataStd))
# Check the converted TensorFlow Lite model.
interpreter = tf.lite.Interpreter(model_content=tflite)

try:
  interpreter.allocate_tensors()
except ValueError:
  assert False

MINI_BATCH_SIZE = 1
correct_case = 0
for i in range(len(x_test)):
  input_index = (interpreter.get_input_details()[0]['index'])
  interpreter.set_tensor(input_index, x_test[i * MINI_BATCH_SIZE: (i + 1) * MINI_BATCH_SIZE])
  interpreter.invoke()
  output_index = (interpreter.get_output_details()[0]['index'])
  result = interpreter.get_tensor(output_index)
  # Reset all variables so it will not pollute other inferences.
  interpreter.reset_all_variables()
  # Evaluate.
  prediction = np.argmax(result)
  if prediction == y_test[i]:
    correct_case += 1

print('TensorFlow Lite Evaluation result is {}'.format(correct_case * 1.0 / len(x_test)))
##from tensorflow.python.tools import freeze_graph
##
##MODEL_NAME = 'har'
##
##input_graph_path = 'checkpoint/' + MODEL_NAME+'.pbtxt'
##checkpoint_path = './checkpoint/' +MODEL_NAME+'.ckpt'
##restore_op_name = "save/restore_all"
##filename_tensor_name = "save/Const:0"
##output_frozen_graph_name = 'frozen_'+MODEL_NAME+'.pb'
##
##freeze_graph.freeze_graph(input_graph_path, input_saver="",
##                          input_binary=False, input_checkpoint=checkpoint_path, 
##                          output_node_names="y_", restore_op_name="save/restore_all",
##                          filename_tensor_name="save/Const:0", 
##                          output_graph=output_frozen_graph_name, clear_devices=True, initializer_nodes="")
# support vector machine
# convert the three dimensions of [samples, timesteps, features] into the two dimensions of [samples, timesteps * features]
##trainX_SVM = dataX_training.reshape(dataX_training.shape[0], dataX_training.shape[1] * dataX_training.shape[2])
##testX_SVM = dataX_testing.reshape(dataX_testing.shape[0], dataX_testing.shape[1] * dataX_testing.shape[2])
##trainY_SVM , testY_SVM = dataY_trainingRaw[:,0], dataY_testingRaw[:,0]
##print(trainX_SVM.shape, trainY_SVM.shape, testX_SVM.shape, testY_SVM.shape)
##modelSVM = SVC();
##modelSVM.fit(trainX_SVM, trainY_SVM)
##prediction = modelSVM.predict(testX_SVM);
##accuracy = accuracy_score(testY_SVM, prediction) * 100
##print("SVM Accuracy:",'Accuracy=%.4f' % (accuracy))
##
### CNN with LSTM
##
##def CNNLTSM_Model(dataX_training,dataY_training,dataX_testing,dataY_testing):
##    verbose, epochs, batch_size = 1, 25, 64
##    numTimestep, numFeatures, numOutputs = dataX_training.shape[1], dataX_training.shape[2], dataY_training.shape[1]
##    numSteps, numLength = 4, 32
##    dataX_training = dataX_training.reshape(dataX_training.shape[0], numSteps, numLength, numFeatures)
##    dataX_testing = dataX_testing.reshape(dataX_testing.shape[0], numSteps, numLength, numFeatures)
##    modelCNN = tf.keras.models.Sequential()
##    modelCNN.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,numLength,numFeatures)))
##    modelCNN.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')))
##    modelCNN.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.5)))
##    modelCNN.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=2)))
##    modelCNN.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
##    modelCNN.add(tf.keras.layers.LSTM(100))
##    modelCNN.add(tf.keras.layers.Dropout(0.5))
##    modelCNN.add(tf.keras.layers.Dense(100, activation='relu'))
##    modelCNN.add(tf.keras.layers.Dense(numOutputs, activation='softmax'))
##    modelCNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
##	# fit network
##    modelCNN.fit(dataX_training, dataY_training, epochs=epochs, batch_size=batch_size, verbose=verbose)
##	# evaluate model
##    _, accuracy = modelCNN.evaluate(dataX_testing, dataY_testing, batch_size=batch_size, verbose=0)
##    return accuracy
##totalResultAccuracy_CNN = list()
##for i in range(num_runs):
##    result = CNNLTSM_Model(dataX_training,dataY_training,dataX_testing,dataY_testing)
##    result = result * 100.0
##    print('>#%d: %.4f' % (i+1, result))
##    totalResultAccuracy_CNN.append(result)
##dataMean, dataStd = mean(totalResultAccuracy_CNN), std(totalResultAccuracy_CNN)
##print('Accuracy: %.4f%% (+/-%.4f)' % (dataMean, dataStd))




##def normalization(input_data):
##    normalized_array = np.empty([np.size(input_data,0),np.size(input_data,1),np.size(input_data,2)])
##    for features in range(0,np.size(input_data,2),3):
##        minV = np.amin(input_data[:,:,features:features+3]);
##        print("min value: ", minV)
##        maxV = np.amax(input_data[:,:,features:features+3]);
##        print("max value: ", maxV)
##        print(input_data[0,0,features:features+3])
##        normalized_array[:,:,features:features+3] = 2* ((input_data[:,:,features:features+3]- minV)/(maxV-minV))-1
##        print(normalized_array[0,0,features:features+3])
##    return normalized_array
