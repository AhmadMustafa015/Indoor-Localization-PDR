#!/usr/bin/env python3

import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

print("Tensorflow version: "+ tf.__version__)

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.iloc[:, -1]
  dataframe.drop(dataframe.columns[-1], axis=1, inplace=True)
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

df = pd.read_csv("Data.txt")
print(df.head())

train, test = train_test_split(df, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

for feature_batch, label_batch in train_ds.take(2):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of Param1:', feature_batch['Param1'])
  print('A batch of labels:', label_batch )

feature_columns = []
for header in ['Param1', 'Param2']:
    feature_columns.append(feature_column.numeric_column(header))

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=15)
          
loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)

#model.save('my_model.h5')
#model.save_weights('my_model')
print(len(model.get_weights()[0]))
