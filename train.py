import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Reshape, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

XDim=32
YDim=32
ZDim=1

sizeBatchTrain=64
sizeBatchValid=64
nEpochs=50
nExamples=1000
sizeShuffle=512

pathTrain = '../data/train.tfr'  # The TFRecord file containing the training set
pathValid = '../data/val.tfr'    # The TFRecord file containing the validation set
pathTest = '../data/test.tfr'    # The TFRecord file containing the test set
pathWeight = '../data/neonet-v1.h5'  # The HDF5 weight file generated for the trained model
pathModel = '../data/neonet-v1.nn'  # The model saved as a YAML file

def CNN2D():
  inputs = Input((YDim, XDim, ZDim))
  # learn 32 filters
  conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
  conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
  # reduce input to 16x16
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

  # learn 64 filters
  conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
  conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
  # reduce input to 8x8
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

  # learn 128 filters
  conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
  conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
  # reduce input to 4x4
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

  # learn 256 filters
  conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
  conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
  # reduce input to 2x2
  pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
  flat4 = Flatten()(pool4)

  fc5 = Dense(256, activation='relu')(flat4)
  fc6 = Dense(512, activation='relu')(fc5)
  predict = Dense(1, activation = 'softmax')(fc6)

  model = Model(inputs=[inputs], outputs=[predict])

  model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

  return model


def train():
  K.set_image_data_format('channels_last')  # TF dimension ordering in this code
  featdef = {
    #'detID': tf.FixedLenSequenceFeature(shape=[], dtype=tf.string, allow_missing=True),
    'isNEO': tf.FixedLenSequenceFeature(shape=[1], dtype=tf.float32, allow_missing=True),
    'cutout': tf.FixedLenSequenceFeature(shape=[], dtype=tf.float32, allow_missing=True)
    #'params': tf.FixedLenSequenceFeature(shape=[], dtype=tf.float32, allow_missing=True)
    }
        
  def _parse_record(example_proto, clip=False):
    """Parse a single record into x and y images"""
    example = tf.parse_single_example(example_proto, featdef)
    #x = tf.decode_raw(example['cutout'], tf.float32)
    # unroll into a 2D array
    x = example['cutout']
    x = tf.reshape(x, (YDim, XDim, 1))
    
    y = example['isNEO']
    #y = tf.decode_raw(y, tf.float32)
    #y = tf.reshape(y, (1,1))
    #y = tf.cast(y, tf.float32)
    #y = tf.reshape(y, (1,1))
    y = y[0]

    return x, y

  #construct a TFRecordDataset
  dsTrain = tf.data.TFRecordDataset(pathTrain).map(_parse_record)
  dsTrain = dsTrain.shuffle(sizeShuffle)
  dsTrain = dsTrain.repeat()
  dsTrain = dsTrain.batch(sizeBatchTrain)

  dsValid = tf.data.TFRecordDataset(pathValid).map(_parse_record)
  dsValid = dsValid.shuffle(sizeShuffle)
  dsValid = dsValid.repeat()
  dsValid = dsValid.batch(sizeBatchValid)

  #dsTest = tf.data.TFRecordDataset(pathTest).map(_parse_record)
  #dsTest = dsValid.repeat(30)
  #dsTest = dsValid.shuffle(10).batch(sizeBatch)

  print('-'*30)
  print('Creating and compiling model...')
  print('-'*30)
  model = CNN2D()

  print(model.summary())

  callbacks = [
      tf.keras.callbacks.ModelCheckpoint(pathWeight, verbose=1, save_best_only=True),
      tf.keras.callbacks.TensorBoard(log_dir='../logs')
  ]

  print('-'*30)
  print('Fitting model...')
  print('-'*30)

  #print(dsTrain)
  history = model.fit(dsTrain, validation_data=dsValid, validation_steps=1, steps_per_epoch=int(np.ceil(nExamples/sizeBatchTrain)), epochs=nEpochs, verbose=1, callbacks=callbacks)
  #history = model.fit(dsTrain, validation_data=dsValid, epochs=nEpochs, verbose=1, callbacks=callbacks)

  # serialize model to JSON
  model_serial = model.to_json()
  with open(pathModel, "w") as yaml_file:
    yaml_file.write(model_serial)

  #Y = model.predict(dsTest, steps=1)
  #image = Y[0,:,:,0]
  
  #fig = plt.figure(num='Level 2 - Predicted')

  #plt.gray()
  #plt.imshow(image)
  #plt.show()
  #plt.close(fig)

print(tf.__version__)
train()
