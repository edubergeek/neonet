import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

XDim=32
YDim=32
ZDim=1
PDim=65

pathTrain = '../data/train.tfr'  # The TFRecord file containing the training set
pathValid = '../data/val.tfr'    # The TFRecord file containing the validation set
pathTest = '../data/test.tfr'    # The TFRecord file containing the test set

batchSize=8
batchN=1

with tf.Session() as sess:
    feature = {
        'cutout': tf.FixedLenSequenceFeature(shape=[], dtype=tf.float32, allow_missing=True),
        'isNEO': tf.FixedLenSequenceFeature(shape=[1], dtype=tf.float32, allow_missing=True),
        'detID': tf.FixedLenSequenceFeature(shape=[], dtype=tf.string, allow_missing=True),
        'params': tf.FixedLenSequenceFeature(shape=[], dtype=tf.float32, allow_missing=True)
      }
        
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([pathTrain], num_epochs=1)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)

    """Parse a single record into x and y images"""
    x = features['cutout']
    # unroll into a 2D array
    x = tf.reshape(x, (YDim, XDim))
    # use slice to crop the data to the model dimensions - powers of 2 are a factor
    #x = tf.slice(x, (0, 0, 0, 0), (WStokes, YDim, XDim, ZStokes))
    
    y = features['isNEO']
    # use slice to crop the data
    #y = tf.slice(y, (0), (1,1))
    y = tf.cast(y, tf.float32)
    y = tf.reshape(y, (1,1))
    # Any preprocessing here ...

    #p = features['params']
    #p = tf.reshape(p, (1, PDim))
    #did = features['detID']
    #did = tf.cast(y, tf.string)
    
    # Creates batches by randomly shuffling tensors
    #X, Y, P = tf.train.batch([x, y, p], batch_size=batchSize)
    X, Y = tf.train.batch([x, y], batch_size=batchSize)

    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    plt.gray()
    # Now we read batches of images and labels and plot them
    for batch_index in range(batchN):
        #image, actual, params = sess.run([X, Y, P])
        image, actual = sess.run([X, Y])
        print(image.shape, actual.shape)
        for i in range(X.shape[0]):
            #print(params[i][0])
            print(actual[i][0])
            img = image[i, 0:YDim,0:XDim]
            plt.imshow(img)
            plt.show()


    # Stop the threads
    coord.request_stop()
    
    # Wait for threads to stop
    coord.join(threads)
    sess.close()

