import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

XDim=32
YDim=32
ZDim=1
PDim=3

pathTrain = '../data/train.tfr'  # The TFRecord file containing the training set
pathValid = '../data/val.tfr'    # The TFRecord file containing the validation set
pathTest = '../data/test.tfr'    # The TFRecord file containing the test set

batchSize=20
batchN=1

with tf.Session() as sess:
    feature = {
        'detID': tf.FixedLenFeature(shape=[], dtype=tf.string),
        'isNEO': tf.FixedLenFeature(shape=[], dtype=tf.float32),
        'cutout': tf.FixedLenFeature(shape=[YDim*XDim], dtype=tf.float32),
        'params': tf.FixedLenFeature(shape=[PDim], dtype=tf.float32),
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
    y = features['isNEO']
    did = features['detID']
    p = features['params']
    #p = np.random.rand(65)
    # Any preprocessing here ...

    
    # Creates batches by randomly shuffling tensors
    #X, Y, P, ID = tf.train.batch([x, y, p, did], batch_size=batchSize)
    ID, Y, X, P = tf.train.batch([did, y, x, p], batch_size=batchSize)

    # unroll into a 2D array
    #X = X[0]
    #Y = Y[0]
    #ID = ID[0]
    #P = tf.reshape(P, (-1, PDim))

    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    plt.gray()
    # Now we read batches of images and labels and plot them
    for batch_index in range(batchN):
        detID, actual, image, params = sess.run([ID, Y, X, P])
        for m in range(batchSize):
            #print(detID[m])
            print("Detection %.0f y=%f,x=%f"%(params[m,0],params[m,1],params[m,2]))
            #print(actual[m])
            #print(params[m])
            #print(image.shape)
            img = np.reshape(image[m], (YDim, XDim))
            plt.imshow(img)
            plt.show()


    # Stop the threads
    coord.request_stop()
    
    # Wait for threads to stop
    coord.join(threads)
    sess.close()

