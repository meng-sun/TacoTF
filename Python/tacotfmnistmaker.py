import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


FLAGS = None
mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data", one_hot=True)
train_filename = 'train.tfrecords'  # address to save the TFRecords file

#mnist.train.next_batch(1)
#mnist.test.labels = [1] mnist.test.images  [784]


# open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)


for i in range(20):

    nn_val = []
    nn_idx = []
    image, label = mnist.train.next_batch(1)
    for i in range(784):
      if image[0][i]!=0:
        nn_val.append(image[0][i])
        nn_idx.append(i)

    # Load the image
    log_fmt = ["dense"]
    nn_fmt = ["sparse"]
    log_val = label[0]
    #nn_val = [3,5]
    log_idx = [10]
    #nn_idx = [0,1]
    # Create a feature
    feature = {	'nV': _float_feature(nn_val),
		'lV': _float_feature(log_val),
               	'lF': _bytes_feature(log_fmt),
		'nF': _bytes_feature(nn_fmt),
		'nI': _int64_feature(nn_idx),
		'lI': _int64_feature(log_idx)
		}

    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
    
writer.close()
