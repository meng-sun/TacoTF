import tensorflow as tf
import numpy as np
#np.set_printoptions(threshold=np.nan)
import time
import threading

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.client import timeline

taco_tensor_module = tf.load_op_library('/data/scratch/mysun/tensorflow/bazel-bin/tensorflow/core/user_ops/taco_tensor.so')
taco_expr_op_module = tf.load_op_library('/data/scratch/mysun/tensorflow/bazel-bin/tensorflow/core/user_ops/taco_expr_op.so')
taco_fast_dense_module = tf.load_op_library('/data/scratch/mysun/tensorflow/bazel-bin/tensorflow/core/user_ops/taco_fast_dense.so')

@ops.RegisterGradient("TacoFastDense")
def _taco_tensor_grad(op,*grad):
  gradprop_list = [None]*len(op.inputs)
  return gradprop_list

@ops.RegisterGradient("TacoTensor")
def _taco_tensor_grad(op,*grad):
  gradprop_list = [None]*len(op.inputs)
  gradprop_list[2] = grad[0]
  return gradprop_list


@ops.RegisterGradient("TacoExprOp")
def _taco_expr_op_grad(op, *grad):
  """The gradients for `taco_expr_op`.

  Args:
    op: The `taco_expr_op` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `taco_expr_op` op.

  Returns:
    Gradients with respect to the input of `taco_expr_op`.
  """
#under weird circumstances user may have to debug this
  shape = array_ops.shape(grad[0])
  tval, tidx = taco_tensor_module.taco_tensor(shape, ["dense"], [grad[0]],[tf.constant(1)])

  def f1():
    new_matrix = taco_expr_op_module.taco_expr_op(
                                               op.outputs[len(op.outputs)-2],
                                               [op.outputs[1],tval],
                                               [op.outputs[3],tidx],
                                               which_grad=-1,
                                               )
   
    gradprop_list = []
    for i in range(len(op.inputs)):
      gradprop_list.append(tf.constant(0, dtype=tf.float64))
    gradprop_list[op.get_attr("which_grad")+1] = new_matrix[0][0]
   
    return gradprop_list;

  def f2():
    new_matrix = taco_expr_op_module.taco_expr_op(
                                               op.outputs[len(op.outputs)-2],
                                               [tval, op.outputs[1]],
                                               [tidx, op.outputs[3]],
                                               which_grad=-1,
                                               )
    gradprop_list = []
    for i in range(len(op.inputs)):
      gradprop_list.append(tf.constant(0, dtype=tf.float64))
    return gradprop_list;

  def f3():
    gradprop_list = []
    for i in range(len(op.inputs)):
      gradprop_list.append(tf.constant(0, dtype=tf.float64))
    return gradprop_list;

  print(op.outputs[len(op.outputs)-1])
  result = tf.case({
                   tf.equal(1,op.outputs[len(op.outputs)-1]): f1,
                   tf.equal(2,op.outputs[len(op.outputs)-1]): f2
                   },
                   default = f3,
                   strict = True
                   )

  return result

filename = "train.tfrecords"
filename_queue = tf.train.string_input_producer([filename],num_epochs=None)
reader = tf.TFRecordReader()
lineno, line = reader.read(filename_queue)
"""remove if not using batch"""
with tf.name_scope('pre-processing'):
  features = tf.parse_single_example(line, features={
	'lF': tf.VarLenFeature(tf.string),
	'nF': tf.VarLenFeature(tf.string),
	'nI' : tf.VarLenFeature(tf.int64),
        #'nI' : tf.FixedLenFeature([-1,2], tf.int64),
	'nV': tf.VarLenFeature(tf.float32),
	'lI': tf.VarLenFeature(tf.int64),
	'lV': tf.VarLenFeature(tf.float32)
  })

  nnIdxBlob = features['nI']
  nnVal = features['nV']
  logIdxBlob = features['lI']
  logVal = features['lV']
  nnFmt = features['nF']
  logFmt = features['lF']


  nnIdxDense = tf.sparse_tensor_to_dense(tf.cast(nnIdxBlob,tf.int32))
  #nnIdxDense = tf.cast(nnIdxBlob,tf.int32)

  logIdxDense = tf.sparse_tensor_to_dense(tf.cast(logIdxBlob,tf.int32))
  #this always has to change BELOW
  split_nnidx = tf.expand_dims(nnIdxDense, -1)
  split_logidx=[logIdxDense]
  #^this si the other problem if u cant split

  casted = tf.cast(nnVal.values, tf.float64)

start = time.clock()
# this is a major problem! Right not we can't do bs > 1 because there is no concat operator for taco yet 
#dequeue up to is not supported bc there is no concat operator for taco yet
sess = tf.InteractiveSession()
#multiple eds, if neededhreads, if needed
coord = tf.train.Coordinator()
#for i in range(0,1):
#    thread = threading.Thread(target=features)
#    thread.start()
 
with tf.name_scope('dropout'):
  """need to check that this size is right within the tensor"""
  matrix =tf.get_variable("W", dtype =tf.float64, initializer=tf.ones([7840],dtype=tf.float64))
  dropped = tf.nn.dropout(matrix, 0.9)

with tf.name_scope('layer1'):
  with tf.name_scope('taco_tensor_creation'):
    val, idx = taco_tensor_module.taco_tensor(
	tf.constant([784]),
	nnFmt.values,
        casted,
	split_nnidx,
	)

    l_val,l_idx = taco_tensor_module.taco_tensor(tf.constant([10]),logFmt.values,tf.cast(logVal.values,tf.float64),split_logidx)
  #_, mat_idx = taco_tensor_module.taco_tensor(
	#tf.constant([784,10]),
	#["dense","dense"],
        #dropped,
	#[tf.constant([-1])],
	#)
    mat_idx = taco_fast_dense_module.taco_fast_dense(tf.constant([784,10], dtype=tf.int32))
    mat_val = matrix

  with tf.name_scope('taco_expression_calculation'):
    pre_entr = taco_expr_op_module.taco_expr_op("ans(k) =(val(j)*mat(j,k))",[val, mat_val, l_val],[idx, mat_idx, l_idx],which_grad=1)

  with tf.name_scope('bias'):
    entr = pre_entr[0][0] + tf.get_variable("b", [10],dtype=tf.float64, initializer=tf.zeros_initializer)



with tf.name_scope('cross_entropy'):
  loss = tf.nn.softmax_cross_entropy_with_logits(labels=l_val,logits=entr)
  #entropy = tf.reduce_mean(loss)
#correct_prediction = tf.equal(tf.argmax(entr), tf.argmax(l_val))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#train_step = tf.train.AdamOptimizer().minimize(entropy, var_list=[matrix])

with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer().minimize(loss, var_list=[matrix])
#instead use this, which will be wrapped up in a new op:
#opt = GradientDescentOptimizer(learning_rate=0.5)
#grads_and_vars = opt.compute_gradients(loss, )
#capped_grads_and_vars = [(MyCapper(gv[0]), gv[1]) for gv in grads_and_vars]
#opt.apply_gradients(capped_grads_and_vars)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('/data/scratch/mysun/tensorflow/tacotf/logs/train', sess.graph)
test_writer = tf.summary.FileWriter('/data/scratch/mysun/tensorflow/tacotf/logs/test')
#gu taco nn graph
sess.run(tf.global_variables_initializer())
#use built in queue runner and coordinator from shuffle batch
tf.train.start_queue_runners(coord=coord, sess=sess)
for i in range(10):
        #the weird list issue has not been resolved
        #sess.run(train_step, feed_dict={nn_in:batch_nn_in, logit_idx_in=batch_logit_idx, nn_idx_in=batch_nn_idx, nn_ft_in = batch_nn_ft , l_in:batch_logit})
        #print(sess.run([nnIdxBlob,nnVal]))
        #print(sess.run(split_nnidx))
        #print(sess.run(tf.split(tf.expand_dims(nnIdxDense, -1),[1,1],axis=0)))
        #print(sess.run([entr_val,entr_idx]))
  run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()
  summaries, _ = sess.run([merged, train_step],
                               options = run_options,
                               run_metadata = run_metadata
                               )
  if i==9:
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('/data/scratch/mysun/timelines/timeline_tacotf.json', 'w') as f:
     f.write(ctf)        
  train_writer.add_run_metadata(run_metadata, 'step%03d' %i)
  train_writer.add_summary(summaries, i)

test_writer.close()
train_writer.close()

print(sess.run(loss))
#coord.request_stop()
#coord.join(thread)
end = time.clock()
print ("all done in:","%.10f" % (end - start)," seconds")
sess.close()
