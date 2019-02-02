from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from nets.resnet import resnet_model
from nets.student_model import model
import input

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--data_dir', type=str,
                    help='The path to the data directory.')

parser.add_argument('--teacher_dir', type=str,
                    help='The path to the max log.')

parser.add_argument('--train_epochs', type=int, default=80,
                    help='The number of epochs to train.')

parser.add_argument('--batch_size', type=int, default=32,
                    help='The number of images per batch.')

parser.add_argument('--learning_rate_start', type=float, default=0.0002,
                    help='The leatning rate at the beginning.')

parser.add_argument('--learning_rate_end', type=float, default=0.0000001,
                    help='The leatning rate in the end.')

parser.add_argument('--log_dir', type=str, default='./log',
                    help='The path to the log.')

parser.add_argument('--log_dir_best', type=str, default='./maxlog',
                    help='The path to the max log.')

FLAGS = parser.parse_args()

model_file_name = "model_imagenet_tfrecord.ckpt"
checkpoint_path = os.path.join(FLAGS.log_dir, model_file_name)
max_checkpoint_path = os.path.join(FLAGS.log_dir_best, model_file_name)

batch_size = FLAGS.batch_size

train_epochs = FLAGS.train_epochs
LR_start = FLAGS.learning_rate_start
LR_fin = FLAGS.learning_rate_end
LR_decay = (LR_fin / LR_start) ** (1. / train_epochs)
decayed_LR = tf.Variable(LR_start, dtype=tf.double)

num_classes = 1001
train_num = 1281167
val_num = 50000
printp = 100000
loss_ratio = 50

def main():

  with tf.device('/gpu'):
    train_dataset = input.input_fn(is_training=True,
                                   data_dir=FLAGS.data_dir,
                                   batch_size=batch_size,
                                   num_epochs=train_epochs + 1)
    train_iterator = train_dataset.make_initializable_iterator()
    x_train_batch, y_train_batch = train_iterator.get_next()

    y_train_batch_onehot = tf.one_hot(y_train_batch, depth=num_classes)

    val_dataset = input.input_fn(is_training=False,
                                 data_dir=FLAGS.data_dir,
                                 batch_size=batch_size,
                                 num_epochs=train_epochs + 1)
    val_iterator = val_dataset.make_initializable_iterator()
    x_val_batch, y_val_batch = val_iterator.get_next()

    y_val_batch_onehot = tf.one_hot(y_val_batch, depth=num_classes)

    with tf.variable_scope("resnet_model") as scope:
      network_teacher, logits_teacher = resnet_model(
        inputs=x_train_batch,
        training=False,
        resnet_size=50,
        bottleneck=True,
        num_classes=num_classes,
        num_filters=64,
        kernel_size=7,
        conv_stride=2,
        first_pool_size=3,
        first_pool_stride=2,
        second_pool_size=7,
        second_pool_stride=1,
        block_sizes=[3, 4, 6, 3],
        block_strides=[1, 2, 2, 2],
        final_size=2048,
        version=2, data_format=None
      )

    with tf.variable_scope("student") as scope:
      network_student, logits_train_stu = \
            model(x_train_batch, num_classes, is_training=True)
      scope.reuse_variables()
      _, logits_val_stu = \
            model(x_val_batch, num_classes, is_training=False)

    softmax_train_student0 = tf.nn.softmax(logits_train_stu * 0.016)
    softmax_train_student1 = tf.nn.softmax(logits_train_stu * 0.16)
    softmax_train_teacher = tf.nn.softmax(logits_teacher*0.1)

    loss0_ = tf.reduce_mean(- tf.reduce_sum(tf.stop_gradient(softmax_train_teacher) * \
                                            tf.log(softmax_train_student0), reduction_indices=1))
    loss1_ = tf.reduce_mean(- tf.reduce_sum(y_train_batch_onehot * \
                                            tf.log(softmax_train_student1), reduction_indices=1))
    loss_ = loss1_ / loss_ratio + loss0_

    val_loss_ = tf.losses.softmax_cross_entropy(
      onehot_labels=y_val_batch_onehot,
      logits=logits_val_stu
    )

    correct_prediction = tf.nn.in_top_k(logits_train_stu, y_train_batch, 1)
    acc1_ = tf.reduce_mean(tf.cast(correct_prediction, "double"))

    correct_prediction_top5 = tf.nn.in_top_k(logits_train_stu, y_train_batch, 5)
    acc5_ = tf.reduce_mean(tf.cast(correct_prediction_top5, "double"))

    correct_prediction_validation = tf.nn.in_top_k(logits_val_stu, y_val_batch, 1)
    val_acc1_ = tf.reduce_mean(tf.cast(correct_prediction_validation, "double"))

    correct_prediction_validation_top5 = tf.nn.in_top_k(logits_val_stu, y_val_batch, 5)
    val_acc5_ = tf.reduce_mean(tf.cast(correct_prediction_validation_top5, "double"))

    v_list = tl.layers.get_variables_with_name('resnet_model', train_only=False)
    v_list_student = tl.layers.get_variables_with_name('student', train_only=False)

    scale_variable_list = []
    else_variable_list = []
    for v in tl.layers.get_variables_with_name('student'):
      if '_Binary' in v.op.name:
        if '/b' not in v.op.name:
          scale_variable_list.append(v)
        else:
          else_variable_list.append(v)
      else:
        else_variable_list.append(v)

    len_scale = len(scale_variable_list)
    print('scale_variable_list')
    for var in scale_variable_list:
      print(var.op.name)

    len_else = len(else_variable_list)
    print('else_variable_list')
    for var in else_variable_list:
      print(var.op.name)

    fan_in = []
    fan_out = []
    W_LR_scale = []
    for i in range(0, len_scale):
      variable_shape = scale_variable_list[i].get_shape()
      fan_in1 = tf.cast(variable_shape[-2], dtype=tf.double)
      fan_out1 = tf.cast(variable_shape[-1], dtype=tf.double)
      for dim in variable_shape[:-2]:
        fan_in1 *= tf.cast(dim, dtype=tf.double)
        fan_out1 *= tf.cast(dim, dtype=tf.double)
      fan_in.append(fan_in1)
      fan_out.append(fan_out1)
      W_LR_scale.append(1. / tf.sqrt(1.5 / (fan_in1 + fan_out1)))

    with tf.device('/gpu'):   # <-- remove it if you don't have GPU
      opt_else = tf.train.AdamOptimizer(decayed_LR, beta1=0.99)
      opt = []
      for i in range(0, len_scale):
        opt.append(tf.train.AdamOptimizer(decayed_LR * W_LR_scale[i]))

      lr_assign_op = tf.assign(decayed_LR, decayed_LR * tf.cast(LR_decay, dtype=tf.double))
      grads = tf.gradients(loss_, else_variable_list + scale_variable_list)

      for i in range(0, len_scale):
        tf.add_to_collection('train_op', opt[i].apply_gradients(zip([grads[i+len_else]], [scale_variable_list[i]])))
      tf.add_to_collection('train_op', opt_else.apply_gradients(zip(grads[:len_else], else_variable_list)))
      train_op = tf.get_collection('train_op')

    print("clip begin")
    for v in scale_variable_list:
      print(v.op.name)
      tf.add_to_collection('clip', tf.assign(v, tf.clip_by_value(v, -1.0, 1.0)))
    print("clip end")
    assign_v_op = tf.get_collection('clip')

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tl.layers.initialize_global_variables(sess)
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    print("---------------------W_LR_scale------------")
    for i in range(0, len_scale):
      print(sess.run(W_LR_scale[i]))

    saver_teacher = tf.train.Saver(var_list=v_list)
    dirs = FLAGS.teacher_dir
    checkpoint_path = tf.train.latest_checkpoint(dirs)
    saver_teacher.restore(sess, checkpoint_path)

    saver = tf.train.Saver(max_to_keep=1, var_list=v_list_student)
    saver_best = tf.train.Saver(max_to_keep=1, var_list=v_list_student)

    sess.run(train_iterator.initializer)
    sess.run(val_iterator.initializer)

    def train_epoch():
      train_acc1, train_acc5, train_loss = 0, 0, 0
      batches = int(np.ceil(train_num / batch_size))
      for s in range(batches):
        acc1, acc5, loss, LR, _ = sess.run([acc1_, acc5_, loss_, decayed_LR, train_op])
        sess.run(assign_v_op)  # weight clip
        train_acc1 += acc1
        train_acc5 += acc5
        train_loss += loss
        if (s != 0) & (s % printp == 0):
          print("train-----------", "i", s, "batches", batches)
          print(acc1)
          print(acc5)
          print("loss", loss)
          print("LR:", LR)
          sys.stdout.flush()

      train_acc1 = train_acc1 / batches * 100
      train_acc5 = train_acc5 / batches * 100
      train_loss = train_loss / batches
      return train_acc1, train_acc5, train_loss, LR

    def val_epoch():
      val_acc1, val_acc5, val_loss = 0, 0, 0
      batches = int(np.ceil(val_num / batch_size))
      for s in range(batches):
        acc1, acc5, loss = sess.run([val_acc1_, val_acc5_, val_loss_])
        val_acc1 += acc1
        val_acc5 += acc5
        val_loss += loss

      val_acc1 = val_acc1 / batches * 100
      val_acc5 = val_acc5 / batches * 100
      val_loss = val_loss / batches
      return val_acc1, val_acc5, val_loss

    best_val_acc = 0
    best_epoch = 0

    for epoch in range(train_epochs):
      start_time = time.time()
      acc1, acc5, train_loss, LR = train_epoch()
      val_acc1, val_acc5, val_loss = val_epoch()

      summary = tf.Summary(value=[
        tf.Summary.Value(tag="learning_rate", simple_value=LR),
        tf.Summary.Value(tag="train_loss", simple_value=train_loss),
        tf.Summary.Value(tag="train_acc1", simple_value=acc1),
        tf.Summary.Value(tag="val_loss", simple_value=val_loss),
        tf.Summary.Value(tag="val_acc1", simple_value=val_acc1),
      ])
      summary_writer.add_summary(summary, epoch)
      summary_writer.flush()
      saver.save(sess, checkpoint_path, global_step=epoch + 1)

      if val_acc5 >= best_val_acc:
        best_val_acc = val_acc5
        best_epoch = epoch + 1
        saver_best.save(sess, max_checkpoint_path, global_step=epoch + 1)

      epoch_duration = time.time() - start_time
      sess.run(lr_assign_op)

      print("Epoch " + str(epoch + 1) + " of " + str(train_epochs) + " took   " + str(epoch_duration) + "s")
      print("  LR:                              " + str(LR))
      print("  training loss:                   " + str(train_loss))
      print("  train accuracy rate1:            " + str(acc1) + "%")
      print("  train accuracy rate5:            " + str(acc5) + "%")
      print("  validation loss:                 " + str(val_loss))
      print("  validation accuracy rate1:       " + str(val_acc1) + "%")
      print("  validation accuracy rate5:       " + str(val_acc5) + "%")
      print("  best epoch:                      " + str(best_epoch))
      print("  best validation accuracy rate:   " + str(best_val_acc) + "%")
      sys.stdout.flush()
    summary_writer.close()
    sess.close()

if __name__ == '__main__':
  main()
