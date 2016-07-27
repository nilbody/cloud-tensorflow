#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import json
import os
import argparse

flags = tf.app.flags
flags.DEFINE_string("mode", "train", "Mode to run, options: train or predict")
FLAGS = flags.FLAGS

parser = argparse.ArgumentParser(description='Train an MNIST model.')

# Flags that are part of the contract.
parser.add_argument('--cluster',
                      type=json.loads,
                      default={},
                      help='The JSON cluster description.')
parser.add_argument('--task',
                      type=json.loads,
                      default={'type': 'master', 'index': 0},
                      help='The JSON task type and index.')
parser.add_argument('--job',
                      type=json.loads,
                      default={},
                      help='The JSON request as submitted to the service.')
parser.add_argument('--mode',type=str, default="train", help="Mode to run, options: train or predict")
args = parser.parse_args()

if args.job:
    job = args.job
    model_dir = os.path.join(job["output_path"], "model")
    os.makedirs(model_dir)
else:
    model_dir = "/tmp/mnist_160727_153131/model"

saver_path = os.path.join(model_dir, "export")

train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10

keys_placeholder = tf.placeholder("float")
keys = tf.identity(keys_placeholder)
X = tf.placeholder("float")
Y = tf.placeholder("float")
w = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="bias")

global_step = tf.Variable(0, name='global_step', trainable=False)
loss = tf.square(Y - tf.mul(X, w) - b)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss, global_step=global_step)
predict_op1 = tf.mul(X, w) + b
predict_op2 = tf.mul(X, w) + b + 1
checkpoint_period = 100
saver = tf.train.Saver(sharded=True)
tf.add_to_collection("inputs", json.dumps({'key': keys_placeholder.name, 'X': X.name, 'Y': Y.name}))
tf.add_to_collection("outputs", json.dumps({'key': keys.name, 'predict_op1': predict_op1.name, 'predict_op2': predict_op2.name}))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    tf.train.write_graph(sess.graph.as_graph_def(), "./", "linear_graph.pb", as_text=False)

    #if FLAGS.mode == "train":
    if args.mode == "train":
        for i in range(1000):
            for (x, y) in zip(train_X, train_Y):
                _, epoch = sess.run([train_op, global_step], feed_dict={X: x, Y: y})

            if i % checkpoint_period == 0:
                saver.save(sess, saver_path, global_step=epoch)
                print("Save checkpoint with global step: {}".format(epoch))
        
    elif args.mode == "predict":
        #ckpt = tf.train.get_checkpoint_state(saver_path)
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            predict_x = 10
            predict_y = 20
            predict_result = sess.run(predict_op1, feed_dict={keys_placeholder: 10, X: predict_x, Y: predict_y})
            print("x: {}, y: {}, predict result: {}".format(predict_x, predict_y, predict_result))
        else:
            print("No model found, exit")
    else:
        print("Invalide mode, exit")

    print(sess.run(w))
    print(sess.run(b))
