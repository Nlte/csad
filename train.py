from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn as tflearn
from datetime import datetime
from sklearn import metrics

LOG_DIR = "log"

class Dataset(object):
    """Simple dataset wrapper."""

    def __init__(self, filename, batchsize):
        """Constructor"""
        # csv textfile reader
        self.reader = pd.read_csv(filename, compression='gzip',
                        names=['date','out_id', 'in_id', 'calls', 'duration'], parse_dates=['date'],
                        sep=',', quotechar='"', chunksize=batchsize)

    def next_batch(self):
        d = self.reader.next()
        d["day"] = d.date.dt.day
        d["hour"] = d.date.dt.hour
        d["week"] = d.date.dt.week
        d["month"] = d.date.dt.month
        features = np.asarray(d[["day", "hour", "in_id", "out_id"]])
        targets_flat = np.asarray(d["duration"])
        targets = np.expand_dims(targets_flat, 1)
        return features, targets

class NN(object):

    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        self.loss = None
        self.train = None
        self.net = None

    def linear(self, x, name, size):
        W = tf.get_variable(name+"/W", [x.get_shape()[1], size])
        b = tf.get_variable(name+"/b", [size], initializer=tf.zeros_initializer)

        return tf.matmul(x, W) + b

    def build(self):

        net = self.linear(self.data, "linear_1", 20)
        #net = self.linear(net, "linear_2", 4)
        net = self.linear(net, "linear_3", 1)
        #prediction, loss = tflearn.models.linear_regression(self.data, self.targets)
        self.result = net
        with tf.name_scope("Loss"):
            loss = tf.contrib.losses.mean_squared_error(net, self.targets)
            #loss = tf.clip_by_value(tf.reduce_mean(euclidean_loss), 1e+9, 1e-9)
        train_step = tf.train.AdamOptimizer(0.005).minimize(loss)
        tf.scalar_summary("Loss", loss)

        self.loss = loss
        self.train = train_step



if __name__ == "__main__":
    if not tf.gfile.IsDirectory(LOG_DIR):
        tf.gfile.MakeDirs(LOG_DIR)

    dataset = Dataset("project_data/SET1V_01.CSV.gz", 64)

    x = tf.placeholder(tf.float32, shape=[None, 4])
    y_ = tf.placeholder(tf.float32, shape=[None, 1])

    model = NN(x, y_)
    model.build()

    init_op = tf.global_variables_initializer()
    merged = tf.merge_all_summaries()

    with tf.Session() as sess:

        train_writer = tf.train.SummaryWriter(LOG_DIR + '/train', sess.graph)
        sess.run(init_op)

        features, targets = dataset.next_batch()

        for i in range(10000):
            _, loss, summ = sess.run([model.train, model.loss, merged], {x:features, y_:targets})
            print("%s - Loss : %f" % (datetime.now(), loss))
            train_writer.add_summary(summ, i)
