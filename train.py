from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from datetime import datetime
from sklearn import metrics

# FLAGS : (flag name, default value, description)
tf.app.flags.DEFINE_string("log_dir", "logs",
                            """Log directory.""")

tf.app.flags.DEFINE_integer("batch_size", 64,
                            """Batch size.""")

tf.app.flags.DEFINE_integer("base_station", 1,
                            """Base station to train on.""")

tf.app.flags.DEFINE_string("data_dir", "project_data/week_lon_lat_bs",
                            """Data csv directory.""")

FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

class Dataset(object):
    """Simple dataset wrapper."""

    def __init__(self, filename, batchsize):
        """Constructor"""
        # csv textfile reader
        self.reader = pd.read_csv(filename,
                        names=['date','out_id', 'in_id', 'calls', 'duration'], parse_dates=['date'],
                        sep=',', quotechar='"', chunksize=batchsize)

    def next_batch(self):
        d = next(self.reader)
        d["day"] = d.date.dt.day
        d["hour"] = d.date.dt.hour
        d["week"] = d.date.dt.week
        d["month"] = d.date.dt.month
        features = np.asarray(d[["day", "hour", "in_id", "out_id"]])
        targets_flat = np.asarray(d["calls"])
        targets = np.expand_dims(targets_flat, 1)
        return features, targets


def build_input_pipeplines(filenames, batch_size):
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    record_defaults = [[1.0]]*12
    year, month, day, hour, in_id, in_long, in_lat, out_id, out_long, out_lat, calls, durations = tf.decode_csv(
        value, record_defaults=record_defaults)
    features = tf.pack([day, hour, in_id, out_id])
    targets = tf.expand_dims(calls, 0)
    feature_batch, target_batch = tf.train.batch([features, targets], batch_size=batch_size)

    return feature_batch, target_batch


class NN(object):

    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        self.loss = None
        self.train = None
        self.net = None

    def build(self):
        net = tf.contrib.layers.relu(self.data, 50)
        net = tf.contrib.layers.relu(net, 50)
        net = tf.contrib.layers.relu(net, 50)
        net = tf.contrib.layers.linear(net, 1)
        predictions = tf.round(net)
        loss = tf.losses.mean_squared_error(net, self.targets)
        train_step = tf.train.AdamOptimizer(5e-4).minimize(loss)
        tf.summary.scalar("Loss", loss)

        self.predictions = predictions
        self.loss = loss
        self.optimize = train_step


def main(_):

    if not tf.gfile.IsDirectory(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)

    # Build data pipelines
    train_files = ["exo1_b%s_week%s.csv" % (FLAGS.base_station, i) for i in range(1,7)]
    train_files = [os.path.join(FLAGS.data_dir, f) for f in train_files]
    train_features, train_targets = build_input_pipeplines(train_files, FLAGS.batch_size)

    val_files = ["exo1_b%s_week7.csv" % FLAGS.base_station]
    val_files = [os.path.join(FLAGS.data_dir, f) for f in val_files]
    val_features, val_targets = build_input_pipeplines(val_files, FLAGS.batch_size)

    test_file = os.path.join(FLAGS.data_dir, "exo1_b%s_week8.csv" % FLAGS.base_station)
    test_df = pd.read_csv(test_file, names=["year", "month", "day", "hour", "in_id",
        "in_long", "in_lat", "out_id", "out_long", "out_lat", "calls", "durations"])
    test_features = test_df[["day", "hour", "in_id", "out_id"]]

    # Define placeholders and model
    x = tf.placeholder(tf.float32, shape=[None, 4])
    y_ = tf.placeholder(tf.float32, shape=[None, 1])
    model = NN(x, y_)
    model.build()

    init_op = tf.global_variables_initializer()
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        # init threads for queues (don't touch)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        train_writer = tf.summary.FileWriter(FLAGS.log_dir + "/train", sess.graph)
        val_writer = tf.summary.FileWriter(FLAGS.log_dir + "/validation")
        sess.run(init_op)

        for i in range(5000):
            # Training
            features, targets = sess.run([train_features, train_targets])
            _, loss, summ = sess.run([model.optimize, model.loss, merged], {x:features, y_:targets})
            train_writer.add_summary(summ, i)
            # Validation
            if i%50==0:
                features, targets = sess.run([val_features, val_targets])
                loss, summ = sess.run([model.loss, merged], {x: features, y_: targets})
                val_writer.add_summary(summ, i)
                print("%s - Loss : %f" % (datetime.now(), loss))

        # Test
        dummy_y = [[0]]
        predictions = []
        for i, row in test_features.iterrows():
            pred = sess.run(model.predictions, {x: np.expand_dims(np.array(row), 0), y_: dummy_y})
            predictions.append(pred.flatten()[0])
        test_df["predictions"] = pd.Series(predictions)
        score = mean_squared_error(test_df["predictions"], test_df["calls"])
        print("MSE: %f" % score)
        df_test = test_df[["calls", "predictions"]].groupby([test_df.hour]).sum()
        df_test.plot()
        plt.title("Calls semaine 8 base station %s" % FLAGS.base_station)
        plt.show()

        # close threads for queues (don't touch)
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    tf.app.run()
