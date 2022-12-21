# date:2022/12/19 下午3:56
# author:ls
import collections

import tensorflow as tf
import tensorflow_federated as tff
from tensorflow.keras import datasets, layers, models
import numpy as np

EPOCHS = 1
BATCH_SIZE = 32
# ROUND_CLIENTS <= NUM_CLIENTS
ROUND_CLIENTS = 2
NUM_CLIENTS = 2


def make_client(num_clients, X, y):
    total_image_count = len(X)
    image_per_set = int(np.floor(total_image_count / num_clients))

    client_train_dataset = collections.OrderedDict()
    for i in range(1, num_clients + 1):
        client_name = i - 1
        start = image_per_set * (i - 1)
        end = image_per_set * i

        print(f"Adding data from {start} to {end} for client : {client_name}")
        data = collections.OrderedDict((('label', y[start:end]), ('pixels', X[start:end])))
        client_train_dataset[client_name] = data

    train_dataset = tff.simulation.datasets.TestClientData(client_train_dataset)

    return train_dataset


def make_test(num_clients, X, y):
    client_train_dataset = collections.OrderedDict()
    for i in range(1, num_clients + 1):
        client_name = i - 1
        data = collections.OrderedDict((('label', y), ('pixels', X)))
        client_train_dataset[client_name] = data

    train_dataset = tff.simulation.datasets.TestClientData(client_train_dataset)

    return train_dataset


(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
cifarFedTrain = make_client(NUM_CLIENTS, X_train, y_train)
test = make_test(NUM_CLIENTS, X_test, y_test)


def map_fn(example):
    return collections.OrderedDict(
        x=example['pixels'],
        y=example['label']
    )


def client_data(client_id):
    ds = cifarFedTrain.create_tf_dataset_for_client(cifarFedTrain.client_ids[client_id])
    return ds.repeat(EPOCHS).shuffle(500).batch(BATCH_SIZE).map(map_fn)


def test_data(client_id):
    ds = test.create_tf_dataset_from_all_clients(test.client_ids[client_id])
    return ds.repeat(EPOCHS).shuffle(500).batch(BATCH_SIZE).map(map_fn)


test_data = [test_data(n) for n in range(ROUND_CLIENTS)]
train_data = [client_data(n) for n in range(ROUND_CLIENTS)]
element_spec = train_data[0].element_spec

OPTIMIZER = tf.keras.optimizers.Adam()
LOSS = tf.keras.losses.SparseCategoricalCrossentropy()
METRICS = [tf.keras.metrics.SparseCategoricalAccuracy()]


def create_keras_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    return model


callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)


def model_fn():
    model = create_keras_model()
    return tff.learning.from_keras_model(
        model,
        input_spec=element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],

    )


iterative_process = tff.learning.algorithms.build_weighted_fed_avg(model_fn,
                                                                   client_optimizer_fn=lambda: tf.keras.optimizers.Adam(),

                                                                   server_optimizer_fn=lambda: tf.keras.optimizers.Adam())

NUM_ROUNDS = 500
logdir = "/home/ls/PycharmProjects/TFF_Image_Classification-main/FedAvg_cifar/"
evaluation_process = tff.learning.algorithms.build_fed_eval(model_fn)
evaluation_state = evaluation_process.initialize()

summary_writer = tf.summary.create_file_writer(logdir)
state = iterative_process.initialize()
with summary_writer.as_default():
    for round_num in range(0, NUM_ROUNDS):
        result = iterative_process.next(state, train_data)
        state = result.state
        metrics = result.metrics
        print('round {:2d}, metrics={}'.format(round_num, metrics))
        for name, value in metrics['client_work']['train'].items():
            tf.summary.scalar(name, value, step=round_num)
    model_weights = iterative_process.get_model_weights(state)
    evaluation_state = evaluation_process.set_model_weights(evaluation_state, model_weights)
    evaluation_output = evaluation_process.next(evaluation_state, test_data)
    metrics = evaluation_output.metrics
    print('test , metrics={}'.format(metrics))
