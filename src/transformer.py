# Derived from https://keras.io/examples/timeseries/timeseries_transformer_classification/
import numpy as np, tensorflow as tf
import matplotlib.pyplot as plt
import os, random, time
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
from kf_attention import KfAttention



def get_dataset():
    dataset = os.path.join(os.path.dirname(__file__), "..", "data")
    print(f"file = {dataset}")
    train_dataset = pd.read_csv(os.path.join(dataset, "twitter_trace.csv"))
    print(f"head = {train_dataset.head()}")

    d0 = datetime.strptime("01/01/2023 00:00", "%m/%d/%Y %H:%M");
    print(datetime.strptime(train_dataset['End'][0], "%m/%d/%Y %H:%M"))
    train_dataset['Start'] = train_dataset['Start'].transform(
        lambda d: (datetime.strptime(d, "%m/%d/%Y %H:%M") - d0).total_seconds())
    train_dataset['End'] = train_dataset['End'].transform(
        lambda d: (datetime.strptime(d, "%m/%d/%Y %H:%M") - d0).total_seconds())
    train_dataset['Tweet Count'] = train_dataset['Tweet Count'].transform(
        lambda c: int(c))
    print(train_dataset.head())
    return train_dataset


def get_history(feature, age):
    pad = feature[age:].transform(lambda d: feature[0])
    return pd.concat((pad, feature[0:age]))


def get_features():
    train_dataset = get_dataset()
    max_len, w, h = len(train_dataset), 25, -10
    features = train_dataset
    #pad = train_dataset['Tweet Count'][h:].transform(lambda d: d0)
    #features['Tweet Count'] = pd.concat((pad,train_dataset['Tweet Count'][0:h]))
    #features['Tweet Count'] = history(train_dataset['Tweet Count'], -10)
    f, (f0, f1) = 2*features.shape[1], features.shape
    for i in range((f) - features.shape[1]):
        features[f"h{i}"] = get_history(train_dataset['Tweet Count'], -10-i)
    target = train_dataset['Tweet Count']
    features = np.array(features) 
    target = np.array(target)
    print(f"features.raw.shape = {features.shape}, target = {target}")
    n = int(int(f0*f1/(f**2))*(f**2)/f1)
    n1d = int(n*f1/((f**2)))
    print(f"features.n = {n}")
    #features = features.reshape((features.shape[0], f, f))
    #features = features[0:n].reshape((n1d, f, f)) 
    #target = np.array(target[n-n1d:n]).reshape((n1d, 1, 1))
    #target = np.array(target[n-n1d:n]).reshape((n1d, 1, 1))
    features = features.reshape((features.shape[0], features.shape[1], 1))
    target = target.reshape((target.shape[0], 1, 1))
    print(f"features = {features.shape}, target = {target.shape}")
    N = int(0.8*len(features))
    return features[0:N], target[0:N], features[N:], target[N:], f1


def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


def prepare_data():
    x_train, y_train, x_test, y_test, w = get_features()
    #x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
    #x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

    #x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    #x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    n_classes = max(np.concatenate((y_train, y_test)))+1

    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    #y_train[y_train == -1] = 0
    #y_test[y_test == -1] = 0
    return x_train, y_train, x_test, y_test, n_classes



def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = KfAttention(
        #key_dim=head_size, num_heads=num_heads, dropout=dropout
        head_size, num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
    n_classes=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    print(f"build_model().x = {x}, input_shape = {input_shape}")
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    print(f"build_model().pool.x = {x}")
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def train_eval(x_train, y_train, x_test, y_test, n_classes):
  input_shape = x_train.shape[1:] #x_train[0:x_train.shape[1]].shape 
  print(f"train_eval().input_shape={input_shape}, xtrain.shape={x_train.shape}")

  model = build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
    n_classes=n_classes,
  )

  model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy"],
  )
  model.summary()

  callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

  history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=100, #300,
    batch_size=64,
    callbacks=callbacks
  )

  plot(history)
  model.evaluate(x_test, y_test, verbose=1)
  save_model(model)

  get_layer(model, x_test[0:20], 3)

  return model


def save_model(model):
    path = os.path.join(os.path.dirname(__file__), "..", "data", "kfmodel")
    return model.save(path)


def get_model(data = None):
    path = os.path.join(os.path.dirname(__file__), "..", "data", "kfmodel")
    if os.path.exists(path):
        return keras.models.load_model(path)
    else:
        return train_eval(*data)


def get_layer(model, inputs, depth=3):
    print(f"get_layer().inputs = {inputs.shape}")
    value, n_layers = inputs, 3
    for i in range(depth): #len(model.layers)):
        value = model.layers[i](value, value) \
          if isinstance(model.layers[i],KfAttention) else model.layers[i](value)
        if i == depth - 1:
          print(f"get_layer(): layer {i} = {value.shape}, " \
                f"type={type(model.layers[i])}, value={tf.transpose(value)}")
    return value



def plot(history):
    print(f"history = {history.history}")
    #print(f"loss = {history.history['loss']}")
    #print(f"val_loss = {history.history['val_loss'_]}")
    plt.figure(figsize = (20, 5))
    plt.plot(history.history['loss'], label = "loss")
    plt.plot(history.history['val_loss'], label = "val_loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print("Transformer")
    #x_train, y_train, x_test, y_test, n_classes = prepare_data()
    #model = train_eval(x_train, y_train, x_test, y_test, n_classes)
    data = (prepare_data())
    attention_val = get_layer(get_model(data=data), data[2][0:20], 3)


