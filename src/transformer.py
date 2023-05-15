# Derived from https://keras.io/examples/timeseries/timeseries_transformer_classification/
import numpy as np, tensorflow as tf
import matplotlib.pyplot as plt
import os, random, time
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
#from kfattention import MultiHeadedAttention


def get_features():
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
    #print(f"end = {np.unique(train_dataset['End'])}")
    #print(f"start = {np.unique(train_dataset['Start'])}")
    print(train_dataset.head())

    max_len, w = len(train_dataset), 25
    #features = train_dataset.drop(['Tweet Count'], axis=1)
    features, d0 = train_dataset, train_dataset['Tweet Count'][0]
    pad = train_dataset['Tweet Count'][-10:].transform(lambda d: d0)
    features['Tweet Count'] = pd.concat((pad, train_dataset['Tweet Count'][0:-10]))
    target = train_dataset['Tweet Count']
    features = np.array(features) #[0:max_len]
    target = np.array(target) #.argsort() # converting counts to ranks: use 2 argsorts
    #target = target.argsort() # ranks 
    print(f"features.raw.shape = {features.shape}, target = {target}")
    features = features.reshape((features.shape[0], features.shape[1], 1))    
    target = np.array(target).reshape((target.shape[0], 1))
    print(f"features = {features.shape}, target = {target.shape}")
    N = int(0.8*len(features))
    return features[0:N], target[0:N], features[N:], target[N:], features.shape[1]


def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


def prepare():
    root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

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
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
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
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)
    #return keras.Model(inputs, transformer_encoder1(inputs, head_size, num_heads, ff_dim, dropout))


def train_eval(x_train, y_train, x_test, y_test, n_classes):
  input_shape = x_train.shape[1:]

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
    epochs=200,
    batch_size=64,
    callbacks=callbacks
  )

  plot(history)
  #model.evaluate(x_test, y_test, verbose=1)

  value = x_test[0:20]
  #value = model.layers[0](x_test)
  #print(f"UnitTest: model = {model}, L = {len(model.layers)}, x_test = {x_test.shape}, attention = {value.shape}")
  for i in range(3): #len(model.layers)):
      value = model.layers[i](value, value) if isinstance(model.layers[i], layers.MultiHeadAttention) else model.layers[i](value)
      print(f"UnitTest: layer {i} = {value.shape}, type = {type(model.layers[i])}, value.T = {tf.transpose(value)}")
  print("UnitTest: done")

  return model


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
    #x_train, y_train, x_test, y_test, w = get_features()
    x_train, y_train, x_test, y_test, n_classes = prepare()
    train_eval(x_train, y_train, x_test, y_test, n_classes)

