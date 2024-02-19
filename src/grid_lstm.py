from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Bidirectional
from scipy.signal import savgol_filter
from functools import reduce
from ekf import test_pca
import numpy as np
import pandas as pd
import tensorflow as tf
import sys



class GridLSTM(LSTM):

  def __init__(self, *args, **kwargs):
      if not len(args):
          args = (32) # units = 32
      print("args = {}, kwargs = {}".format(args, kwargs))
      super(GridLSTM, self).__init__(*args, **kwargs)
      self.args, self.kwargs = args, kwargs


  def build(self, input_shape):
      print("build.input_shape = {}".format(input_shape))
      self.lstms = [Bidirectional(LSTM(*self.args, **self.kwargs)) \
                    for i in range(input_shape[1])]
      (a,b,c) = input_shape
      list(map(lambda lstm: lstm.build((a,1,c)), self.lstms))


  def build_basic(self, input_shape):
      self.w = [self.add_weight(shape=(input_shape[-1], self.units),
                               initializer='random_normal',
                               trainable=True) for i in range(input_shape[1])]
      self.b = [self.add_weight(shape=(self.units,),
                               initializer='random_normal',
                               trainable=True) for i in range(input_shape[1])]
      print("input_shape = {}, = {}, b = {}".format(
          input_shape, self.w[0].shape, self.b[0].shape))
      #super(GridLsTM, self).build(input_shape)


  def call(self, inputs):
      #print("inputs = {}, w = {}".format(inputs.shape, self.w[0].shape))
      print("inputs = {}".format(inputs[:,0:1,:].shape))
      return reduce(tf.add, map(lambda i: self.lstms[i].call(inputs[:,i:i+1,:]),
                                range(inputs.shape[1])))

  def call_basic(self, inputs):
      out = [tf.matmul(inputs[:,i,:], self.w[i]) + self.b[i] \
              for i in range(inputs.shape[1])]
      return reduce(tf.add, out)


def to_grid(x, w=50):
    flt = lambda n: int(float(n)*100)
    x = np.array([list(x[i:i+w])+list(np.zeros((max(0,i+w-len(x)),1))) \
                  for i in range(len(x))])
    x = np.array([[flt(x[i,j]) for j in range(w)] for i in range(len(x))])
    return x


def generate_from_csv(fname, xcol = '$uAppP', ycol = '$fGet_n', y1col = '$fGet',
                      predfn = None, dopca = True, pre = "", 
                      filter_savgol=False, filter_kalman=False):
    (r, rows, hdrs) = (-1, {}, [])
    flt = lambda n: int(float(n)*100)
    rows = pd.read_csv(fname)
    y = np.array([[flt(v), flt(v)] for v in rows[ycol][1:]] + [[0, 0]])
    x = np.array(to_grid(rows[xcol]))
    if filter_savgol:
        x = savgol_filter(x, 5, 2)
        x = np.array([[max(0,v) for v in r] for r in x])
        print("savgol.x = {}".format(x))
    elif filter_kalman:
        x = to_grid(np.array(test_pca()))
        print("kalman.x = {}".format(x))
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    print("X_train = {}, y_train = {}".format(X_train.shape, y_train.shape))
    print("X_test = {}, y_test = {}".format(X_test.shape, y_test.shape))
    y_train = pd.get_dummies(y_train[:,1]).values
    y_test = pd.get_dummies(y_test[:,1]).values

    return X_train, y_train[:,1], X_test, y_test[:,1]



def generate_data(test_split = 0.2, variable=False, filter_savgol=False):
    print("---Generating Data---")

    X = []
    y = []

    for i in range(10000):
        x_hold = []

        if variable:
            value = np.random.randint(2, size=np.random.randint(1, 50))
        else:
            value = np.random.randint(2, size=50)

        for x in value:
            x_hold.append(int(x))
        if sum(x_hold) % 2 == 1:
            y.append(0)
        else:
            y.append(1)
        X.append(x_hold)

    X = np.array(X)
    print("X = {}".format(X))
    if filter_savgol:
        X = savgol_filter(X, 5, 2)
    print("X1 = {}".format(X.shape))
    #exit(0)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    y_train = pd.get_dummies(y_train).values
    y_test = pd.get_dummies(y_test).values

    return X_train, y_train, X_test, y_test



def create_GridLSTM_model(rows=10000, length=50):
    print('---Creating GridLSTM model---')
    embed_dim = 128
    lstm_out = 200

    model = Sequential()
    model.add(Embedding(rows, embed_dim, input_length=length)) #, dropout=0.2))
    model.add(GridLSTM(lstm_out)) #, dropout_U = 0.2, dropout_W = 0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model



def run_test(*args, **kwargs):
    X_train, y_train, X_test, y_test = generate_data() if not len(args) else \
                                       generate_from_csv(*args, **kwargs)
    #print("x_train, x_test = {}, {}".format(X_train, X_test))

    model = create_GridLSTM_model(len(X_train), len(X_train[0]))
    model.fit(X_train, y_train, batch_size=10, epochs=10)
    err, acc = model.evaluate(X_test, y_test, batch_size=4)
    print("eval = {}".format((err, acc)))



if __name__ == "__main__":
    f = sys.argv[sys.argv.index("-f")+1] if "-f" in sys.argv else None
    xcol = sys.argv[sys.argv.index("-x")+1] if "-x" in sys.argv else 'P'
    ycol = sys.argv[sys.argv.index("-y")+1] if "-y" in sys.argv else 'P'
    sf = True if "--savgol" in sys.argv else False
    kf = True if "--kalman" in sys.argv else False
    run_test(filter_savgol=sf, filter_kalman=kf) if f is None else \
        run_test(f, xcol=xcol, ycol=ycol, filter_savgol=sf,filter_kalman=kf)



