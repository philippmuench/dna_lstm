import tensorflow as tf
import theano
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, LSTM, Embedding, Activation, Lambda, Bidirectional
from keras.engine import Input, Model, InputSpec
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.preprocessing import sequence
from keras.models import model_from_json
from kerastoolbox.visu import plot_weights
import os
import pydot
import graphviz

EPCOHS = 100 #  an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.
BATCH_SIZE = 10 # a set of N samples. The samples in a batch are processed` independently, in parallel. If training, a batch results in only one update to the model.
INPUT_DIM = 4 # a vocabulary of 5 words in case of fnn sequence (ATCGN)
OUTPUT_DIM = 128
RNN_HIDDEN_DIM = 128
DROPOUT_RATIO = 0.2 # proportion of neurones not used for training
MAXLEN = 500 # cuts text after number of these characters in pad_sequences
LEARNING_RATE = 0.01
checkpoint_dir ='checkpoints'
os.path.exists(checkpoint_dir)

input_file = 'train.csv'

def letter_to_index(letter):
    _alphabet = 'ATGC'
    return next((i for i, _letter in enumerate(_alphabet) if _letter == letter), None)

def load_data(test_split = 0.33, maxlen = MAXLEN):
    print ('Loading data...')
    df = pd.read_csv(input_file)
    df['sequence'] = df['sequence'].apply(lambda x: [int(letter_to_index(e)) for e in x])
    df = df.reindex(np.random.permutation(df.index))
    train_size = int(len(df) * (1 - test_split))
    X_train = df['sequence'].values[:train_size]
    y_train = np.array(df['target'].values[:train_size])
    X_test = np.array(df['sequence'].values[train_size:])
    y_test = np.array(df['target'].values[train_size:])
    print('Average train sequence length: {}'.format(np.mean(list(map(len, X_train)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, X_test)), dtype=int)))
    return pad_sequences(X_train, maxlen=maxlen), y_train, pad_sequences(X_test, maxlen=maxlen), y_test


def create_lstm_bidirectional(input_length, rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO):
    model = Sequential()
    model.add(Embedding(input_dim = INPUT_DIM, output_dim = output_dim, input_length = input_length, name='embedding_layer'))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model


def create_model(input_length, rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO):
    print ('Creating model...')
    model = Sequential()
    # we start off with an efficient embedding layer which maps our vocab indices into embedding_dims dimensions
    model.add(Embedding(input_dim = INPUT_DIM, output_dim = output_dim, input_length = input_length, name='embedding_layer'))
    model.add(Bidirectional(LSTM(units=rnn_hidden_dim, dropout = dropout, recurrent_dropout= dropout, return_sequences=True, name='recurrent_layer')))
   # model.add(LSTM(units=rnn_hidden_dim, dropout = dropout, recurrent_dropout= dropout, return_sequences=True, name='recurrent_layer2'))
    model.add(Bidirectional(LSTM(units=rnn_hidden_dim, dropout = dropout, recurrent_dropout= dropout, return_sequences=True, name='recurrent_layer3')))
    model.add(Lambda(lambda x: x[:, -1, :], output_shape=(rnn_hidden_dim, ), name='last_step_layer'))
    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1, activation='sigmoid', name='output_layer'))
    print ('Compiling...')
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=LEARNING_RATE),
                  metrics=['accuracy'])
    return model

def create_plots(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy.png')
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')
    plt.clf()

if __name__ == '__main__':
    # train
    X_train, y_train, X_test, y_test = load_data()
    print(type(X_train))
    print(X_train.shape)
   # model = create_model(len(X_train[0]))
    model = create_lstm_bidirectional(len(X_train[0])) 
    # checkpoint
    filepath= checkpoint_dir + "/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    print ('Fitting model...')
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPCOHS, callbacks=callbacks_list, validation_split = 0.1, verbose = 1)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    create_plots(history)
    plot_model(model, to_file='model.png')

    # summarize history for loss
    score, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print('Test score:', score)
    print('Test accuracy:', acc)