from keras.layers import Dense, Dropout, LSTM, Embedding, Activation, Lambda
from keras.engine import Input, Model, InputSpec
from keras.preprocessing.sequence import pad_sequences
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.models import model_from_json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

test_file = 'test.csv'
model_file = 'model.json'
model_weights = 'model.h5'

def letter_to_index(letter):
    _alphabet = 'ATGC'
    return next((i for i, _letter in enumerate(_alphabet) if _letter == letter), None)

def load_test():
    print ('Loading data...')
    df = pd.read_csv(test_file)
    df['sequence'] = df['sequence'].apply(lambda x: [int(letter_to_index(e)) for e in x])
    df = df.reindex(np.random.permutation(df.index))
    sample = df['sequence'].values[:len(df)]
    return pad_sequences(sample)

print ('Loading model...')
json_file = open(model_file, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(model_weights)
print("Loaded model from disk")

print('Predict samples...')
X_case = load_test()
y_case = loaded_model.predict(X_case, batch_size=128, verbose=0)
print('prediction:', y_case)
