from keras.layers import Dense, Dropout, LSTM, Embedding, Activation, Lambda
from keras.engine import Input, Model, InputSpec
from keras.preprocessing.sequence import pad_sequences
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.models import model_from_json
from keras import backend as K
from sklearn import decomposition
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

def visualize_model(model, include_gradients=False):
    recurrent_layer = model.get_layer('recurrent_layer')
    output_layer = model.get_layer('output_layer')
    inputs = []
    inputs.extend(model.inputs)
    outputs = []
    outputs.extend(model.outputs)
    outputs.append(recurrent_layer.output)
    outputs.append(recurrent_layer.trainable_weights[1])  # -- weights of the forget gates (assuming LSTM)
    if include_gradients:
        loss = K.mean(model.output)  # [batch_size, 1] -> scalar
        grads = K.gradients(loss, recurrent_layer.output)
        grads_norm = grads / (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        outputs.append(grads_norm)
    all_function = K.function(inputs, outputs)
    output_function = K.function([output_layer.input], model.outputs)
    return all_function, output_function

def get_compare_embeddings(original_embeddings, tuned_embeddings, vocab, dimreduce_type="pca", random_state=0):
    """ Compare embeddings drift. """
    if dimreduce_type == "pca":
        from sklearn.decomposition import PCA
        dimreducer = PCA(n_components=2, random_state=random_state)
    elif dimreduce_type == "tsne":
        from sklearn.manifold import TSNE
        dimreducer = TSNE(n_components=2, random_state=random_state)
    else:
        raise Exception("Wrong dimreduce_type.")
    reduced_original = dimreducer.fit_transform(original_embeddings)
    reduced_tuned = dimreducer.fit_transform(tuned_embeddings)
    def compare_embeddings(word):
        if word not in vocab:
            return None
        word_id = vocab[word]
        original_x, original_y = reduced_original[word_id, :]
        tuned_x, tuned_y = reduced_tuned[word_id, :]
        return original_x, original_y, tuned_x, tuned_y
    return compare_embeddings


print ('Loading model...')
json_file = open(model_file, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(model_weights)
print("Loaded model from disk")

print('Predict samples...')
X = load_test()
y = model.predict(X, batch_size=128, verbose=0)
print('prediction:', y)

# visualize
all_function, output_function = visualize_model(model, include_gradients=True)
#cores, rnn_values, rnn_gradients, W_i = all_function([X])
#print(scores.shape, rnn_values.shape, rnn_gradients.shape, W_i.shape)

#time_distributed_scores = map(lambda x: output_function([x]), rnn_values)
#print("Time distributed (word-level) scores:", map(lambda x: x[0], time_distributed_scores))

#embeddings = model.get_weights()[0]
#compare_embeddings = get_compare_embeddings(embeddings, embeddings, vocab, dimreduce_type="pca", random_state=0)
#print("Embeddings drift:", compare_embeddings('d'))
