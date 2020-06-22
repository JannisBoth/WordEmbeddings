import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import GRU

def encode_and_pad_seqs(train_docs, test_docs):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_docs)
    
    vocab_size = len(tokenizer.word_index) + 1
    
    max_length = max([len(s.split()) for s in train_docs])

    # sequence encode
    encoded_docs_train = tokenizer.texts_to_sequences(train_docs)
    encoded_docs_test = tokenizer.texts_to_sequences(test_docs)    

    Xtrain = pad_sequences(encoded_docs_train, maxlen=max_length, padding='post')
    Xtest = pad_sequences(encoded_docs_test, maxlen=max_length, padding='post')
    
    return tokenizer.word_index, Xtrain, Xtest, vocab_size, max_length

def load_embedding(file_path):
    with open (file_path,'r', encoding ="utf-8") as f:
        embedding = dict()
        lines = f.readlines()
        for line in lines:
            parts = line.split()
            embedding[parts[0]] = np.asarray(parts[1:], dtype='float32')
        return embedding

def get_weight_matrix(embedding, word_index, vocab_size, vocab):
    weight_matrix = np.zeros((vocab_size, 100))
    
    for word, i in word_index.items():
        if word in embedding.keys():
            weight_matrix[i] = embedding.get(word)
            
    return weight_matrix


def create_embedding_layer(file_path, vocab_size, max_length, tokenizer_index, vocab):
    raw_embedding = load_embedding(file_path)
    embedding_vectors = get_weight_matrix(raw_embedding, tokenizer_index, vocab_size, vocab)
    return Embedding(vocab_size, 100, weights=[embedding_vectors], input_length=max_length, trainable=False)


def create_model(model_params, embedding_layer):
    p = model_params
    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(p["filters"],
                     p["kernel_size"],
                     activation=p["conv1D_activation"]))
    model.add(MaxPooling1D(p["pool_size"]))
    model.add(Flatten())
    model.add(Dense(p["dense_units"],
                    activation=p["dense_acitvation"]))

    print(model.summary())
    # compile network
    model.compile(loss=p["loss"],
                  optimizer=p["optimzer"],
                  metrics=p["metrics"])
    return model
    
def evaluate_model(model, Xtest, ytest, metrics):
    
    returns = model.evaluate(Xtest, ytest, verbose=0)
    loss = returns[0]
    metric_values = returns[1:]
    
    assert len(metric_values) == len(metrics), "Different count of metric values and metric names."
    
    print("Test Loss:", loss)
    for metric, metric_value in zip(metrics, metric_values):
        print("Test {}: {}".format(metric, round(metric_value*100,2)))