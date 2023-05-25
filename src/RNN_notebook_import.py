#!/usr/bin/env python3
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
from utils.rnn_utils import clean_text, sequence_of_tokens, generate_padded_sequences, create_model, generate_text

# ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# load and preprocess data
def load_data():
    data_dir = os.path.join("in")
    all_text = []
    for filename in os.listdir(data_dir):
        if 'Comments' in filename:
            comments_df = pd.read_csv(data_dir + filename)
            all_text.extend(list(comments_df.values))
    # clean up data
    all_text = [h for h in all_text if h != "Unknown"]
    #call out ```clean_text()``` function
    corpus = [clean_text(x) for x in all_text]

    return corpus

# Tokenize data
def tokenise_data(corpus):
    # using ```Tokenizer()``` class from ```TensorFlow```, read more here: (https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer).
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    # using ```sequence_of_tokens()``` function from the RNN_utils script
    # turns every text into a sequence of tokens based on the vocabulary from the tokenizer.
    input_sequences = sequence_of_tokens(tokenizer, corpus)
    max_sequence_len = max(len(x) for x in input_sequences)
    # padding input sequences to make them the same length
    predictors, label, max_sequence_len = generate_padded_sequences(input_sequences)
    training_data = (predictors, label, tokenizer, max_sequence_len, total_words)
    return training_data

if __name__ == "__main__":
    # load and preprocess data
    corpus = load_data()
    # tokenize data
    training_data = tokenise_data(corpus)
    predictors, label, tokenizer, max_sequence_len, total_words = training_data
    # create model
    model = create_model(max_sequence_len, total_words)
    # train model
    model.fit(predictors, label, epochs=100, verbose=1)
    # generate text
    seed_text = "I love"
    generated_text = generate_text(seed_text, 10, model, max_sequence_len, tokenizer)
    print(generated_text)
