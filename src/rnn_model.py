#!/usr/bin/env python3
# data processing tools
import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

# importing argument parser
import argparse
from argparse import ArgumentParser
parser = ArgumentParser()

# ensuring reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# surpress warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
from utils import rnn_utils as rnf

# load and preprocess data
def load_data():
    data_dir = os.path.join("in", "subset")
    all_text = []
    for filename in os.listdir(data_dir):
        if 'Comments' in filename:
            comments_df = pd.read_csv(os.path.join(data_dir, filename))
            all_text.extend(list(comments_df['commentBody'].head(1000).values)) # TODO fix head to param?
    # clean up data
    all_text = [h for h in all_text if h != "Unknown"]
    #call out ```clean_text()``` function
    corpus = [rnf.clean_text(x) for x in all_text]

    return corpus

# Preprocessing, tokenize data

def tokenise_data(corpus):
    # using ```Tokenizer()``` class from ```TensorFlow```, read more here: (https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer).
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    # using ```sequence_of_tokens()``` function from the rnf_utils script
    # turns every text into a sequence of tokens based on the vocabulary from the tokenizer.
    input_sequences = rnf.sequence_of_tokens(tokenizer, corpus)
    max_sequence_len = max([len(x) for x in input_sequences])
    # padding input sequences to make them the same length
    predictors, label, max_sequence_len = rnf.generate_padded_sequences(input_sequences, total_words)
    training_data = (predictors, label, tokenizer, max_sequence_len, total_words)
    return training_data, tokenizer

# Using the ```create_model()``` function to initialize a model,
# Can be trained provided with the length of sequences and the total size of the vocabulary.
def create_train_model(training_data):
    predictors, label, tokenizer, max_sequence_len, total_words = training_data
    model = rnf.create_model(max_sequence_len, total_words)
    model.summary()

    # train model
    history = model.fit(predictors, 
                    label, 
                    epochs=100, #TODO IS 100 TOO MUCH?
                    batch_size=128, 
                    verbose=1)
    return history, model

def main():
    #load and preprocess data
    corpus = load_data()
    #Tokenize
    training_data, tokenizer = tokenise_data(corpus)
    #generate model
    history, model = create_train_model(training_data)
    #save model
    tf.keras.models.save_model(model, os.path.join("out", "text_generate_model.tf"))
    # TODO: load model

#calling main function
if __name__== "__main__":
    main()