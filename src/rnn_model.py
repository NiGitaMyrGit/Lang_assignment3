#!/usr/bin/env python3
# data processing tools
import sys
import string, os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import joblib

# importing argument parser
import argparse
from argparse import ArgumentParser
parser = ArgumentParser()
default_model_path = os.path.join("out", "text_generate_model.tf")
default_tokenizer_path = os.path.join("out", "tokenizer.joblib")
default_data_path = os.path.join("in", "archive")
parser.add_argument("--data_path", "-d", type=str, default=default_data_path, help="input path for loading the data")
parser.add_argument("--model_path", "-m", type=str, default=default_model_path, help="output path for saving the model")
parser.add_argument("--tokenizer_path", "-t", type=str, default=default_tokenizer_path, help="output path for saving the tokenizer")
# ensuring reproducibility after H2G2 standards
np.random.seed(42)
tf.random.set_seed(42)

# surpress warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
from utils import rnn_utils as rnf

# load and preprocess data
def load_data(args):
    all_text = []
    for filename in os.listdir(args.data_path):
        if 'Comments' in filename:
            comments_df = pd.read_csv(os.path.join(args.data_path, filename))
            all_text.extend(list(comments_df['commentBody'].values))
    # clean up data
    all_text = [str(h) for h in all_text if h != "Unknown"]
    # call the `clean_text()` utils function
    corpus = [rnf.clean_text(x) for x in all_text]

    return corpus

# Preprocessing, tokenize data

def tokenise_data(corpus):
    # using ```Tokenizer()``` class from ```TensorFlow```, read more here: (https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer).
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    # turns every text into a sequence of tokens based on the vocabulary from the tokenizer.
    # by calling the ```sequence_of_tokens()``` utils function
    input_sequences = rnf.sequence_of_tokens(tokenizer, corpus)
    max_sequence_len = max([len(x) for x in input_sequences])
    # padding input sequences to make them the same length
    # calling the ```generate_padded_sequences``` utils function
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
    history = model.fit(
                predictors,
                label,
                epochs=10, #TODO how many epochs are fitting? 
                batch_size=128, 
                verbose=1)
    return history, model


def main():
    args = parser.parse_args()
    #load and preprocess data
    corpus = load_data(args)
    #Tokenize
    training_data, tokenizer = tokenise_data(corpus)
    #generate model
    history, model = create_train_model(training_data)
    #save model
    tf.keras.models.save_model(model, args.model_path)
    # save the tokenizer
    joblib.dump(tokenizer, args.tokenizer_path)

#calling main function
if __name__== "__main__":
    main()