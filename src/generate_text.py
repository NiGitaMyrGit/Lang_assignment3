#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from argparse import ArgumentParser
from utils.rnn_utils import generate_text
import joblib

def main():
    default_model_path = os.path.join("out", "text_generate_model.tf")
    default_tokenizer_path = os.path.join("out", "tokenizer.joblib")

    parser = ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str, default=default_model_path, help="output path for loading the model")
    parser.add_argument("--tokenizer_path", "-t", type=str, default=default_tokenizer_path, help="output path for loading the tokenizer")
    parser.add_argument("--prompt","-p", type=str, default="hello, you look nice today", help="User-suggested-prompt")
    parser.add_argument("--next_words","-nw", type=int, default=10, help="Number of words to generate")
    
    args = parser.parse_args()

    # Load the saved model
    model = load_model(args.model_path)

    # Load the tokenizer
    tokenizer = joblib.load(args.tokenizer_path)

    # Set the maximum sequence length
    max_sequence_len = model.input_shape[1]

    # Generate text using the user-suggested prompt
    generated_text = generate_text(args.prompt, args.next_words, model, max_sequence_len, tokenizer)
    print("Generated Text:", generated_text)

if __name__ == "__main__":
    main()