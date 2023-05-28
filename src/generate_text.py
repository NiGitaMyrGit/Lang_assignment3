import os
from tensorflow.keras.models import load_model
from utils.rnn_utils import generate_text
import argparse

def parse_args():

# Load the saved model
model = load_model(os.path.join("out","text_generate_model.tf"))

# User suggested promt, the text in parentheses can be modified
prompt = input("language is for humans, what math is for computers")
# generate text from prompt
generated_text = generate_text(prompt, next_words,  model, 100, tokenizer) # TODO: next_words and tokenizer
print(generated_text)

