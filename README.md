# Language Assignment 3 - Language modelling and text generation using RNNs
This is the repository for the third assignment in the course Language Analytics from the bachelors elective course Cultural Data Science at Aarhus University

## 1. Contributions
This code was written independently by me. 

## 2. Assignment description by instructor
Text generation is hot news right now!

For this assignemnt, you're going to create some scripts which will allow you to train a text generation model on some culturally significant data - comments on articles for *The New York Times*. You can find a link to the data [here](https://www.kaggle.com/datasets/aashita/nyt-comments).

You should create a **collection of scripts** which do the following:

- **Train** a model on the Comments section of the data
  - [**Save** the trained model](https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model)
- **Load** a saved model
  - **Generate** text from a user-suggested prompt

## 2.1 Objectives

Language modelling is hard and training text generation models is doubly hard. For this course, we lack somewhat the computationl resources, time, and data to train top-quality models for this task. So, if your RNNs don't perform overwhelmingly, that's fine (and expected). Think of it more as a proof of concept.

- Using TensorFlow to build complex deep learning models for NLP
- Illustrating that you can structure repositories appropriately
- Providing clear, easy-to-use documentation for your work.

## 2.2 Some tips

One big thing to be aware of - unlike the classroom notebook, this assignment is working on the *Comments*, not the articles. So two things to consider:

1) The Comments data might be structured differently to the Articles data. You'll need to investigate that;
2) There are considerably more Comments than articles - plan ahead for model training!

## 2.3 Additional pointers

- Make sure not to try to push the data to Github!
- *Do* include the **saved models** that you output
- Make sure to structure your repository appropriately
  - Include a readme explaining relevant info
    - E.g where does the data come from?
    - How do I run the code?
- Make sure to include a requirements file, etc...

## 3. Methods 
The script ```rnn_model.py``` loads in the data, preprocesses it and saves the model and vectorizer in the output folder ```out```. The model is saved as a tensorflow model and called ```text_generate_model.tf``` adn the vectorizer ```tokenizer.joblib```.
The script ```generate_text.py``` loads this saved model and vectorizer, and enables the user to choose a suggested prompt, which in will then generate text from.
## 4. Usage
These scripts were made with python 3.10.7. If you have any problems running the script, try switching to python 3.10.7.
### 4.1 Install required packages
From the command line, make sure you are located inside the main directory, then run the command `bash setup.sh` to install the required packages.
### 4.2 Get the data
The data can be retrieved [here](https://www.kaggle.com/datasets/aashita/nyt-comments).
This will download a folder called ```archive.zip```. Place the ```archive.zip```folder in the subdirectory ```in```. To unzip the ```archive.zip```, open a command line in the main directory annd run the command `unzip archive.zip`. This will unzip a folder called ```archive``` which contains the data-files. 
. 
### 4.3 Run the script 
#### 4.3.1 Save model and tokenizer
From the command line, located in the main directory, run the command `python3 src/rnn_model.py`. 
If you wish to change the paths of the data (because the data has been placed elsewhere), tokenizer output or model output, this can be done with the compiler flags:
"--data_path", "-d", to change the input path for loading the data
"--model_path", "-m", to change the output path for saving the model
"--tokenizer_path," "-t", to change the output path for saving the tokenizer
as an example: `python3 src/rnn_model.py -d data/archive -m out/model.joblib`
This will cahnge the relative path of where the data is loaded in the script from to `data/archive` and change the output path and the name of the saved model to `output/model.joblib`
beware when using the compiler flags for the saved model and tokenizer, the model need the extension '.tf' and the tokenizer the extenstion '.joblib'

#### 4.3.2 Generate text
When the model and tokenizer has been saved , you should run the command `python3 src/generate_text.py`. This will run the script with the default output paths for the tokenizer and model (```out```), default prompt ("hello, you look nice today") and default amount of words to be generated from the prompt (10).
These thing can be altered with the following flags:
"-m", "--model_path", which allows the user (you) to alter the output path of the saved model.
"-t", "--tokenizer_path", which allows the user to alter the output path of the tokenizer
"-p"--prompt", which allows the user to come up with their own prompt
"-nw",--next_words", which allows the user to decide how many words, the generated text should contain.
As an example `python3 src/generate_text.py -p "I like python as much as I like socks" -nw 20`
This would alter the prompt text and the lenght of the generated output text, but not the model-path nor the tokenizer path. 

## 5. Discussion of results
Due to the size of the dataset, the model has not been able to train properly. instead I trained on only a very small subset of the data. I have chosen a single CSV-file called ```Comments_april2027.csv```out of the 9 csv-files. Furthermore only the 1000 first comments of this was able to run. For this I altered line 34 to `all_text.extend(list(comments_df['commentBody'].head(1000).values))` instead of  `all_text.extend(list(comments_df['commentBody'].values))`. Furthermore I have chosen 10 epochs, which is in the lower end. 
This has resulted in that the model generates text rather badly. When run on the default prompt "hello, you look nice today" it generates the text: "hello, you look nice today to be a lot of the trump of the trump". This showcases that it has been trained on something, probably something very Trump-related, since no matter what prompt I've given it, it has returned something along the lines of "trump of the trump of the trump". This is in no way a very good generated text, but given the very small training sample, it makes complete sense.  
The script is set to run on all of the CSV-files and all of the text, and i'd be interested in how well it would work, if having been trained on all of the data.