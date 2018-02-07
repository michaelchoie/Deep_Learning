# TV Script Generator

## Overview
By using machine learning techniques such as word2vec and seq2seq, we can teach a recurrent neural network how to write a TV script by training it on sample Simpsons text. The following modules prepare and process our text data.

## Key Files
preprocess.py - in charge of creating an embedding layer, punctuation tokenization, functions to handle pickle files <br />
script_generator.py - in charge of creating the RNN, training the network, and producing a script <br />
test_file.py - in charge of testing functions in the modules above

## Getting Started

To get started with this, make sure you set up your local environment
1. Download the script data using the following bash script:
```
mkdir -p data/simpsons
cd data/simpsons
curl https://raw.githubusercontent.com/udacity/deep-learning/master/tv-script-generation/data/simpsons/moes_tavern_lines.txt > moes_tavern_lines.txt
```
2. Install appropriate packages into a virtual environment
```
pip install virtualenv
cd <your_project_folder>
virtualenv my_project
source my_project/bin/activate
pip install -r requirements.txt
```

## How to run
First, preprocess the data by running the script preprocess.py <br />
Then run the script_generator.py file
```
python preprocess.py && python script_generator.py
```

## Notes (and possible areas of improvement)
- Can reduce number of passed arguments to class methods, as some parameters have been made into class attributes, but this would involve a big change to both the script_generator.py as well as the test_file.py modules. Not doing that for the interest of time. <br />
- I run into errors trying to refer to seq2seq.sequence_loss tensor by name; using the tf.Graph.get_tensor_by_name() method doesn't work, so I made a work around by passing the tensor itself. <br />
