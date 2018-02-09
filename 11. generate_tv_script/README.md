# TV Script Generator

## Overview
By using machine learning techniques such as word2vec and seq2seq, we can teach a recurrent neural network how to write a TV script by training it on sample Simpsons text. The following modules prepare and process our text data.

## Key Files
__preprocess.py__ - in charge of creating an embedding layer, punctuation tokenization, functions to handle pickle files <br />
__script_generator.py__ - in charge of creating the RNN, training the network, and producing a script <br />
__test_file.py__ - in charge of testing functions in the modules above

## Getting Started
To get started with this, make sure you set up your local environment
1. Download the script data using the following bash script:
```
mkdir -p data/simpsons
cd data/simpsons
curl https://raw.githubusercontent.com/udacity/deep-learning/master/tv-script-generation/data/simpsons/moes_tavern_lines.txt > moes_tavern_lines.txt
```
2. Install appropriate packages into a virtual environment <br />
Make sure your current directory does not contain any spaces, otherwise you will not be able to pip install the libraries.
```
pip install virtualenv
cd <your_project_folder>
virtualenv my_project
source my_project/bin/activate
pip install -r requirements.txt
```

## How to run
In order to produce an output, first run __preprocess.py__ to prep the data, and then run __script_generator.py__ to produce the script.
```
python preprocess.py && python script_generator.py
```

## Data Summary
This text data is a script from a particular scene in an episode from the hit TV show,  "The Simpsons." <br />
The following analytics and visualizations were generated in the __text_viz_analytics.py__ script:

Size of text file: 298 kb <br />
Number of characters in script: 305,270 <br />
Number of words in script: 48,986 <br />
Number of lines in script: 4,519 <br />

The following visualization highlights the main characters of the scene based off the amount of times they are mentioned in the script. Naturally, as the file is called "Moe's Tavern Lines," we would expect that the main characters of this episode to be the characters that frequent Moe's tavern the most. The histogram verifies this initial assumption. <br />
![alt text](https://github.com/michaelchoie/Deep_Learning/blob/master/11.%20generate_tv_script/top_characters.png)

By producing a wordcloud, we can get a sense of how characters communicated throughout the scene. <br />
In order to produce this wordcloud, I removed all stopwords, punctuation, the names that denoted who was speaking at each line (i.e Homer_Simpson: "blah blah blah"), and words with a length less than 3. My rationale behind this was that this subset of the text would contain more meaningful words. <br />
I ran into a lot of difficulty trying to install the wordcloud package in Python, so I did a workaround using R. To achieve this, I created a sorted dictionary whose keys were words and values were the respective counts, and then I saved that into a JSON file. I loaded this JSON into R and used the wordcloud library to produce a visualization. My R code is as follows: <br />
```
library(rjson)
library(wordcloud)

word_counts <- fromJSON(file="word_counts.json")

topwords <- unlist(word_counts)[seq(1, 100, 2)]
topcounts <- as.integer(unlist(word_counts)[seq(2, 101, 2)])

wordcloud(topwords, topcounts, colors=brewer.pal(8,"Dark2"))
```
And the output is shown below. This wordcloud contains the top 50 words of the subsetted tv script data.
![alt text](https://github.com/michaelchoie/Deep_Learning/blob/master/11.%20generate_tv_script/wordcloud.png)

Although there are some words in this visualization that contain little information such as "hey" or "yeah", the wordcloud also contains words that verify that the text data was indeed a bar scene. For example, it contains words like "bar", "drink", "money", and we can presume that the main characters speaking are Moe and Homer. This fact is vetted by the histogram above.

## Model Strategy
Word2vec, RNN, LSTM cells, Seq2Seq. <br />
Although these modules can be run on a local computer, I used an AWS GPU cluster to streamline the training process.

## Modeling Performance
The produced TV script doesn't make much sense, but that is to be expected. We trained on less than a megabyte of text. In order to get good results, you'll have to use a smaller vocabulary or get more data. While this data is a subset of a larger dataset, for demonstration purposes, this particular dataset will suffice. The point is that the script was able to generate somewhat cohesive English.

## Notes
- I run into errors trying to refer to seq2seq.sequence_loss tensor by name; using the tf.Graph.get_tensor_by_name() method doesn't work, so I made a work around by passing the tensor itself.
