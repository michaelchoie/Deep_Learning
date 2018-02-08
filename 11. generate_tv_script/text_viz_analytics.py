"""Create visualizations and analytics for text data."""

import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter
from os import getcwd
from os.path import abspath, join


def get_data():
    """Retrieve data from file system."""
    file_name = 'moes_tavern_lines.txt'
    path_to_file = abspath(join(getcwd(), 'data', file_name))

    with open(path_to_file, "r") as f:
        data = f.read()

    return data


def identify_characters(data):
    """
    Get names of all characters in the script.

    Args
        data (str): text data
    """
    characters = set(re.findall("\n.*?:", data))
    characters = [x[1:] for x in characters]

    return characters


def count_characters(data, characters):
    """
    Return amount of times a character appears in text.

    Args
        data (str): text data
    """
    counts = Counter(data.split())
    character_counts = {name: freq for (name, freq) in counts.items()
                        if name in characters}

    labels, values = zip(*character_counts.items())

    # Sort ascending order
    sorted_idx = np.argsort(values)[::-1]

    labels = np.array(labels)[sorted_idx]
    values = np.array(values)[sorted_idx]

    return labels, values


def text_analytics(data):
    """
    asdf.

    Args
        data (str): text data
    """
    num_chars = len(data)
    num_words = len(data.split())
    num_lines = len(re.findall("\n", data))

    print("Number of characters in script: {} \
           Number of words in script: {} \
           Number of lines in script: {}".format(num_chars, num_words,
                                                 num_lines))


def visualize_counts(labels, values):
    """
    Visualize the # of times top 5 characters appear in text using a histogram.

    Args
        labels (str): character names
        values (str): frequency count
    """
    bar_width = 0.5
    indices = np.arange(5)

    plt.bar(indices, values[:5], width=bar_width, align="center")
    plt.xticks(indices, labels[:5])
    plt.show()


def main():
    """asdf."""
    data = get_data()
    characters = identify_characters(data)
    labels, values = count_characters(data, characters)
    text_analytics(data)
    visualize_counts(labels, values)

if __name__ == "__main__":
    main()
