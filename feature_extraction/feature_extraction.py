import os
import string


def get_all_words(filename):
    """
    Extracts all words from a song lyrics. Takes into consideration each word and removes the punctuation.

    :param filename: the filename (.txt) containing the song lyrics
    :return: list containing each word of the song
    """
    if not os.path.exists(filename):
        raise Exception('File %s not found.' % filename)

    remove_punctuation_map = dict.fromkeys(map(ord, string.punctuation))
    words = []
    with open(filename, mode='r') as file:
        for line in file:
            words = words + line.translate(remove_punctuation_map).lower().split()
    return words


if __name__ == '__main__':
    filename = '../data/raw_lyrics/5 Seconds of Summer - Amnesia'
    print(get_all_words(filename))