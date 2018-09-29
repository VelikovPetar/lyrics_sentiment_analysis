import os

import pandas

import common_utils
from feature_extraction.feature_extraction import get_all_words

ANNOTATED_DATASET = {}
ANEW_EMOTION_DICTIONARY = common_utils.get_anew_emotion_dictionary()


def calculate_avg_valence_arousal(word_list):
    """
    Calculates the average valence and arousal of words in a list using the values defined in
    emotion_dictionary_anew.csv.
    Ignores the words that are not in the ANEW dictionary.

    :param word_list: the input list of words
    :return: tuple containing the average valence and the average arousal of the words in the list, and a tuple
    containing the number of included words and the number of total words
    """
    total_words = len(word_list)
    included_words = 0
    valence_total = 0
    arousal_total = 0
    for word in word_list:
        if word in ANEW_EMOTION_DICTIONARY.keys():
            valence, arousal = ANEW_EMOTION_DICTIONARY[word]
            valence_total += valence
            arousal_total += arousal
            included_words += 1
    valence_avg = 0 if included_words == 0 else (valence_total / included_words)
    arousal_avg = 0 if included_words == 0 else (arousal_total / included_words)
    return (valence_avg, arousal_avg), (included_words, total_words)


if __name__ == '__main__':

    data_dir = '../data/full'

    total = 0
    matches = 0

    data_frame = pandas.read_csv('../data/full/dataset.csv', header=None)
    for _, row in data_frame.iterrows():
        text_id = row[0]
        label = row[1]
        ANNOTATED_DATASET[text_id] = label

    for song_name in os.listdir(data_dir):
        if song_name.endswith('.csv'):
            continue
        words = get_all_words(data_dir + '/' + song_name)
        (avg_valence, avg_arousal), (included_words, total_words) = calculate_avg_valence_arousal(words)
        calculated_quadrant = common_utils.get_quadrant_for_valence_arousal(avg_valence, avg_arousal)
        expected_quadrant = ANNOTATED_DATASET[song_name.replace('.txt', '')]
        if calculated_quadrant == expected_quadrant:
            matches += 1
        total += 1

    print('Matches: %d' % matches)
    print('Total: %d' % total)
    print('Accuracy:\t%.3f%%' % (matches / total * 100))
