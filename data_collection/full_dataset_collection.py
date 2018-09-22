import os
import random

import pandas

import common_utils
import feature_extraction.feature_extraction as fe


TRAIN_FILENAME = '../data/full/train.csv'
TEST_FILENAME = '../data/full/test.csv'


def create_id(number):
    return 'AS%03d' % number


def separate_and_id_annotated_sentences_and_labels():
    if os.path.exists('../data/raw_sentences'):
        return

    annotated_sentences_file_1 = '../data/annotated_sentences/Dataset-129_Sentences.txt'
    annotated_sentences_file_2 = '../data/annotated_sentences/Dataset-239_Sentences.txt'
    lines = fe.get_all_lines(annotated_sentences_file_1) + fe.get_all_lines(annotated_sentences_file_2)
    raw_sentences_dir = '../data/raw_sentences'
    if not os.path.exists(raw_sentences_dir):
        os.mkdir(raw_sentences_dir)
    ind = 1
    for line in lines:
        outfile = raw_sentences_dir + '/' + create_id(ind) + '.txt'
        with open(outfile, mode='w') as file:
            file.write(line)
        ind += 1

    labels_1 = '../data/annotated_sentences/Dataset-129_Sentences-Classes.txt'
    labels_2 = '../data/annotated_sentences/Dataset-239_Sentences-Classes.txt'
    lines = fe.get_all_lines(labels_1) + fe.get_all_lines(labels_2)
    labels_output_file = '../data/raw_sentences/labels.csv'
    ind = 1
    with open(labels_output_file, mode='w') as file:
        file.write('id,label\n')
        for line in lines:
            file.write(create_id(ind) + ',' + line + '\n')
            ind += 1


def get_full_dataset():
    """
    Returns the full data set as dictionaries: ['id':[(word, POS tag)...]] and ['id':label]
    :return:
    """
    # Expand the contents files 'Dataset-129_Sentences.txt' and 'Dataset-239_Sentences.txt' to a separate file for each
    # sentence. Each of these files is given an unique id. These files are then stored in the 'data/raw_sentences'
    # directory. Create a single labels.csv file containing info about the labels of the sentences.
    separate_and_id_annotated_sentences_and_labels()

    raw_lyrics_directory = '../data/raw_lyrics'
    raw_sentences_directory = '../data/raw_sentences'

    annotated_lyrics = common_utils.get_annotated_dataset()

    words_dataset = {}
    labels = {}

    # Retrieve all words from each song lyrics from the 'data/raw_lyrics' directory and populate the 'words_dataset' and
    # 'labels' dictionaries.
    for file_name in os.listdir(raw_lyrics_directory):
        song_words = fe.get_all_words(raw_lyrics_directory + '/' + file_name)
        song_words = fe.remove_tailing_s(song_words)
        song_words = fe.correct_shortened_gerund(song_words)
        song_words = fe.get_svana_words(fe.pos_tag(song_words))
        key = file_name.replace('.txt', '')
        words_dataset[key] = song_words
        labels[key] = annotated_lyrics[key]

    # Retrieve all words from each sentence from the 'data/raw_sentences directory and populate the 'words_dataset'
    # dictionary.
    for file_name in os.listdir(raw_sentences_directory):
        if not file_name.endswith('.txt'):
            continue
        words = fe.get_all_words(raw_sentences_directory + '/' + file_name)
        words = fe.remove_tailing_s(words)
        words = fe.correct_shortened_gerund(words)
        words = fe.get_svana_words(fe.pos_tag(words))
        words_dataset[file_name.replace('.txt', '')] = words

    # Iterate ove the '/data/raw_sentences/labels.csv' file, and populate the 'labels' dictionary.
    annotated_sentences_labels_file = '../data/raw_sentences/labels.csv'
    annotated_sentences_labels = pandas.read_csv(annotated_sentences_labels_file, header=0)
    for _, row in annotated_sentences_labels.iterrows():
        id = row['id']
        label = row['label']
        labels[id] = label.replace('Q', '')

    return words_dataset, labels


def separate_set_to_train_test(labels, train_ratio):
    """
    Create .csv files with data: (text_id, label) for train and test sets.
    :param labels: dictionary containing data about each text label
    :param train_ratio: the ration of the train set
    """
    train_set = {}
    test_set = {}
    random.seed(0)

    for key in labels.keys():
        val = random.uniform(0, 1)
        if val <= train_ratio:
            train_set[key] = labels[key]
        else:
            test_set[key] = labels[key]

    with open(TRAIN_FILENAME, mode='w') as file:
        for key in train_set.keys():
            file.write('%s,%s\n' % (key, train_set[key]))

    with open(TEST_FILENAME, mode='w') as file:
        for key in test_set.keys():
            file.write('%s,%s\n' % (key, test_set[key]))


if __name__ == '__main__':

    words_dataset, labels = get_full_dataset()
    separate_set_to_train_test(labels, 0.75)

    # IT'S CORRECT!
    print(len(words_dataset.keys()))
    print(words_dataset['AS368'])
    print(words_dataset['L001'])
    print(len(labels.values()))
