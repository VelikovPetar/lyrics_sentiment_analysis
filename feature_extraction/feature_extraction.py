import os
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

STEMMER = PorterStemmer()


def get_all_words(filename):
    """
    Extracts all words from a song lyrics. Takes into consideration each word and removes the punctuation and
    non-alphabetic words.

    :param filename: the filename (.txt) containing the song lyrics
    :return: list containing each word of the song
    """
    if not os.path.exists(filename):
        raise Exception('File %s not found.' % filename)

    remove_punctuation_map = dict.fromkeys(map(ord, string.punctuation.replace("'", "")))
    words = []
    with open(filename, mode='r') as file:
        for line in file:
            words = words + line.translate(remove_punctuation_map).lower().split()
    return words


def remove_stopwords(words_list):
    """
    Removes all stopwords from a list
    :param words_list: the list of words
    :return: same list without the stopwords
    """
    stopwords_eng = stopwords.words('english')
    clean_words_list = []
    for word in words_list:
        if word not in stopwords_eng:
            clean_words_list.append(word)
    return clean_words_list


def get_all_lines(filename):
    """
    Returns a list of each line from a file
    :param filename: the filename (.txt) containing the song lyrics
    :return: list containing each non-empty line
    """
    if not os.path.exists(filename):
        raise Exception('File %s not found.' % filename)

    lines = []
    with open(filename, mode='r') as file:
        for line in file:
            if len(line.strip()) > 0:
                lines.append(line.replace('\n', ''))
    return lines


def remove_tailing_s(word_list):
    """
    Removes each 's from words. ex. "my mother's house" becomes "my mother house".
    :param word_list: list of words
    :return: same list of words with removed tailing 's
    """
    words = []
    for word in word_list:
        if word.endswith("'s"):
            word = word[:-2]
        if len(word) > 0:
            words.append(word)
    return words


def correct_shortened_gerund(pos_tagged_word_list):
    """
    Corrects the usage of a shortened gerund: ex. 'sittin' -> 'sitting', 'sippin' -> 'sipping'
    :param pos_tagged_word_list: list of words
    :return: same list of words with corrected gerunds
    """
    words = []
    for word in pos_tagged_word_list:
        if word.endswith("'"):
            word = word[:-1] + 'g'
        words.append(word)
    return words


def pos_tag(word_list):
    return nltk.pos_tag(word_list)


def get_vana_words(pos_tagged_word_list):
    """
    Returns a filtered list containing only VANA words - Verbs, Adjectives, Nouns, Adverbs
    :param pos_tagged_word_list: a list of POS tagged words
    :return: list containing only VANA POS tagged words
    """
    allowed_vana_tags = [
        'VB',  # verb, base form
        'VBD',  # verb, past tense
        'VBG',  # verb, gerund/present participle
        'VBN',  # verb, past participle
        'VBP',  # verb, sing. present, non-3rd person
        'VBZ',  # verb, 3rd person sing. present
        'JJ',  # adjective
        'JJR',  # adjective, comparative
        'JJS',  # adjective, superlative
        'NN',  # noun, singular
        'NNS',  # noun plural
        'NNP',  # proper noun, singular
        'NNPS',  # proper noun, plural
        'RB',  # adverb
        'RBR',  # adverb, comparative
        'RBS',  # adverb, superlative
    ]
    vana_words = []
    for (word, tag) in pos_tagged_word_list:
        if tag in allowed_vana_tags:
            vana_words.append((word, tag))
    return vana_words


def get_svana_words(pos_tagged_word_list):
    """
        Returns a filtered list containing only Selected-VANA words - Verbs, Adjectives, Nouns.
        Doesn't take into consideration Proper nouns and adverbs, and removes auxiliary verbs
        :param pos_tagged_word_list: a list of POS tagged words
        :return: list containing only SVANA POS tagged words
        """
    allowed_svana_tags = [
        'VB',  # verb, base form
        'VBD',  # verb, past tense
        'VBG',  # verb, gerund/present participle
        'VBN',  # verb, past participle
        'VBP',  # verb, sing. present, non-3rd person
        'VBZ',  # verb, 3rd person sing. present
        'JJ',  # adjective
        'JJR',  # adjective, comparative
        'JJS',  # adjective, superlative
        'NN',  # noun, singular
        'NNS',  # noun plural
        # 'NNP',  # proper noun, singular
        # 'NNPS',  # proper noun, plural
        # 'RB',  # adverb
        # 'RBR',  # adverb, comparative
        # 'RBS',  # adverb, superlative
    ]
    auxiliary_verbs = ['be', 'am', 'are', 'is', 'was', 'were', 'being', 'been', 'can', 'could', 'dare', 'do', 'does',
                       'did', 'have', 'has', 'had', 'having', 'may', 'might', 'must', 'need', 'ought', 'shall',
                       'should', 'will', 'would']
    svana_words = []
    for (word, tag) in pos_tagged_word_list:
        if tag in allowed_svana_tags and word not in auxiliary_verbs:
            svana_words.append((word, tag))
    return svana_words


def remove_pos_tags(pos_tagged_word_list):
    """
    Removes POS tags from a list
    :param pos_tagged_word_list: a list of POS tagged words
    :return: the same list without pos tags
    """
    return [pos[0] for pos in pos_tagged_word_list]


def stem_words(words_list):
    """
    Stems each word in a list
    :param words_list: the list of words
    :return: the list with stemmed words
    """
    return [STEMMER.stem(word) for word in words_list]


if __name__ == '__main__':
    filename = '../data/raw_lyrics/L001.txt'
    words = get_all_words(filename)
    # print(words)
    # print(pos_tag(words))
    print(get_all_lines(filename))
