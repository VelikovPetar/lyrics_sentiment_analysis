"""
A collection of methods that are often used in other scripts.
"""
import os
import pandas


def get_anew_emotion_dictionary():
    """
    Creates a dictionary containing the 1245 words and their valence/arousal from the emotion_dictionary_anew,csv file.

    :return: dictionary in the form {'word':(valence, arousal)}
    """
    dictionary_filename = 'D:\ФИНКИ\Седми семестар\Обработка на природни јазици\lyrics_sentiment_analysis\data\words\emotion_dictionary_anew.csv'
    data = pandas.read_csv(dictionary_filename, header=0, engine='python')

    anew_emotion_dictionary = {}
    for _, row in data.iterrows():
        word = row['word']
        valence = row['valence']
        arousal = row['arousal']
        anew_emotion_dictionary[word] = (valence, arousal)

    return anew_emotion_dictionary


def get_full_dataset():
    """
    Returns the full dataset (lyrics_data.tsv) as a pandas dataframe.
    :return: pandas dataframe contaning the dataset
    """
    dataset_filename = 'D:\ФИНКИ\Седми семестар\Обработка на природни јазици\lyrics_sentiment_analysis\data\lyrics_data.tsv'
    data = pandas.read_csv(dataset_filename, header=0, sep='\t', engine='python')
    return data


def get_annotated_dataset():
    """
    Creates a dictionary of each song from the manually annotated set with the class (Q1-Q4).

    :return: dictionary in the form {'song_code':quadrant}
    """
    data = get_full_dataset()

    annotated_dataset = {}
    for _, row in data.iterrows():
        song_code = row['Code']
        valence = float(row['Avg_Valence'].replace(',', '.'))
        arousal = float(row['Avg_Arousal'].replace(',', '.'))
        quadrant = get_quadrant_for_valence_arousal(valence, arousal)
        annotated_dataset[song_code] = quadrant

    return annotated_dataset


def get_quadrant_for_valence_arousal(valence, arousal):
    """
    Returns the quadrant (1-4) for valence and arousal values.

    :param valence: the valence of the word
    :param arousal: the arousal of the word
    :return: the quadrant it belongs to
    """
    if valence >= 0:
        if arousal >= 0:
            quadrant = 1
        else:
            quadrant = 4
    else:
        if arousal >= 0:
            quadrant = 2
        else:
            quadrant = 3

    return quadrant


if __name__ == '__main__':
    print(get_anew_emotion_dictionary()['happy'])
    print(get_annotated_dataset()['L001'])
