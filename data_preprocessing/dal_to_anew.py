# Script to convert the values of the DAL (Dictionary of affect in language) scale to the
# ANEW (Affective norms for English Words) scale.


import pandas


def get_range_of_values(values):
    """
    Calculates the range of list of values.

    :param values: the list ov values to calculate the range
    :return: a tuple consisting of (min, max)
    """
    start = min(values)
    end = max(values)
    return start, end


def calculate_train_data_anew_range():
    training_data_file = '../data/lyrics_data.tsv'
    data = pandas.read_csv(training_data_file, sep='\t')
    valences = []
    arousals = []
    for _, row in data.iterrows():
        valence = float(row['Avg_Valence'].replace(',', '.'))
        arousal = float(row['Avg_Arousal'].replace(',', '.'))
        valences.append(valence)
        arousals.append(arousal)
    valence_range = get_range_of_values(valences)
    arousal_range = get_range_of_values(arousals)
    return valence_range, arousal_range


def convert_dal_to_anew_value(dal_value):
    """
    Converts value from the DAL range into the corresponding value in the ANEW range.

    :param dal_value: the value from the DAL range
    :return: the value converted to ANEW value
    """
    old_range = DAL_MAX_VALUE - DAL_MIN_VALUE
    new_range = ANEW_MAX_VALUE - ANEW_MIN_VALUE
    anew_value = ANEW_MIN_VALUE + ((dal_value - DAL_MIN_VALUE) / old_range) * new_range
    return anew_value


DAL_MIN_VALUE = 1
DAL_MAX_VALUE = 3

anew_valence_range, anew_arousal_range = calculate_train_data_anew_range()

ANEW_MIN_VALUE = min(anew_valence_range[0], anew_arousal_range[0])
ANEW_MAX_VALUE = max(anew_valence_range[1], anew_arousal_range[1])

if __name__ == '__main__':
    print("DAL range: (%d, %d)" % (DAL_MIN_VALUE, DAL_MAX_VALUE))
    print("ANEW range: (%d, %d)" % (ANEW_MIN_VALUE, ANEW_MAX_VALUE))

    anew_emotion_dictionary = []
    emotion_dictionary_file = '../data/words/emotion_dictionary.txt'
    with open(emotion_dictionary_file, mode='r') as file:
        for line in file:
            parts = line.split()
            word = parts[0]
            valence = float(parts[1])
            valence = convert_dal_to_anew_value(valence)
            arousal = float(parts[2])
            arousal = convert_dal_to_anew_value(arousal)
            new_line = '%s,%f,%f' % (word, valence, arousal)
            anew_emotion_dictionary.append(new_line)

    anew_emotion_dictionary_file = '../data/words/emotion_dictionary_anew.csv'
    with open(anew_emotion_dictionary_file, mode='w') as file:
        file.write('word,valence,arousal\n')
        for line in anew_emotion_dictionary:
            file.write(line + '\n')
