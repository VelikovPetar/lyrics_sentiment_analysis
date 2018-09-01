import os
import string

import common_utils
import feature_extraction.feature_extraction as fe

ANNOTATED_DATASET = common_utils.get_annotated_dataset()
ANEW_EMOTION_DICTIONARY = common_utils.get_anew_emotion_dictionary()

remove_punctuation_map = dict.fromkeys(map(ord, string.punctuation))


def calculate_dominant_emotion_for_verse(verse):
    dominant_distance_from_center = -1
    dominant_point = (0, 0)
    word_list = verse.translate(remove_punctuation_map).lower().split()
    for word in word_list:
        if word in ANEW_EMOTION_DICTIONARY.keys():
            point = ANEW_EMOTION_DICTIONARY[word]
            distance = common_utils.euclidean_distance(point, (0, 0))
            if distance > dominant_distance_from_center:
                dominant_distance_from_center = distance
                dominant_point = point

    if dominant_distance_from_center > -1:
        predicted_quadrant = common_utils.get_quadrant_for_valence_arousal(dominant_point[0], dominant_point[1])
        return predicted_quadrant
    else:
        return -1


def calculate_dominant_emotion_full_song(word_list):
    dominant_distance_from_center = -1
    dominant_point = (0, 0)
    for word in word_list:
        if word in ANEW_EMOTION_DICTIONARY.keys():
            point = ANEW_EMOTION_DICTIONARY[word]
            distance = common_utils.euclidean_distance(point, (0, 0))
            if distance > dominant_distance_from_center:
                dominant_distance_from_center = distance
                dominant_point = point

    predicted_quadrant = common_utils.get_quadrant_for_valence_arousal(dominant_point[0], dominant_point[1])
    return predicted_quadrant


def get_max_with_index(list):
    max_val = list[0]
    idx = 0
    for i in range(1, len(list)):
        val = list[i]
        if val > max_val:
            max_val = val
            idx = i
    return idx, max_val


if __name__ == '__main__':

    data_dir = '../data/raw_lyrics'

    # Dominant emotion word for full song
    total = 0
    matches = 0

    for song_name in os.listdir(data_dir):
        filename = data_dir + '/' + song_name
        words = fe.get_all_words(filename)
        dominant_emotion_quadrant = calculate_dominant_emotion_full_song(words)
        expected_quadrant = ANNOTATED_DATASET[song_name.replace('.txt', '')]
        if dominant_emotion_quadrant == expected_quadrant:
            matches += 1
        total += 1

    print('Matches: %d' % matches)
    print('Total: %d' % total)
    print('Accuracy:\t%.3f%%' % (matches / total * 100))

    # Dominant emotion word per lyrics
    total = 0
    matches = 0

    for song_name in os.listdir(data_dir):
        # print(song_name)
        filename = data_dir + '/' + song_name
        lines = fe.get_all_lines(filename)
        quadrants_num = [0, 0, 0, 0, 0]
        for verse in lines:
            verse_emotion = calculate_dominant_emotion_for_verse(verse)
            if verse_emotion > -1:
                quadrants_num[verse_emotion] = quadrants_num[verse_emotion] + 1
        # print(quadrants_num[1:])
        predicted_quadrant, _ = get_max_with_index(quadrants_num)
        # print(predicted_quadrant)
        expected_quadrant = ANNOTATED_DATASET[song_name.replace('.txt', '')]
        if predicted_quadrant == expected_quadrant:
            matches += 1
        total += 1

    print('Matches: %d' % matches)
    print('Total: %d' % total)
    print('Accuracy:\t%.3f%%' % (matches / total * 100))
