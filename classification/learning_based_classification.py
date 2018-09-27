import pandas
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

import common_utils
import feature_extraction.feature_extraction as fe

DATASET_FILENAME = '../data/full/dataset.csv'

DATA_DIRECTORY = '../data/full'

MAX_FEATURES = 2500

CORPUS = []
VECTORIZER = CountVectorizer(max_features=MAX_FEATURES)

CLASSIFIERS = {
    'GaussianNB': GaussianNB(),
    'MultinomialNB': MultinomialNB(),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'LogisticRegression': LogisticRegression(),
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=100),
    'AdaBoost': AdaBoostClassifier(),
    'MLP': MLPClassifier(max_iter=500),
    'SVC(linear, C=0.025)': SVC(kernel="linear", C=0.025)
}

ANEW_EMOTION_DICTIONARY = common_utils.get_anew_emotion_dictionary()


def text_id_to_filename(text_id):
    """
    Creates the full filename for the text_id
    :param text_id: the id
    :return: the full filename
    """
    return DATA_DIRECTORY + '/' + text_id + '.txt'


def get_processed_words(text_id):
    """
    Calculates features for the text identified by the text_id
    :param text_id: identifier of the text
    :return: feature set
    """
    filename = text_id_to_filename(text_id)
    words = fe.get_all_words(filename)
    words = fe.remove_tailing_s(words)
    words = fe.correct_shortened_gerund(words)
    words = fe.remove_stopwords(words)
    # Use POS tagging to exclude irrelevant words
    words = fe.pos_tag(words)
    words = fe.get_svana_words(words)
    # Remove unnecessary POS tags
    words = fe.remove_pos_tags(words)
    words = fe.stem_words(words, "PorterStemmer")
    return words


def train_test_classifier(clf_name, classifier, train_x, train_y, test_x, test_y):
    classifier.fit(train_x, train_y)
    prediction = classifier.predict(test_x)
    accuracy = 0
    for i in range(0, len(prediction)):
        if prediction[i] == test_y[i]:
            accuracy += 1
    print('Accuracy[%s]: %d/%d (%.5f)' % (clf_name, accuracy, len(test_y), float(accuracy) / float(len(test_y))))
    print(f1_score(test_y, prediction, labels=[1, 2, 3, 4], average='micro'))


def predict_label_probabilities(classifier, test_instance):
    probabilities = classifier.predict_proba([test_instance])
    return probabilities[0]


def evaluate_classifier_kfold(clf_name, classifier, X, y, k):
    scores = cross_val_score(classifier, X, y, cv=k, verbose=1)
    print(clf_name + ':', end='\t')
    print("Accuracy: %0.5f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
    # prediction = cross_val_predict(classifier, X, y, cv=k)
    # print('Accuracy[%s]: %.5f' % (clf_name, accuracy_score(y, prediction)))


def get_avg_valence_arousal_for_quadrant(words, quadrant):
    """
    Calculates the average valence/arousal of the words in the text that belong to the provided quadrant,
    and calculates the number of words in the text that belong to the provided quadrant.
    :param words: the word list(text)
    :param quadrant: the quadrant
    :return: the average valence and arousal of the words in the text that belong to the quadrant,
    and the number of words in the text that belong to the quadrant
    """
    included_words = 0
    valence_total = 0
    arousal_total = 0
    for word in words:
        if word in ANEW_EMOTION_DICTIONARY.keys():
            valence, arousal = ANEW_EMOTION_DICTIONARY[word]
            q = common_utils.get_quadrant_for_valence_arousal(valence=valence, arousal=arousal)
            if q == quadrant:
                valence_total += valence
                arousal_total += arousal
                included_words += 1
    valence_avg = 0 if included_words == 0 else (valence_total / included_words)
    arousal_avg = 0 if included_words == 0 else (arousal_total / included_words)
    return valence_avg, arousal_avg, included_words


def get_avg_valence_arousal_for_text(words):
    """
    Calculates the average valence and arousal of words in a list using the values defined in
    emotion_dictionary_anew.csv.
    Ignores the words that are not in the ANEW dictionary.
    :param words: the input list of words
    :return: tuple containing the average valence and the average arousal of the words in the list, and a tuple
    containing the number of included words and the number of total words
    """
    included_words = 0
    valence_total = 0
    arousal_total = 0
    for word in words:
        if word in ANEW_EMOTION_DICTIONARY.keys():
            valence, arousal = ANEW_EMOTION_DICTIONARY[word]
            valence_total += valence
            arousal_total += arousal
            included_words += 1
    valence_avg = 0 if included_words == 0 else (valence_total / included_words)
    arousal_avg = 0 if included_words == 0 else (arousal_total / included_words)
    return valence_avg, arousal_avg


def get_keywords_based_features(text_id):
    """
    Calculates the keywords based features for the text identified by text_id
    :param text_id: identifier of the text
    :return: feature vector consisting of (AQ1, VQ1, #Q1, AQ2, VQ2, #Q2, AQ3, VQ3, #Q3, AQ4, VQ4, #Q4, AQ1234, VQ1234)
    """
    # Pre-process the words
    words = fe.get_all_words(text_id_to_filename(text_id))
    words = fe.remove_tailing_s(words)
    words = fe.correct_shortened_gerund(words)
    words = fe.remove_stopwords(words)
    # Use POS tagging to exclude irrelevant words
    words = fe.pos_tag(words)
    words = fe.get_vana_words(words)
    # Remove unnecessary POS tags
    words = fe.remove_pos_tags(words)

    features = []
    features += get_avg_valence_arousal_for_quadrant(words, quadrant=1)
    features += get_avg_valence_arousal_for_quadrant(words, quadrant=2)
    features += get_avg_valence_arousal_for_quadrant(words, quadrant=3)
    features += get_avg_valence_arousal_for_quadrant(words, quadrant=4)
    features += get_avg_valence_arousal_for_text(words)
    return features


def classification_using_keywords_based_features():
    # Load full data
    data_frame = pandas.read_csv(DATASET_FILENAME, header=None)
    X = []
    y = []
    for _, row in data_frame.iterrows():
        text_id = row[0]
        label = row[1]
        X.append(get_keywords_based_features(text_id))
        y.append(label)

    # Scale data to range [0, 1]
    min_max_scaler = MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    # Evaluate each classifier using 10-fold cross validation
    for classifier_name in CLASSIFIERS.keys():
        evaluate_classifier_kfold(classifier_name, CLASSIFIERS[classifier_name], X, y, 10)


def classification_using_words():
    # Load full data
    data_frame = pandas.read_csv(DATASET_FILENAME, header=None)
    X = []
    y = []
    for _, row in data_frame.iterrows():
        text_id = row[0]
        label = row[1]
        words = get_processed_words(text_id)
        X.append(' '.join(words))
        y.append(label)

    # Separate data into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Create BoW for the train data
    X_train = VECTORIZER.fit_transform(X_train).toarray()

    # Create BoW for the test data
    X_test = VECTORIZER.transform(X_test).toarray()

    # Create BoW for full data-set
    X = VECTORIZER.fit_transform(X).toarray()
    # Evaluate each classifier using 10-fold cross validation
    for classifier_name in CLASSIFIERS.keys():
        evaluate_classifier_kfold(classifier_name, CLASSIFIERS[classifier_name], X, y, 10)
        # train_test_classifier(classifier_name, CLASSIFIERS[classifier_name], X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    classification_using_words()
    # classification_using_keywords_based_features()
