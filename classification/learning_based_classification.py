import pandas

import feature_extraction.feature_extraction as fe
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

TRAIN_FILENAME = '../data/full/train.csv'
TEST_FILENAME = '../data/full/test.csv'

DATA_DIRECTORY = '../data/full'

MAX_FEATURES = 5000

CORPUS = []
VECTORIZER = CountVectorizer(max_features=MAX_FEATURES)


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
    words = fe.stem_words(words)
    return words


def train_test_classifier(clf_name, classifier, train_x, train_y, test_x, test_y):
    classifier.fit(train_x, train_y)
    prediction = classifier.predict(test_x)
    accuracy = 0
    for i in range(0, len(prediction)):
        if prediction[i] == test_y[i]:
            accuracy += 1
    print('Accuracy[%s]: %d/%d' % (clf_name, accuracy, len(test_y)))
    print('Accuracy[%s]: %.5f' % (clf_name, float(accuracy) / float(len(test_y))))


if __name__ == '__main__':
    # Create train set
    train_x = []
    train_y = []
    data_frame = pandas.read_csv(TRAIN_FILENAME, header=None)
    for _, row in data_frame.iterrows():
        text_id = row[0]
        label = row[1]
        words = get_processed_words(text_id)
        train_x.append(' '.join(words))
        train_y.append(label)
    # Create bag of words for the train data
    train_x = VECTORIZER.fit_transform(train_x).toarray()

    # Create test set
    test_x = []
    test_y = []
    data_frame = pandas.read_csv(TEST_FILENAME, header=None)
    for _, row in data_frame.iterrows():
        text_id = row[0]
        label = row[1]
        words = get_processed_words(text_id)
        test_x.append(' '.join(words))
        test_y.append(label)
    # Create bag of words for the test data
    test_x = VECTORIZER.transform(test_x).toarray()

    # Test classifiers
    # todo test other classifiers
    classifier = GaussianNB()
    train_test_classifier('GaussianNB', classifier, train_x, train_y, test_x, test_y)
