import make_training_set
import numpy
from pandas import DataFrame
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def test(filename="all_data.txt", size=1000, type="url"):
    # trainingSet has the form: list of dictionaries
    # Each dictionary is a sample
    # with keys URL and result
    trainingSet = make_training_set.create_set(filename, size, type)

    data_frame = DataFrame(trainingSet)

    analyze = "char"
    if type == "ip":
        analyze = "word"

    pipeline = Pipeline([('vectorizer',  CountVectorizer(analyzer=analyze, ngram_range = (2,4))),
                         ('classifier',  MultinomialNB()) ])

    k_fold = KFold(n_splits=2)
    scores = []
    accuracies = []
    confusion = numpy.array([[0, 0], [0, 0]])

    for train_indices, test_indices in k_fold.split(data_frame):
        train_text = data_frame.iloc[train_indices][type].values
        train_y = data_frame.iloc[train_indices]["result"].values

        test_text = data_frame.iloc[test_indices][type].values
        test_y = data_frame.iloc[test_indices]["result"].values

        pipeline.fit(train_text, train_y)
        predictions = pipeline.predict(test_text)

        confusion += confusion_matrix(test_y, predictions)
        accuracy = accuracy_score(test_y, predictions)
        accuracies.append(accuracy)
        score = f1_score(test_y, predictions, pos_label="malicious")
        scores.append(score)



    # Variables are in predicted_actual order
    total = float(len(data_frame))
    clean_clean = confusion[0][0]
    mal_clean = confusion[0][1]
    clean_mal = confusion[1][0]
    mal_mal = confusion[1][1]

    mal = mal_mal + clean_mal
    clean = clean_clean + mal_clean
    true_positive = float(mal_mal)
    false_positive = float(mal_clean)
    true_negative = float(clean_clean)
    false_negative = float(clean_mal)

    print "Total: " + str(int(total))
    print "Malware: " + str(mal)
    print "Clean: " + str(clean)
    print "True positives: " + str(mal_mal) + " (" + "{:.1%}".format(true_positive/mal) + " of all malicious samples)"
    print "False negatives: " + str(clean_mal) + " (" + "{:.1%}".format(false_negative/mal) + " of all malicious samples)"
    print "True negatives: " + str(clean_clean) + " (" + "{:.1%}".format(true_negative/clean) + " of all clean samples)"
    print "False positives: " + str(mal_clean) + " (" + "{:.1%}".format(false_positive/clean) + " of all cleansamples)"


def test_all(type="url"):
    test(size=140000, type=type)
