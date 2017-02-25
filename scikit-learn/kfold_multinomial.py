import make_training_set
import numpy
from pprint import pprint
from pandas import DataFrame
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def test(filename="all_data.txt", size=1000):
    # trainingSet has the form: list of dictionaries
    # Each dictionary is a sample
    # with keys URL and result
    trainingSet = make_training_set.create_set(filename, size)

    data_frame = DataFrame(trainingSet)

    # count_vectorizer = CountVectorizer(analyzer="char")
    # counts = count_vectorizer.fit_transform(data_frame["url"].values)

    # classifier = MultinomialNB()
    # targets = data_frame["result"].values
    # classifier.fit(counts, targets)

    # examples = ["http://796481.finewatch2016.ru"]

    # example_counts = count_vectorizer.transform(examples)
    # predictions = classifier.predict(example_counts)
    # print predictions

    pipeline = Pipeline([('vectorizer',  CountVectorizer(analyzer="word", ngram_range = (2,4))),
                         ('classifier',  MultinomialNB()) ])

    # pipeline.fit(data_frame["url"].values, data_frame["result"].values)

    # res = pipeline.predict(examples)

    k_fold = KFold(n_splits=2)
    scores = []
    accuracies = []
    confusion = numpy.array([[0, 0], [0, 0]])

    probs = []


    for train_indices, test_indices in k_fold.split(data_frame):
        train_text = data_frame.iloc[train_indices]["url"].values
        train_y = data_frame.iloc[train_indices]["result"].values

        test_text = data_frame.iloc[test_indices]["url"].values
        test_y = data_frame.iloc[test_indices]["result"].values

        pipeline.fit(train_text, train_y)
        predictions = pipeline.predict(test_text)

        probs = pipeline.predict_proba(test_text)

        confusion += confusion_matrix(test_y, predictions)
        accuracy = accuracy_score(test_y, predictions)
        accuracies.append(accuracy)
        score = f1_score(test_y, predictions, pos_label="malicious")
        scores.append(score)

    # print 'Total URLs classified: ' + str(len(data_frame))
    # print 'Score: ' + str(sum(scores)/len(scores))
    # print 'Confusion matrix:'
    # print confusion

    # Variables are in predicted_actual order
    total = float(len(data_frame))
    clean_clean = confusion[0][0]
    mal_clean = confusion[0][1]
    clean_mal = confusion[1][0]
    mal_mal = confusion[1][1]
    # prop_caught = float(mal_mal + clean_clean)/total
    mal = mal_mal + clean_mal
    clean = clean_clean + mal_clean
    true_positive = float(mal_mal)
    false_positive = float(mal_clean)
    true_negative = float(clean_clean)
    false_negative = float(clean_mal)
    # prop_missed = float(clean_mal + mal_clean)/total
    # false_positive = float(clean_mal)/float(clean_mal + mal_mal)

    print "Total: " + str(int(total))
    print "Malware: " + str(mal)
    print "Clean: " + str(clean)
    print "Malicious Caught: " + str(mal_mal) + " (" + "{:.1%}".format(true_positive/mal) + " of all malicious samples)"
    print "Malicious Missed: " + str(clean_mal) + " (" + "{:.1%}".format(false_negative/mal) + " of all malicious samples)"
    print "Clean Caught: " + str(clean_clean) + " (" + "{:.1%}".format(true_negative/clean) + " of all clean samples)"
    print "Clean Missed: " + str(mal_clean) + " (" + "{:.1%}".format(false_positive/clean) + " of all cleansamples)"

    print probs

    return probs, test_y

    # print "Caught: " + str(mal_mal + clean_clean) + " (" + "{:.1%}".format(prop_caught) + " of all samples)"
    # print "Missed: " + str(clean_mal + mal_clean) + " (" + "{:.1%}".format(prop_missed) + " of all samples)"
    # print "Malicious missed: " + str(clean_mal) + " (" + "{:.1%}".format(false_positive) + " of all malicious samples)"

def test_all():
    return test(size=140000)
