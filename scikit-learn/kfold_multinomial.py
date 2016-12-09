import make_training_set
import numpy
from pprint import pprint
from pandas import DataFrame
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
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

    pipeline = Pipeline([('vectorizer',  CountVectorizer(analyzer="char", ngram_range = (4,4))),
                         ('classifier',  MultinomialNB()) ])

    # pipeline.fit(data_frame["url"].values, data_frame["result"].values)

    # res = pipeline.predict(examples)

    k_fold = KFold(n=len(data_frame), n_folds=6)
    scores = []
    accuracies = []
    confusion = numpy.array([[0, 0], [0, 0]])

    for train_indices, test_indices in k_fold:
        train_text = data_frame.iloc[train_indices]["url"].values
        train_y = data_frame.iloc[train_indices]["result"].values

        test_text = data_frame.iloc[test_indices]["url"].values
        test_y = data_frame.iloc[test_indices]["result"].values

        pipeline.fit(train_text, train_y)
        predictions = pipeline.predict(test_text)

        confusion += confusion_matrix(test_y, predictions)
        accuracy = accuracy_score(test_y, predictions)
        accuracies.append(accuracy)
        score = f1_score(test_y, predictions, pos_label="malicious")
        scores.append(score)

    # print 'Total URLs classified: ' + str(len(data_frame))
    # print 'Score: ' + str(sum(scores)/len(scores))
    # print 'Confusion matrix:'
    # print confusion

    total = float(len(data_frame))
    clean_clean = confusion[0][0]
    mal_clean = confusion[0][1]
    clean_mal = confusion[1][0]
    mal_mal = confusion[1][1]
    prop_caught = float(mal_mal + clean_clean)/total
    prop_missed = float(clean_mal + mal_clean)/total
    false_positive = float(clean_mal)/float(clean_mal + mal_mal)

    print "Total: " + str(int(total))
    print "Malware: " + str(mal_mal + clean_mal)
    print "Clean: " + str(mal_clean + clean_clean)
    print "Caught: " + str(mal_mal + clean_clean) + " (" + "{:.1%}".format(prop_caught) + " of all samples)"
    print "Missed: " + str(clean_mal + mal_clean) + " (" + "{:.1%}".format(prop_missed) + " of all samples)"
    print "Malicious missed: " + str(clean_mal) + " (" + "{:.1%}".format(false_positive) + " of all malicious samples)"

def test_all():
    test(size=140000)
