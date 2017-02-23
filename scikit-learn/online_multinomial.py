import make_training_set
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def test_fixed(filename="all_data.txt", size=137000):
    # trainingSet has the form: list of dictionaries
    # Each dictionary is a sample
    # with keys URL and result

    # Size is fixed to 140k -- using all samples!

    # Iteration size can be chosen, usually 1000
    iteration = 1000
    trainingSet = make_training_set.create_set(filename, size)

    data_frame = DataFrame(trainingSet)

    pipeline = Pipeline([('vectorizer',  CountVectorizer(analyzer="char", ngram_range = (4,4))),
                         ('classifier',  MultinomialNB()) ])

    clean_clean = 0
    clean_mal = 0
    mal_clean = 0
    mal_mal = 0

    # Start by getting first however many samples (given by iteration)
    train_text = data_frame.iloc[xrange(iteration)]["url"].values
    train_y = data_frame.iloc[xrange(iteration)]["result"].values

    train_malicious = np.array([], dtype=object)
    train_malicious_y = np.array([], dtype=object)

    malicious_indices = []

    # Need to know how far along data set we are
    current_index = iteration

    # Train on first however many samples
    pipeline.fit(train_text, train_y)

    # To get entire data set, will need to do this size/iteration times
    # E.g. 140000/1000 = 140 times
    for i in xrange(size/iteration-1):

        # Each iteration, test on the next however many samples
        for j in xrange(iteration):
            test_text = [data_frame.iloc[current_index+j]["url"]]
            test_y = [data_frame.iloc[current_index+j]["result"]]

            prediction = pipeline.predict(test_text)

            # Keep track of how good it was
            if prediction[0] == 'clean' and test_y[0] == 'clean':
                clean_clean += 1
            elif prediction[0] == 'clean' and test_y[0] == 'malicious':
                clean_mal += 1
            elif prediction[0] == 'malicious' and test_y[0] == 'clean':
                mal_clean += 1
            elif prediction[0] == 'malicious' and test_y[0] == 'malicious':
                mal_mal += 1

        # Now let's look at all the malicious ones we trained from, and keep them around
        for k in xrange(iteration):
            if train_y[k] == 'malicious':
                malicious_indices += [k]

        train_malicious = np.append(train_malicious, np.take(train_text, malicious_indices))
        train_malicious_y = np.append(train_malicious_y, np.take(train_y, malicious_indices))

        if train_malicious.size > 500:
            train_malicious = train_malicious[-500:]
            train_malicious_y = train_malicious_y[-500:]

        # Having tested on 1000 samples, now train on them
        new_train = data_frame.iloc[xrange(current_index, current_index+iteration)]["url"].values
        new_y = data_frame.iloc[xrange(current_index, current_index+iteration)]["result"].values

        current_index += iteration

        print "End of", current_index

        pipeline.fit(np.append(new_train, train_malicious), np.append(new_y, train_malicious_y))

    # Variables are in predicted_actual order
    mal = mal_mal + clean_mal
    clean = clean_clean + mal_clean
    true_positive = float(mal_mal)
    false_positive = float(mal_clean)
    true_negative = float(clean_clean)
    false_negative = float(clean_mal)

    print "Total: " + str(size)
    print "Malware: " + str(mal)
    print "Clean: " + str(clean)
    print "True positives: " + str(mal_mal) + " (" + "{:.1%}".format(true_positive/mal) + " of all malicious samples)"
    print "False negatives: " + str(clean_mal) + " (" + "{:.1%}".format(false_negative/mal) + " of all malicious samples)"
    print "True negatives: " + str(clean_clean) + " (" + "{:.1%}".format(true_negative/clean) + " of all clean samples)"
    print "False positives: " + str(mal_clean) + " (" + "{:.1%}".format(false_positive/clean) + " of all clean samples)"


# def test_all():
#     test(size=140000)
