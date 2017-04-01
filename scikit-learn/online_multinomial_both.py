import make_training_set
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def test_fixed(filename="all_data.txt", size=137000, mal_num=500, iteration=1000):
    # trainingSet has the form: list of dictionaries
    # Each dictionary is a sample
    # with keys URL and result, or IP and result

    # Size is fixed to 137k -- using all samples!

    trainingSet_ip = make_training_set.create_set(filename, size, "ip")
    trainingSet_url = make_training_set.create_set(filename, size, "url")

    data_frame_ip = DataFrame(trainingSet_ip)
    data_frame_url = DataFrame(trainingSet_url)


    pipeline_url = Pipeline([('vectorizer',  CountVectorizer(analyzer="char", ngram_range = (4,4))),
                             ('classifier',  MultinomialNB()) ])

    pipeline_ip = Pipeline([('vectorizer',  CountVectorizer(analyzer="word", ngram_range = (2,2))),
                            ('classifier',  MultinomialNB()) ])

    clean_clean = 0
    clean_mal = 0
    mal_clean = 0
    mal_mal = 0

    # Start by getting first however many samples (given by iteration)
    train_text_ip = data_frame_ip.iloc[xrange(iteration)]["ip"].values
    train_text_url = data_frame_url.iloc[xrange(iteration)]["url"].values

    # This should be the same for IP and URL data sets
    train_y = data_frame_ip.iloc[xrange(iteration)]["result"].values

    # Initialize empty data sets to keep around the extra malicious samples
    train_malicious_ip = np.array([], dtype=object)
    train_malicious_url = np.array([], dtype=object)

    train_malicious_y = np.array([], dtype=object)

    malicious_indices = []

    # For visualization purposes
    tps, tns, fps, fns = [], [], [], []

    # Need to know how far along data set we are
    current_index = iteration

    # Train on first however many samples
    pipeline_ip.fit(train_text_ip, train_y)
    pipeline_url.fit(train_text_url, train_y)

    # To get entire data set, will need to do this size/iteration times
    # E.g. 140000/1000 = 140 times
    howmany = xrange(size/iteration-1)

    for i in howmany:

        # For this iteration, reset current variables
        current_tp, current_tn, current_fp, current_fn, current_clean, current_mal = 0, 0, 0, 0, 0, 0

        # Each iteration, test on the next however many samples
        for j in xrange(iteration):
            test_text_ip = [data_frame_ip.iloc[current_index+j]["ip"]]
            test_text_url = [data_frame_url.iloc[current_index+j]["url"]]

            test_y = [data_frame_ip.iloc[current_index+j]["result"]]

            prediction_ip_clean = pipeline_ip.predict_proba(test_text_ip)[0][0]
            prediction_url_clean = pipeline_url.predict_proba(test_text_url)[0][0]

            if prediction_ip_clean + prediction_url_clean < 1.0:
                prediction = 'clean'
            else:
                prediction = 'malicious'

            # print (prediction_ip_clean + prediction_url_clean)

            # Keep track of how good it was overall as well as this iteration
            if prediction == 'clean' and test_y[0] == 'clean':
                clean_clean += 1
                current_tn += 1
                current_clean += 1
            elif prediction == 'clean' and test_y[0] == 'malicious':
                clean_mal += 1
                current_fn += 1
                current_mal += 1
            elif prediction == 'malicious' and test_y[0] == 'clean':
                mal_clean += 1
                current_fp += 1
                current_clean += 1
            elif prediction == 'malicious' and test_y[0] == 'malicious':
                mal_mal += 1
                current_tp += 1
                current_mal += 1

        # Now let's look at all the malicious ones we trained from, and keep them around
        for k in xrange(iteration):
            if train_y[k] == 'malicious':
                malicious_indices += [k]

        train_malicious_ip = np.append(train_malicious_ip, np.take(train_text_ip, malicious_indices))
        train_malicious_url = np.append(train_malicious_url, np.take(train_text_url, malicious_indices))

        train_malicious_y = np.append(train_malicious_y, np.take(train_y, malicious_indices))

        if train_malicious_ip.size > mal_num:
            train_malicious_ip = train_malicious_ip[-mal_num:]
            train_malicious_url = train_malicious_url[-mal_num:]

            train_malicious_y = train_malicious_y[-mal_num:]

        # Having tested on 1000 samples, now train on them
        new_train_ip = data_frame_ip.iloc[xrange(current_index, current_index+iteration)]["ip"].values
        new_train_url = data_frame_url.iloc[xrange(current_index, current_index+iteration)]["url"].values

        new_y = data_frame_ip.iloc[xrange(current_index, current_index+iteration)]["result"].values

        # Finally, for visualization purposes, keep track of how we did in this iteration
        tps += [float(current_tp)/current_mal]
        fps += [float(current_fp)/current_clean]
        tns += [float(current_tn)/current_clean]
        fns += [float(current_fn)/current_mal]

        current_index += iteration

        print "End of", current_index
        print "Got tp rate", float(current_tp)/current_mal

        pipeline_ip.fit(np.append(new_train_ip, train_malicious_ip), np.append(new_y, train_malicious_y))
        pipeline_url.fit(np.append(new_train_url, train_malicious_url), np.append(new_y, train_malicious_y))

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

    # Visualize the rates over time
    # plt.figure(1)
    # plt.plot(howmany, tps, 'ro')
    # plt.ylabel('True positive rate')
    # plt.xlabel('Iteration')
    # plt.title('True positive rates against iteration')

    # plt.figure(2)
    # plt.plot(howmany, fns, 'bo')
    # plt.ylabel('False negative rate')
    # plt.xlabel('Iteration')
    # plt.title('False negative rates against iteration')

    # plt.figure(3)
    # plt.plot(howmany, tns, 'go')
    # plt.ylabel('True negative rate')
    # plt.xlabel('Iteration')
    # plt.title('True negative rates against iteration')

    # plt.figure(4)
    # plt.plot(howmany, fps, 'mo')
    # plt.ylabel('False positive rate')
    # plt.xlabel('Iteration')
    # plt.title('False positive rates against iteration')

    # plt.show()
