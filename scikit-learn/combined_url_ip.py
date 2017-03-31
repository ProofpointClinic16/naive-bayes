import kfold_multinomial
import online_multinomial
import kfold_multinomial_ip
import numpy
import json
from pprint import pprint
from pandas import DataFrame
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def test(filename="all_data.txt", size=10000):
    # trainingSet has the form: list of dictionaries
    # Each dictionary is a sample
    # with keys URL and result

    # url_probs = online_multinomial.test_fixed()
    url_probs, url_res = online_multinomial.test_fixed()
    ip_probs, ip_res = kfold_multinomial.test_all(type="ip")

    ip_res = ip_res[1000:]

    probs = []

    print len(url_probs) 
    print len(ip_probs)

    for i in range(len(url_probs)):
        probs.append([-url_probs[i][0], -url_probs[i][1], -ip_probs[1000 + i][0], -ip_probs[1000 + i][1]])
        # the reason for these offsets, is because the online NB begins classifying after 1000 samples,
        # so offset accordingly if you decide to change the initial number of samples to train on in the online
        # classifier

    data_frame = DataFrame(probs)

    pipeline = Pipeline([('classifier',  MultinomialNB())])

    model1 = pipeline.fit(probs[:len(probs) / 2], ip_res[:(len(probs) / 2)])

    res = (model1.predict_log_proba(probs[len(probs) / 2:])).tolist()

    model2 = pipeline.fit(probs[len(probs) / 2:], ip_res[(len(probs) / 2):])

    res += (model2.predict(probs[:len(probs) / 2])).tolist()

    f = open('nb_probs.txt', 'w')
    json.dump(res, f)
    f.close()

    confusion = metrics.confusion_matrix(ip_res[(len(probs) / 2):], res[(len(res) / 2)])

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

    print "Total: " + str(len(url_probs))
    print "Malware: " + str(mal)
    print "Clean: " + str(clean)
    print "Malicious Caught: " + str(mal_mal) + " (" + "{:.1%}".format(true_positive/mal) + " of all malicious samples)"
    print "Malicious Missed: " + str(clean_mal) + " (" + "{:.1%}".format(false_negative/mal) + " of all malicious samples)"
    print "Clean Caught: " + str(clean_clean) + " (" + "{:.1%}".format(true_negative/clean) + " of all clean samples)"
    print "Clean Missed: " + str(mal_clean) + " (" + "{:.1%}".format(false_positive/clean) + " of all cleansamples)"


def test_all():
    test(size=137000)
