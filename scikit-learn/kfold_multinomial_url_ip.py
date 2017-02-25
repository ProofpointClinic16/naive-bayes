import kfold_multinomial
import kfold_multinomial_ip
import numpy
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

    url_probs, url_res = kfold_multinomial.test_all()
    ip_probs, ip_res = kfold_multinomial_ip.test_all()

    probs = []

    for i in range(len(url_probs)):
        probs.append([url_probs[i][0], ip_probs[i][0]])

    pipeline = Pipeline([('classifier',  MultinomialNB())])

    model = pipeline.fit(probs[(len(probs) / 2):], url_res[(len(url_res) / 2):])

    res = model.predict(probs[:(len(probs) / 2)])

    err = 0
    mal = 0
    for i, prediction in enumerate(res):
        if ip_res[i] == 'malicious':
            mal += 1
            if prediction == 'clean':
                err += 1

    print metrics.classification_report(url_res[:2 * (len(probs) / 3)], res)

def test_all():
    test(size=140000)
