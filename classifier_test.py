import parser
from pprint import pprint
from naiveBayesClassifier import tokenizer
from naiveBayesClassifier.trainer import Trainer
from naiveBayesClassifier.classifier import Classifier


def classify(filename):

    data = parser.parse(filename)

    trainingSet = []

    for datum in data:
        dict = {}
        dict['url'] = datum['url'].replace("/", " ")
        dict['category'] = datum['results']['result']

        trainingSet += [dict]

    trainer = Trainer(tokenizer.Tokenizer(stop_words = [], signs_to_remove = [""]))

    for sample in trainingSet:
        trainer.train(sample['url'], sample['category'])

    classifier = Classifier(trainer.data, tokenizer.Tokenizer(stop_words = [], signs_to_remove = [""]))

    newInstance = u'http://wenerdun.com'

    print classifier.classify(newInstance)


def test():
    classify('sample.txt')
