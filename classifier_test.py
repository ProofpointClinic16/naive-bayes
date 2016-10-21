import parser
from pprint import pprint
from naiveBayesClassifier import tokenizer
from naiveBayesClassifier.trainer import Trainer
from naiveBayesClassifier.classifier import Classifier


def classify(filename):

    trainingSet = parser.parse(filename)

    trainer = Trainer(tokenizer.Tokenizer(stop_words = [], signs_to_remove = [""]))

    for sample in trainingSet:
        trainer.train(sample['url'], sample['result'])

    classifier = Classifier(trainer.data, tokenizer.Tokenizer(stop_words = [], signs_to_remove = [""]))

    newInstance = u'https://secure.actblue.com/contribute/page/dcccactblue?amount=25&recurring=true'

    print classifier.classify(newInstance)


def test():
    classify('sample.txt')
