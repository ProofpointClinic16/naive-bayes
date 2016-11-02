import parser
import make_sets
from pprint import pprint
from naiveBayesClassifier import tokenizer
from naiveBayesClassifier.trainer import Trainer
from naiveBayesClassifier.classifier import Classifier


def classify(filename, size):

    trainingSet, testingSet = make_sets.create_sets(filename, size)

    trainer = Trainer(tokenizer.Tokenizer(stop_words = [], signs_to_remove = [""]))


    for sample in trainingSet:
        trainer.train(sample['url'], sample['result'])

    classifier = Classifier(trainer.data, tokenizer.Tokenizer(stop_words = [], signs_to_remove = [""]))

    mal_mal = 0
    mal_clean = 0
    clean_clean = 0
    clean_mal = 0

    for sample in testingSet:

    	predicted = classifier.classify(sample['url'])[0][0]
    	actual = sample['result']

    	if predicted == 'malicious' and actual == 'malicious':
    		mal_mal += 1
    	elif predicted == 'malicious' and actual == 'clean':
    		mal_clean += 1
    	elif predicted == 'clean' and actual == 'clean':
    		clean_clean += 1
    	elif predicted == 'clean' and actual == 'malicious':
    		clean_mal += 1

    size = float(size)

    mal_mal = float(mal_mal)/size
    mal_clean = float(mal_clean)/size
    clean_mal = float(clean_mal)/size
    clean_clean = float(clean_clean)/size

    confusionMatrix = [[mal_mal, clean_mal], [mal_clean, clean_clean]]

    pprint(confusionMatrix)


def test(size):
    classify('lotsodata.txt', size)
