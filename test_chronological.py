import make_chronological_sets
from pprint import pprint
from naiveBayesClassifier import tokenizer
from naiveBayesClassifier.trainer import Trainer
from naiveBayesClassifier.classifier import Classifier
from tabulate import tabulate


# The size parameter defines how many samples will be in the training/testing set each
def classify(filename, size):

    trainingSet, testingSet = make_chronological_sets.create_sets(filename, size)

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

    # size = float(size)

    # mal_mal = float(mal_mal)/size
    # mal_clean = float(mal_clean)/size
    # clean_mal = float(clean_mal)/size
    # clean_clean = float(clean_clean)/size

    confusionMatrix = [['Actually malicious', mal_mal, clean_mal], ['Actually clean', mal_clean, clean_clean]]

    print tabulate(confusionMatrix, headers=['', 'Predicted malicious', 'Predicted clean'])
    # print "Accuracy: " + str(mal_mal + clean_clean)
    # print "False positives (predicted clean when malicious): " + str(clean_mal)
    # print "False negatives (predicted malicious when clean): " + str(mal_clean)


def test(size):
    classify('lotsodata.txt', size)
