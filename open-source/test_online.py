import make_training_set
from pprint import pprint
from naiveBayesClassifier import tokenizer
from naiveBayesClassifier.trainer import Trainer
from naiveBayesClassifier.classifier import Classifier
from tabulate import tabulate


# The size parameter defines how many samples will be in the training/testing set each
def classify(filename, size):

    trainingSet = make_training_set.create_set(filename, size)

    trainer = Trainer(tokenizer.Tokenizer(stop_words = [], signs_to_remove = [""]))

    mal_mal = 0
    mal_clean = 0
    clean_clean = 0
    clean_mal = 0

    trainer.train(trainingSet[0]['url'], trainingSet[0]['result'])

    classifier = Classifier(trainer.data, tokenizer.Tokenizer(stop_words = [], signs_to_remove = [""]))

    out = open("mislabeled.txt", "w")

    for sample in trainingSet[1:]:

        predicted = classifier.classify(sample['url'])[0][0]
        actual = sample['result']

        if predicted == 'malicious' and actual == 'malicious':
            mal_mal += 1
        elif predicted == 'malicious' and actual == 'clean':
            mal_clean += 1
        elif predicted == 'clean' and actual == 'clean':
            clean_clean += 1
        elif predicted == 'clean' and actual == 'malicious':
            out.write(sample['url'] + '\n')
            clean_mal += 1

        trainer.train(sample['url'], sample['result'])

        classifier = Classifier(trainer.data, tokenizer.Tokenizer(stop_words = [], signs_to_remove = [""]))

    total = float(mal_mal + mal_clean + clean_mal + clean_clean)
    prop_caught = float(mal_mal + clean_clean)/total
    prop_missed = float(clean_mal + mal_clean)/total
    false_positive = float(clean_mal)/float(mal_mal + clean_mal)

    ## Stuff to get proportions:

    # size = float(size)

    # mal_mal = float(mal_mal)/size
    # mal_clean = float(mal_clean)/size
    # clean_mal = float(clean_mal)/size
    # clean_clean = float(clean_clean)/size

    ## Confusion matrix stuff:

    # confusionMatrix = [['Actually malicious', mal_mal, clean_mal], ['Actually clean', mal_clean, clean_clean]]

    # print tabulate(confusionMatrix, headers=['', 'Predicted malicious', 'Predicted clean'])

    print "Total: " + str(int(total))
    print "Malware: " + str(mal_mal + clean_mal)
    print "Clean: " + str(mal_clean + clean_clean)
    print "Caught: " + str(mal_mal + clean_clean) + " (" + "{:.1%}".format(prop_caught) + " of all samples)"
    print "Missed: " + str(clean_mal + mal_clean) + " (" + "{:.1%}".format(prop_missed) + " of all samples)"
    print "Malicious missed: " + str(clean_mal) + " (" + "{:.1%}".format(false_positive) + " of all malicious samples)"


def test(size):
    classify('all_data.txt', size)


def test_all():
    test(140000)
