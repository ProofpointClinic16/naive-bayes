import make_training_set
from pprint import pprint
from naiveBayesClassifier import tokenizer
from naiveBayesClassifier.trainer import Trainer
from naiveBayesClassifier.classifier import Classifier
from tabulate import tabulate


urls = ['http://rjb4sv7cu.biz/%7egr/page/PC:465479457bff7ce5ab8e/IC:REZVWhiLCsAn1IbwXQ90/IP:MjA3OQ==/?guid=ON', 'http://www.simultanews.com', 'http://www.ideliver-inc.com/privacy-policy/', 'https://click.e1.victoriassecret.com/?qs=ec8a199b115b94397306f89c763c315f87302479109ffb901805d9fe2b8a592364cbd294505fdd55', 'http://ehtx769.b47f2gxv.com/U3khVLcviyM009q5']
results = ['clean', 'clean', 'clean', 'clean', 'clean']


# The size parameter defines how many samples will be in the training set each
def classify(filename, size, url, result):

    trainingSet = make_training_set.create_set(filename, size)

    trainer = Trainer(tokenizer.Tokenizer(stop_words = [], signs_to_remove = [""]))

    for sample in trainingSet:
        trainer.train(sample['url'], sample['result'])

    classifier = Classifier(trainer.data, tokenizer.Tokenizer(stop_words = [], signs_to_remove = [""]))

    print "Expected: " + result
    print classifier.classify(url)


def test(size, url, result):
    classify('lotsodata.txt', size, url, result)


def test_lots(size, urls, results):

    for i in xrange(len(urls)):
        classify('lotsodata.txt', size, urls[i], results[i])
