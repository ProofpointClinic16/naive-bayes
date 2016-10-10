import parser
from pprint import pprint


def tokenize(filename):

    tokens_slash = []
    tokens_period = []

    data = parser.parse(filename)

    for i in xrange(len(data)):
        slash_split = data[i]['url'].split('/')
        period_split = data[i]['url'].split('.')

        tokens_slash += [slash_split]
        tokens_period += [period_split]

    print "SLASHES"
    pprint(tokens_slash)
    print "--------------------------------"
    print "PERIODS"
    pprint(tokens_period)


def test():
    tokenize('sample.txt')
