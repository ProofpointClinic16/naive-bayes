import parser
from pprint import pprint


def tokenize(filename):

    tokens_slash = []
    tokens_period = []
    tokens_sp = []

    #Calls parse function from parser file
    #and stores parsed data in 'data' 
    data = parser.parse(filename)

    #This block of code tokenizes the URLs
    #that exist in 'data' by '/' and '.' and
    #creates a list to print out later
    for i in xrange(len(data)):
        slash_split = data[i]['url'].split('/')
        period_split = data[i]['url'].split('.')

        tokens_slash += [slash_split]
        tokens_period += [period_split]

    #Separate block of code that takes the list
    #holding the URLs tokenized by slashes and
    #then tokenizes them further by periods.
    for i in xrange(len(tokens_slash)):
        for j in xrange(len(tokens_slash[i])):
            sp_split = tokens_slash[i][j].split('.')

            #Issue here when creating a LoL
            #URLs are not separated into lists
            tokens_sp += sp_split

    print "SLASHES"
    pprint(tokens_slash)
    print "--------------------------------"
    print "PERIODS"
    pprint(tokens_period)
    print "--------------------------------"
    print "SLASHES AND PERIODS"
    pprint(tokens_sp)


def test():
    tokenize('sample.txt')
