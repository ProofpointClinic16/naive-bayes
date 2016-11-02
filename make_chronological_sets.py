import re
from pprint import pprint

# Size is the total number of samples in one data set
def create_sets(filename, size=10):

    training = []
    testing = []

    training_left = testing_left = size

    with open(filename) as f:
        for line in f:

            datum = {}

            result = re.search(r"result': u'(.+?)'}", line).group(1)
            url = re.search(r"url': u'(.+?)', ", line).group(1)

            datum['url'] = url
            datum['result'] = result

            if result != 'malicious' and result != 'clean':
                continue

            if training_left > 0:
                training_left -= 1
                training += [datum]
            elif testing_left > 0:
                testing_left -= 1
                testing += [datum]

            if training_left == 0 and testing_left == 0:
                break

    # pprint(training)
    # pprint(testing)

    return (training, testing)
