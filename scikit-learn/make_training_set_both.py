import re

# Size is the total number of samples in one data set
def create_set(filename, size=10):

    training = []

    training_left = size

    with open(filename) as f:
        for line in f:

            datum = {}

            result = re.search(r"result': u'(.+?)'}", line).group(1)
            url = re.search(r"url': u'(.+?)', ", line).group(1)
            ip = re.search(r"ip': u'(.+?)', ", line).group(1)

            datum['sample'] = url + ip
            datum['result'] = result

            if result != 'malicious' and result != 'clean':
                continue

            if training_left > 0:
                training_left -= 1
                training += [datum]

            if training_left == 0:
                break

    return training
