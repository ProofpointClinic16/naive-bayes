import json


def parse(filename):
    data = []

    with open(filename) as f:
        for line in f:
            line = line.replace("u'","\"")
            line = line.replace("u\"", "\"")
            line = line.replace("':", "\":")
            line = line.replace("',", "\",")
            line = line.replace("']", "\"]")
            line = line.replace("'}", "\"}")
            line = line.replace(" date", " \"date")
            line = line.replace("),", ")\",")
            line = line.replace(")',", ")\",")
            line = line.replace("None", "\"None\"")
            line = line.replace("Object", "\"Object")
            line = line.replace("True", "\"True\"")
            line = line.replace("False", "\"False\"")
            line = line.replace(" ", "")

            try:
                datum = json.loads(line)
            except ValueError:
                print "Failed at:"
                print line
                continue

            data += [datum]

    return data
