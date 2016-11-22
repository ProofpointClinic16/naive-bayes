
import json

import sys
import pymongo
from bson.objectid import ObjectId
from math import exp
import datetime


P = 100000127
#P = 1000081
PREF = 3500
delta = .002
max_weight = 5
threshold=.6

dun = [0] * P
w = [0] * P

def clean(url):
    url.replace('https://','')
    url.replace('http://','')
    url.replace('ftp://','')
    url.replace('www.','',1)
    return url


def predict_url( url):
    prob = 0
    url = ''.join(sorted(url))
    if len(url) > 4:
        score = 0
        #b = (ord(url[0]) << 16) | (ord(url[1]) << 8) | ord(url[2])
        b = url[0:3]
        for c in url[3:]:
            if c == '/' or c == '.':
                continue
            #b = ((b << 8) & 0xFFFFFFFF) | ord(c)
            b += c
            #h = b % P
            h = abs(hash(b)) % P
            if dun[h]:
                continue
#            dun[h] = 1
            dun[h] = 0 # added
            score += w[h]

        prob = 1 / (1 + exp(-score))
#    if len(url) > 4:
#        # pass 1
#        b = (ord(url[0]) << 16) | (ord(url[1]) << 8) | ord(url[2])
#        for c in url[3:]:
#            if c == '/' or c == '.':
#                continue
#            #    gram=gram[1:]+c
#            b = ((b << 8) & 0xFFFFFFFF) | ord(c)
#            h = b % P
#            if dun[h]:
#                dun[h] = 0

    return prob

def train_url( url,is_malware,learn):
    prob = 0
    url = ''.join(sorted(url))
    if len(url) > 4:
        score = 0
        #b = (ord(url[0]) << 16) | (ord(url[1]) << 8) | ord(url[2])
        b = url[0:3]
        for c in url[3:]:
            if c == '/' or c == '.':
                continue
            #b = ((b << 8) & 0xFFFFFFFF) | ord(c)
            b += c
            h = abs(hash(b)) % P
            #h = b % P
            if dun[h]:
                continue
            dun[h] = 1
            score += w[h]

        prob = 1 / (1 + exp(-score))

    if len(url) > 4:
        # pass 1
        #b = (ord(url[0]) << 16) | (ord(url[1]) << 8) | ord(url[2])
        b = sorted(url[0:3])
        for c in url[3:]:
            if c == '/' or c == '.':
                continue
            #    gram=gram[1:]+c
            #b = ((b << 8) & 0xFFFFFFFF) | ord(c)
            #h = b % P
            b += c
            b = ''.join(sorted(b))
            h = abs(hash(b)) % P
            if dun[h]:
                dun[h] = 0
                if abs(w[h]) < max_weight:
                    w[h] += (is_malware - prob) * delta * learn
    return prob

for filename in sys.argv[1:]:
    with open(filename) as f:
        for line in iter(f):
            data = eval(line)
            classify_malware='class_clean'
            if 'url' in data:
                url = clean(data['url'])
                prob=predict_url(url)
                if prob>threshold:
                    classify_malware='class_malware'

                is_malware = -1
                if 'results' in data:
                    if 'result' in data['results']:
                        if data['results']['result'] == 'malicious':
                            is_malware=1
                            learn=20
                        elif data['results']['result'] == 'clean':
                            is_malware=0
                            learn=1
                    if is_malware>=0:
                        prob2=train_url(url,is_malware,learn)
                        prob3=predict_url(url)
                        j={'url':data['url'],'sandbox_results':data['results']['result'],'classification':classify_malware,'prob':prob,'trained_prob':prob3}
                        line = json.dumps(j)
                        print line
        f.close()
