
import json

import sys


total=0
malware_count=0
clean_count=0
false_negative = 0
false_positive = 0
true_negative = 0
true_positive = 0
sandbox_count=0

filename = sys.argv[1]
with open(filename) as f:
    for line in iter(f):
        data = json.loads(line)
        #print data
        if data['sandbox_results'] == 'malicious':
            malware_count+=1
            if data['classification'] == 'class_malware':
                true_positive+=1
            if data['classification'] == 'class_clean':
                false_negative+=1
        elif  data['sandbox_results'] == 'clean':
            clean_count+=1
            if data['classification'] == 'class_malware':
                false_positive+=1
            if data['classification'] == 'class_clean':
                true_negative+=1
        if data['classification'] == 'class_malware':
            sandbox_count+=1

total_count=malware_count+clean_count
print("Total: {0}".format(total_count))
print("Malware: {0}".format(malware_count))
print("Clean: {0}".format(clean_count))
print("False Positive: {0} ({1}% of all clean samples)".format(false_positive, 100 * false_positive/clean_count))
print("False Negative: {0} ({1}% of all malicious samples)".format(false_negative, 100 * false_negative/malware_count))
print("True Positive: {0} ({1}% of all malicious samples)".format(true_positive, 100 * true_positive/malware_count))
print("True Negative: {0} ({1}% of all clean samples)".format(true_negative, 100 * true_negative / clean_count))
print("Sandbox submission: {0} ({1}% malicious)".format(sandbox_count,100*true_positive/sandbox_count))
