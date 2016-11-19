from nltk import ngrams
import re

class Tokenizer(object):
    def __init__(self, stop_words = [], signs_to_remove = ["?!#%&"]):
        self.stop_words = stop_words
        self.signs_to_remove = signs_to_remove

    def tokenize(self,text):
        fourgrams = ngrams(text.split(), 4)
        return text.lower().split('/') + text.lower().split('.')

    def remove_stop_words(self,token):
        if token in self.stop_words:
            return "stop_word"
        else:
            return token

    def remove_punctuation(self,token):
        return re.sub(str(self.signs_to_remove),"",token)
