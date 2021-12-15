import nltk
import string
import re
import random
import csv
import codecs
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import collections
import textstat

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from io import StringIO
from csv import reader
from collections import Counter

path = "training_set_rel3.tsv"
path3 = "word_corpus.csv"
csv_table = pd.read_table(path,sep='\t', encoding="ISO-8859-1")
word_list = words.words()
word_set = set(word_list)
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')
wordnet_lemmatizer = WordNetLemmatizer()

corpus_set = []

with open(path3, newline='', encoding="ISO-8859-1") as file:
    contents = csv.reader(file, delimiter=' ')
    for row in contents:
        lemma = wordnet_lemmatizer.lemmatize(row[0].lower())
        corpus_set.append(lemma)


def writer_label(points):
    if points < 8:
        return 0
    else:
        return 1


misspelled = []


def spelling_errors(words):
    spelling_error = 0
    misspell_arr = []
    for word in words:
        if word.isupper() is False:
            word = word.lower()
            if word not in stop_words:
                word = wordnet_lemmatizer.lemmatize(word)
                if word not in word_set:
                    spelling_error += 1
                    misspell_arr.append(word.lower())
    misspelled.append(misspell_arr)
    return spelling_error


sophist_arr = []


def sophisticated(words):
    sophisticated_words = 0
    sophist_words = []
    for word in words:
        if word.isupper() is False:
            word = word.lower()
            if word not in stop_words:
                word = wordnet_lemmatizer.lemmatize(word)
                if word not in corpus_set:
                    sophisticated_words += 1
                    sophist_words.append(word.lower())
    sophist_arr.append(sophist_words)
    return sophisticated_words


keep_col = ['essay_id', 'essay_set', 'essay', 'rater1_domain1', 'rater2_domain1']
mod_data = csv_table[keep_col]
mod_data = mod_data.dropna(subset=['essay_id', 'essay_set', 'essay', 'rater1_domain1', 'rater2_domain1'], how='any')
mod_data = mod_data[mod_data.essay_set == 1]
rater1_domain_feat = np.ndarray.tolist(mod_data['rater1_domain1'].values)
rater2_domain_feat = np.ndarray.tolist(mod_data['rater2_domain1'].values)
mod_data = mod_data.assign(total_score=(mod_data['rater1_domain1'] + mod_data['rater2_domain1']))
mod_data = mod_data.assign(label = mod_data['total_score'].apply(writer_label))
essay_feats = np.ndarray.tolist(mod_data['essay'].values)
labels = np.ndarray.tolist(mod_data['label'].values)

postag_array = []
errors = []
spell_check = []
num_sophisticated = []


for element in mod_data['essay']:
    text = word_tokenize(element)
    words = [word for word in text if word.isalpha()]
    spell_check.append(spelling_errors(words))
    num_sophisticated.append(sophisticated(words))

counter = []

for soph_words in sophist_arr:
    count = 0
    for word in soph_words:
        if word in misspelled:
            count += 1
    counter.append(count)

sophisticated_revised = []

for i in range(len(counter)):
    sophisticated_num = num_sophisticated[i] - counter[i]
    sophisticated_revised.append(sophisticated_num)

essay_length = []
sentence_length = []
avg_sentences = []
lex_div = []


for element in essay_feats:
    num_words = len(element.split())
    essay_length.append(num_words)
    unique_words = Counter(element.split())
    lex_div.append(len(unique_words))
    num_sentences = textstat.sentence_count(element)
    sentence_length.append(num_sentences)
    avg_sent_length = num_words/num_sentences
    avg_sentences.append(avg_sent_length)

essay_numwords = np.array(essay_length)
essay_numsent = np.array(sentence_length)
essay_avgsentlen = np.array(avg_sentences)
essay_lexdiv = np.array(lex_div)
essay_spelling = np.array(spell_check)
essay_sophisticated = np.array(sophisticated_revised)

mod_data = mod_data.assign(essay_length=essay_numwords)
mod_data = mod_data.assign(num_sentences=essay_numsent)
mod_data = mod_data.assign(avg_sent_length=essay_avgsentlen)

mod_data = mod_data.assign(lexical_diversity=essay_lexdiv)
mod_data = mod_data.assign(spelling_error=essay_spelling)
mod_data = mod_data.assign(sophisticated_words=essay_sophisticated)
mod_data = mod_data.assign(label=mod_data['total_score'].apply(writer_label))
print(mod_data.head())


X_feat = mod_data.iloc[:, 7:].to_numpy()
Y_label = mod_data['label'].to_numpy()
Y_label = Y_label.reshape(Y_label.shape[0], 1)
X_train, X_test, y_train, y_test = train_test_split(X_feat, Y_label, test_size=0.3, random_state=30)
logistic_regression = LogisticRegression(solver='newton-cg').fit(X_train, y_train)
print(classification_report(y_test, logistic_regression.predict(X_test), digits=3))