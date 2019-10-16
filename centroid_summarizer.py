#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 08:37:22 2019

@author: eduardo

From : https://github.com/gaetangate/text-summarizer/tree/master/text_summarizer

"""
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from scipy.spatial.distance import cosine
import re

import HiddenSentences as hs

hidden_sentences = 1

#def find_hidden_sentences(text):
#    pattern = "[a-z]\.[A-Z]"    
#    matches = [m.start(0) for m in re.finditer(pattern, text)]
    
#    for i, index in enumerate(matches):
#        index = index + 2 + i
#        text = text[:index] + " " + text[index:]

#    return text


q_context = "Please note, there are 4 sizes (10, 10.2, 14, and 16 inches) of the same case type; I am writing a review of the 16-inch case.\
This is a well-made but relatively inexpensive bag for a laptop PC. It is so light that I can barely feel its weight, but it still has sufficient padding to \
protect its contents against bumps during everyday use.This is NOT the kind of case with zippers that can be pulled further to the left and right sides when \
you open the bag. In other words, the opening for you to put in and take out things is very small. Does this matter? Yes, if you have a laptop \
PC that measures diagonally 15.6 to 16 inches. The fit is snug, and because of the narrow opening, putting in and taking out a computer is not that smooth and easy. \
Although the zippers are plastic, there is a chance of them scraping against the computer's edges. I would not recommend this case if you have concerns about this aspect. \
If your computer is a netbook, which is about 10 inches diagonal, then this case will be fine.There is an USB thumb drive pocket in the main compartment, \
where the computer is stored, but I feel it is not a good location, because it is likely to chafe against the laptop computer. There are two side pockets. \
I use the larger one to store AC adapter, cord, mouse pad, and mouse.I deem this bag more suitable for carrying smaller laptops. It is not suitable for\
 15.6 to 16-inch laptops because of the tight fit.Update (2-4-2011):Case Logic has another 16-inch case that zipper opens three sides instead of just one. \
 It is called Case Logic VNCi-116 Value 16-Inch Laptop Briefcase. It will easily fit any 16-inch notebook computer. Even my 17.3-inch HP DV7 can be accommodated \
 in the case. You can read my review on its web page on Amazon.com."
 
print q_context

if hidden_sentences:
    q_context = hs.find_hidden_sentences(q_context)
    q_context_sents = sent_tokenize(q_context)
    #print q_context_sents
    #print len(q_context_sents)
else: 
    q_context_sents = sent_tokenize(q_context)
    #print q_context_sents
    #print len(q_context_sents)

print

vectorizer = CountVectorizer()
sent_word_matrix = vectorizer.fit_transform(q_context_sents)
#print sent_word_matrix.toarray()

transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False)
tfidf = transformer.fit_transform(sent_word_matrix)
tfidf = tfidf.toarray()
tfidf.shape

centroid_vector = tfidf.sum(0)


topic_threshold = 0.3
centroid_vector = np.divide(centroid_vector, centroid_vector.max())
for i in range(centroid_vector.shape[0]):
            if centroid_vector[i] <= topic_threshold:
                centroid_vector[i] = 0

#print centroid_vector.shape


def similarity(v1, v2):
    score = 0.0
    if np.count_nonzero(v1) != 0 and np.count_nonzero(v2) != 0:
        score = ((1 - cosine(v1, v2)) + 1) / 2
    return score


sentences_scores = []
for i in range(tfidf.shape[0]):
    score = similarity(tfidf[i, :], centroid_vector)
    sentences_scores.append((i, q_context_sents[i], score, tfidf[i, :]))


sentence_scores_sort = sorted(sentences_scores, key=lambda el: el[2], reverse=True)


limit_type = 'word'
limit = 100
sim_threshold = 0.95
count = 0
sentences_summary = []
for s in sentence_scores_sort:
    if count > limit:
        break
    include_flag = True
    for ps in sentences_summary:
        sim = similarity(s[3], ps[3])
        # print(s[0], ps[0], sim)
        if sim > sim_threshold:
            include_flag = False
    if include_flag:
        # print(s[0], s[1])
        sentences_summary.append(s)
        if limit_type == 'word':
            count += len(s[1].split())
        else:
            count += len(s[1])

summary = "\n".join([s[1] for s in sentences_summary])

print "--=== SUMMARY ===---"
print summary
