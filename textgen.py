import codecs
import tensorflow as tf
import os
import operator
import math

# read data into a giant array

histogram = {}

with codecs.open("data/target.txt", "r", "'utf-8") as f:
    while True:
        c = f.read(1)
        if not c:
            break
        if not c in histogram:
        	histogram[c] = 1
        else:
        	histogram[c] += 1

histogram_sorted = sorted(histogram.items(), key=operator.itemgetter(1), reverse=True)

depth = min(len(histogram_sorted), 30)

print(depth)

char_array = []
char_lookup = {}

for i in range(depth):
    char_array.append(i)
    char_lookup[histogram_sorted[i][0]] = i

one_hot = tf.one_hot(char_array, depth)

sess = tf.Session()

print(char_lookup)