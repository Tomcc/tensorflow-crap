import codecs
import tensorflow as tf
import os
import operator
import math

def makeDictionary(path):
    histogram = {}

    with codecs.open(path, "r", "'utf-8") as f:
        while True:
            c = f.read(1)
            if not c:
                break
            if not c in histogram:
                histogram[c] = 1
            else:
                histogram[c] += 1

    histogram_sorted = sorted(histogram.items(), key=operator.itemgetter(1), reverse=True)

    depth = min(len(histogram_sorted), 60000)

    print(depth)

    char_lookup = {}

    for i in range(depth):
        char_lookup[histogram_sorted[i][0]] = i

    return char_lookup

def readBook(dictionary, path):
    book_indices = []
    with codecs.open(path, "r", "'utf-8") as f:
        while True:
            c = f.read(1)
            if not c:
                break
            if c in dictionary:
                book_indices.append(dictionary[c])

    return book_indices

targetPath = "data/target.txt"

dictionary = makeDictionary(targetPath)
book_indices = readBook(dictionary, targetPath)

one_hot = tf.one_hot(book_indices, len(dictionary))

print(len(book_indices))

sess = tf.Session()

# one-layer GRU network: just take one character and predict one character.

unrolls = 25
rnn_units = 100
batch_size = 100
character_size = len(dictionary)

inputs = tf.placeholder(tf.float32, [batch_size, unrolls, character_size])
future_input = tf.placeholder(tf.float32, [batch_size, character_size])

#create the lstm cells (and unroll t hem)
rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_units)
outputs, final_state = tf.nn.static_rnn(
    rnn_cell,
    inputs,
    sequence_length = tf.fill([batch_size], unrolls),
    dtype=tf.float32)

# TODO which output for the loss???

# then seed the network with a sequence

# and let it go generate a bunch more stuff

print(sess.run(outputs))