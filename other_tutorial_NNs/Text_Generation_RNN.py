import os
import random
from zipfile import ZipFile

import numpy as np
import requests
import tensorflow as tf
from bs4 import BeautifulSoup

BASE_PATH = 'http://export.arxiv.org/api/query'

CATEGORIES = ["Machine Learning", "Neural and Evolutionary Computing", "Optimization"]

KEYWORDS = ['neural', 'network', 'deep']


def build_url(amount, offset):
    categories = ' OR '.join('cat:' + x for x in CATEGORIES)
    keywords = ' OR '.join('all:' + x for x in KEYWORDS)

    url = BASE_PATH
    url += '?search_query=(({}) AND ({}))'.format(categories, keywords)
    url += '&max_results={}&offset={}'.format(amount, offset)
    return url


def get_count():
    url = build_url(0, 0)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')

    count = int(soup.find('opensearch:totalresults').string)
    print(count, 'papers found')
    return count


num_papers = get_count()

PAGE_SIZE = 100


def fetch_page(amount, offset):
    url = build_url(amount, offset)

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')

    for entry in soup.findAll('entry'):
        text = entry.find('summary').text
        text = text.strip().replace('\n', ' ')
        yield text


def fetch_all():
    for offset in range(0, num_papers, PAGE_SIZE):
        print('Fetch papers {}/{}'.format(offset + PAGE_SIZE, num_papers))

        for page in fetch_page(PAGE_SIZE, offset):
            yield page


ZIP_FILENAME = 'arxiv_abstracts.zip'


def download_data():
    if not os.path.isfile(ZIP_FILENAME):
        with ZipFile(ZIP_FILENAME, 'w') as zipFile:
            for abstract in fetch_all():
                zipFile.write(abstract + '\n')

    with ZipFile(ZIP_FILENAME, 'r') as zipFile:
        data1 = zipFile.read('arxiv_abstracts.txt')

    return data1


data = download_data()

# End data downloading
# Start data preparation

MAX_SEQUENCE_LENGTH = 50
BATCH_SIZE = 100

VOCABULARY = " $%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ\\^_abcdefghijklmnopqrstuvwxyz{|}"

lookup = {x: i for i, x in enumerate(VOCABULARY)}

sample_lookup = random.sample(lookup.items(), 10)

SEQUENCE_LENGTH = 2


def one_hot(batch1, sequence_length1 = MAX_SEQUENCE_LENGTH):
    one_hot_batch = np.zeros((len(batch1), sequence_length1, len(VOCABULARY)))

    # iterate through every line of text in a batch
    for index, line in enumerate(batch1):

        line = [x for x in line if x in lookup]
        assert 2 <= len(line) <= MAX_SEQUENCE_LENGTH

        # iterate through every character in a line
        for offset, character in enumerate(line):

            # code is the index of the character in the vocabulary
            code = lookup[character]

            one_hot_batch[index, offset, code] = 1
    return one_hot_batch


def next_batch():
    windows = []
    for line in data:
        for i in range(0, len(line) - MAX_SEQUENCE_LENGTH + 1, MAX_SEQUENCE_LENGTH // 2):
            windows.append(line[i: i + MAX_SEQUENCE_LENGTH])

    # all text at this point is in the form of windows of MAX_SEQUENCE_LENGTH characters
    assert all(len(x) == len(windows[0]) for x in windows)

    while True:
        # we do not want our NN to learn from position by accident
        random.shuffle(windows)
        for j in range(0, len(windows), BATCH_SIZE):
            batch1 = windows[i:i + BATCH_SIZE]
            yield one_hot(batch1)


# test_batch = None
# for batch in next_batch():
#    test_batch = batch
#    print(batch.shape)
#    break;

# end data preparation
# start NN

tf.compat.v1.reset_default_graph()
sequence = tf.compat.v1.placeholder(tf.float32, [1, SEQUENCE_LENGTH, len(VOCABULARY)])

X = tf.slice(sequence, (0, 0, 0), (-1, SEQUENCE_LENGTH - 1, -1), name = 'LastLetterRemoval')
y = tf.slice(sequence, (0, 1, 0), (-1, -1, -1), name = 'FirstLetterRemoval')


def get_mask(target):
    mask1 = tf.reduce_max(tf.abs(target), reduction_indices = 2)
    return mask1


def get_sequence_length(target):
    mask1 = get_mask(target)
    sequence_length1 = tf.reduce_sum(mask1, reduction_indices = 1)
    return sequence_length1


num_neurons = 200
cell_layers = 2
num_steps = MAX_SEQUENCE_LENGTH - 1
num_classes = len(VOCABULARY)

sequence_length = get_sequence_length(y)


def build_RNN(data1, numSteps = num_steps, sequenceLength = sequence_length, initial = None):
    multi_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(num_neurons) for _ in range(cell_layers)])

    output, state = tf.compat.v1.nn.dynamic_rnn(inputs = data1, cell = multi_cell, dtype = tf.float32,
                                                # if initial_stage param would be available, we could generate text with initial_state = initial
                                                sequence_length = sequenceLength)
    # shared softmax layer across all RNN cells
    weight = tf.Variable(tf.random.truncated_normal([num_neurons, num_classes], stddev = 0.01), name = "weight")
    bias = tf.Variable(tf.constant(0.1, shape = [num_classes]), name = "bias")

    flattened_output = tf.reshape(output, [-1, num_neurons])

    prediction1 = tf.nn.softmax(tf.matmul(flattened_output, weight) + bias, name = "softmaxActivation")
    prediction1 = tf.reshape(prediction1, [-1, numSteps, num_classes], name = "reshapedPrediction")

    return prediction1, state


state1 = tf.compat.v1.placeholder(tf.float32, [1, num_neurons])
state2 = tf.compat.v1.placeholder(tf.float32, [1, num_neurons])

prediction, output = build_RNN(X, num_steps = SEQUENCE_LENGTH - 1, sequenceLength = sequence_length, initial = (state1, state2))
mask = get_mask(y)
prediction = tf.clip_by_value(prediction, 1e-10, 1.0, name = 'predictionBoundaries')

length = tf.reduce_sum(sequence_length, 0)

with tf.name_scope("CrossEntropy"):
    cross_entropy = y * tf.math.log(prediction)
    cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices = 2)
    cross_entropy *= mask
    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices = 1) / length
    cross_entropy = tf.reduce_mean(cross_entropy)

with tf.name_scope("Perplexity"):
    logprob = tf.multiply(prediction, y)
    logprob = tf.reduce_max(logprob, reduction_indices = 2)
    logprob = tf.math.log(tf.clip_by_value(logprob, 1e-10, 1.0)) / tf.math.log(2.0)
    logprob *= mask
    logprob = tf.reduce_sum(logprob, reduction_indices = 1) / length
    logprob = tf.reduce_mean(logprob)

with tf.name_scope("optimizer"):
    optimizer = tf.compat.v1.train.RMSPropOptimizer(0.002)
    gradient = optimizer.compute_gradients(cross_entropy)
    optimize = optimizer.apply_gradients(gradient)

num_epochs = 100
epoch_size = 100

logprob_evals = []
checkpoint_dir = './sample_checkpoint_output'
sess = tf.compat.v1.InteractiveSession()
saver = tf.compat.v1.train.Saver()
checkpoint = tf.compat.v1.train.get_checkpoint_state(checkpoint_dir)
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(num_epochs):
    for _ in range(epoch_size):

        batch = next(next_batch())

        logprob_eval, _ = sess.run((logprob, optimize), {sequence: batch})
        logprob_evals.append(logprob_eval)
    saver.save(sess, os.path.join(checkpoint_dir, "char_pred"), epoch)
    perplexity = 2 ** -(sum(logprob_evals[-epoch_size:]) / epoch_size)

    print('Epoch {:2d} perplexity {:5.4f}'.format(epoch, perplexity))
