import random
import zipfile
import itertools
import os

import numpy as np
import collections
import math

import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
import tensorflow as tf

def extract_input_data(matrix):
    """This function extract the data from the adjacency matrix of a bipartite network
    where rows are contexts and colums are words"""
    result = [list(np.nonzero(matrix[i])[0]) for i in range(0,matrix.shape[0])]
    return result

def load_sample_matrix(name = "/sample_bipartite_matrix.npy"):
    """
    Default name refers to the prodided matrix, specify a different name if you want to import a different matrix
    The new matrix should be saved in a *.npy format and the format should be explicitly written when loading a different matrix.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    matrix = np.load(path+name)
    return matrix

def build_dataset(words, vocabulary_size):
    """This function prepares the data for the training"""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def data_preprocessing(input_data, vocabulary_size):
    """ This function accepts as input a 2-level nested list where each sublist is a context formed by words of the vocabulary"""

    flatcodes = [item for sublist in input_data for item in sublist]
    lengths = [len(sub) for sub in input_data]
    accumulated = [0]
    acc = 0
    for l in lengths:
        acc += l - 1
        accumulated.extend([acc])

    data, count, dictionary, reverse_dictionary = build_dataset(flatcodes, vocabulary_size)
    return data, reverse_dictionary, accumulated

def generate_batch(data, accumulated, examp_num, batch_size):
    batch=np.array([], dtype=np.int32)
    labels=np.array([], dtype=np.int32)

    while (len(batch)<=batch_size):
        y=random.randint(0, len(accumulated)-2)
        sel=data[accumulated[y]:accumulated[y+1]]
        permsel=np.random.permutation(sel)
        inp=permsel[0]
        labels_t=permsel[1:examp_num]
        batch_t = np.array([inp for _ in range(len(labels_t))])
        batch = np.concatenate((batch, batch_t), axis=0)
        labels = np.concatenate((labels, labels_t), axis=0)
    batch = batch[:batch_size]
    labels = labels[:batch_size]
    labels = labels.reshape((len(labels), 1))
    return batch, labels

###################################################################################################################

def create_the_embeddings(data, reverse_dictionary, accumulated, vocabulary_size, batch_size = 128, embedding_size = 16, examp_num = 8, num_sampled = 64, num_steps = 50000):
    """
    batch_size: The size of the batch
    embedding_size:  Dimension of the embedding vector.
    examp_num: How many times to reuse an input to generate a label.
    num_sampled: Number of negative examples to sample.
    """
    graph = tf.Graph()

    with graph.as_default():

        # Input data.
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        # Ops and variables pinned to the CPU, they can be pinned to the GPU if CUSA is available
        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_labels,
                                             inputs=embed,num_sampled=num_sampled, num_classes=vocabulary_size))

        # Construct the SGD optimizer using a learning rate of 1.0.
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm

        # Step 5: Begin training.

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        tf.global_variables_initializer().run()
        print("Initialized")

        average_loss = 0
        for step in range(0,num_steps):
            batch_inputs, batch_labels = generate_batch(data, accumulated, examp_num, batch_size)
            feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val
            if step%1000 == 0:
                print("step: "+str(step)+", loss: "+str(average_loss/1000.))
                average_loss = 0
        final_embeddings = normalized_embeddings.eval()
        codestoexp = [reverse_dictionary[i] for i in range(vocabulary_size)]
    return final_embeddings, codestoexp

def save_embeddings(ListOfEmbeddings, List_of_Words, name_e = "/EmbeddingTensor", name_w = "/List_of_Words"):
    """Specify a different fine_name to avoid overwritig the existing embedding files"""
    path = os.path.dirname(os.path.abspath(__file__))

    np.save(path + name_e, np.stack(ListOfEmbeddings))
    np.save(path + name_w, List_of_Words)

def load_embeddings(name_e = "/EmbeddingTensor.npy", name_w = "/List_of_Words.npy"):
    """To load different embeddings, specify the name with which they are saved"""

    path = os.path.dirname(os.path.abspath(__file__))

    ListOfEmbeddings = np.load(path + name_e)
    List_of_Words = np.load(path + name_w)

    return ListOfEmbeddings, list(List_of_Words)

def save_context_similarity(context_similarity_0, name = "/ContextSimilarity"):
    path = os.path.dirname(os.path.abspath(__file__))

    np.save(path + name, context_similarity_0)

def load_context_similarity(name = "/ContextSimilarity.npy"):
    path = os.path.dirname(os.path.abspath(__file__))

    return np.load(path + name)