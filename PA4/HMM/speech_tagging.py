import numpy as np
import time
import random
from hmm import HMM


def accuracy(predict_tagging, true_tagging):
    if len(predict_tagging) != len(true_tagging):
        return 0, 0, 0
    cnt = 0
    for i in range(len(predict_tagging)):
        if predict_tagging[i] == true_tagging[i]:
            cnt += 1
    total_correct = cnt
    total_words = len(predict_tagging)
    if total_words == 0:
        return 0, 0, 0
    return total_correct, total_words, total_correct * 1.0 / total_words


class Dataset:
    def __init__(self, tagfile, datafile, train_test_split=0.8, seed=int(time.time())):
        tags = self.read_tags(tagfile)
        data = self.read_data(datafile)
        self.tags = tags
        lines = []
        for l in data:
            new_line = self.Line(l)
            if new_line.length > 0:
                lines.append(new_line)
        if seed is not None:
            random.seed(seed)
        random.shuffle(lines)
        train_size = int(train_test_split * len(data))
        self.train_data = lines[:train_size]
        self.test_data = lines[train_size:]
        return

    def read_data(self, filename):
        """Read tagged sentence data"""
        with open(filename, 'r') as f:
            sentence_lines = f.read().split("\n\n")
        return sentence_lines

    def read_tags(self, filename):
        """Read a list of word tag classes"""
        with open(filename, 'r') as f:
            tags = f.read().split("\n")
        return tags

    class Line:
        def __init__(self, line):
            words = line.split("\n")
            self.id = words[0]
            self.words = []
            self.tags = []

            for idx in range(1, len(words)):
                pair = words[idx].split("\t")
                self.words.append(pair[0])
                self.tags.append(pair[1])
            self.length = len(self.words)
            return

        def show(self):
            print(self.id)
            print(self.length)
            print(self.words)
            print(self.tags)
            return


# TODO:
def model_training(train_data, tags):
    """
    Train HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - tags: (1*num_tags) a list of POS tags

    Returns:
    - model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
    """
    model = None
    ###################################################
    # tags are the states (X) that are not observed, words are the obs sequence (Z)
    train_size = len(train_data)

    state_dict = make_dict(tags)
    S = len(tags)

    # get obs_dict using training data
    word_bag = []
    for i in range(train_size):
        word_bag += train_data[i].words
    word_bag = list(set(word_bag))
    obs_dict = make_dict(word_bag)
    num_obs_symbols = len(word_bag)

    # initialize all parameter
    pi = np.zeros(S)
    A = np.zeros(shape=(S, S))  # (S, S)
    B = np.zeros(shape=(S, num_obs_symbols))  # (S, num_obs_symbols)

    # compute parameters using MLE
    for i in range(train_size):
        sentence = train_data[i]
        x1 = sentence.tags[0]
        pi[state_dict[x1]] += 1
        L = sentence.length
        for t in range(L - 1):
            x_now = sentence.tags[t]
            z_now = sentence.words[t]
            x_next = sentence.tags[t + 1]
            A[state_dict[x_now], state_dict[x_next]] += 1
            B[state_dict[x_now], obs_dict[z_now]] += 1
        x_now = sentence.tags[L - 1]
        z_now = sentence.words[L - 1]
        B[state_dict[x_now], obs_dict[z_now]] += 1
    # normalized
    A = (A.T / np.sum(A, axis=1)).T
    B = (B.T / np.sum(B, axis=1)).T
    pi = pi / np.sum(pi)

    # build the model
    model = HMM(pi, A, B, obs_dict, state_dict)
    ###################################################
    return model


# TODO:
def speech_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - model: an object of HMM class

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ###################################################
    num_sentence = len(test_data)
    for i in range(len(test_data)):
        sentence = test_data[i]
        obs_seq = sentence.words

        # check whether exist new observations
        new_obs = []
        for word in obs_seq:
            if word not in model.obs_dict:
                new_obs.append(word)
        model.add_obs(new_obs)

        # predict using Viterbi algorithm
        tag = model.viterbi(obs_seq)
        tagging.append(tag)

    ###################################################
    return tagging


def make_dict(text_list):
    size = len(text_list)
    text_dict = dict()
    for i in range(size):
        text_dict[text_list[i]] = int(i)
    return text_dict
