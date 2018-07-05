"""
Author: Tara Rengarajan
Read sample text,run an RNN and generate text
"""

#imports
import numpy as np
import argparse
import copy as cp
import sys
import math
from collections import OrderedDict


def read_file_return_text(_input_file):
    """
    :param _input_file:
    :return:  chars in file
    """

    with open(_input_file, 'r') as f:
        text = f.read()

    chars = list(set([char for char in text]))
    chars.sort()
    return text, chars


def sigmoid(x):
    return 1 / (1 + math.e ** -x)

def relu(x):
    return np.maximum(x, 0)


def initialize(_vocab_size, _hidden_layer_size):
    """
    :param _vocab_size:
    :param _hidden_layer_size:
    :return:
    """
    np.random.seed(10)
    scale_factor = .01
    W_xh = scale_factor*np.random.randn(_hidden_layer_size, _vocab_size)
    W_hh = scale_factor*np.random.randn(_hidden_layer_size, _hidden_layer_size)
        # np.identity(_hidden_layer_size)
    W_hy = scale_factor*np.random.randn(_vocab_size, _hidden_layer_size)

    b = np.zeros(_hidden_layer_size,)
    b_prime = np.zeros(_vocab_size,)

    return W_xh, W_hh, W_hy, b, b_prime


def forward_pass(_input, _hidden_prev, _W_xh, _W_hh, _W_hy, _b, _b_prime):

    """

    :param _input:
    :param _hidden:
    :param _W_xh:
    :param _W_hh:
    :param _W_hy:
    :param _b:
    :param _b_prime:
    :return:
    """

    hidden = np.tanh(np.matmul(_W_xh, _input) + np.matmul(_W_hh, _hidden_prev) + _b)
    # print('Forward pass: hidden prev {0}, hidden {1}'.format(_hidden, hidden))
    y = np.matmul(_W_hy, hidden) + _b_prime
    probs = np.exp(y)/np.sum(np.exp(y))

    return hidden, y, probs


def sample_from_model(_vocab_size, _index_char_dict, _text_size, **kwargs):


    first_char_index  = np.random.randint(0, _vocab_size)
    first_char = _index_char_dict[first_char_index]
    print('First char: {0}'.format(first_char))
    x = np.zeros(_vocab_size)
    x[first_char_index] = 1
    gen_text = first_char

    hidden = kwargs['hidden']
    W_xh = kwargs['W_xh']
    W_hh = kwargs['W_hh']
    W_hy = kwargs['W_hy']
    b = kwargs['b']
    b_prime = kwargs['b_prime']

    for i in range(_text_size):
        hidden, y, probs = forward_pass(x, hidden, W_xh, W_hh,
                             W_hy, b, b_prime)
        # get random char from probs
        gen_char_index = np.random.choice(_vocab_size, p=probs)
        # gen_char_index = np.argmax(probs)
        gen_char = _index_char_dict[gen_char_index]

        gen_text = ''.join([gen_text, gen_char])
        x = np.zeros(_vocab_size)
        x[gen_char_index] = 1

    return gen_text


def train(_input_text, _char_index_dict, _vocab_size, _batch_size, _hidden_layer_size, _learning_rate):

    """

    :param _input_file:
    :param _batch_size:
    :param _hidden_layer_size:
    :param _learning_rate:
    :return:
    """

    W_xh, W_hh, W_hy, b, b_prime = initialize(_vocab_size, _hidden_layer_size)

    epochs = 2000
    mem_factor = 0
    mW_xh, mW_hh, mW_hy = np.zeros_like(W_xh), np.zeros_like(W_hh), np.zeros_like(W_hy)
    mb, mb_prime = np.zeros_like(b), np.zeros_like(b_prime)  # memory variables for Adagrad

    counter = -1

    for j in range(epochs):

        cont = True
        start_index = 0
        end_index = start_index+_batch_size

        while cont:
        # while counter <= 500:

            counter  += 1

            # hidden_prev *= mem_factor
            d_hprev_Wxh = np.zeros(W_xh.shape)
            d_hprev_Whh = np.zeros(W_hh.shape)
            d_hprev_b = np.zeros(b.shape)

            hidden_prev = np.zeros(_hidden_layer_size, )

            #next batch
            input_chars = _input_text[start_index: end_index]
            target_chars = _input_text[start_index+1: end_index+1]
            # print(input_chars, target_chars)
            loss = 0

            dW_xh = np.zeros_like(W_xh)
            dW_hh = np.zeros_like(W_hh)
            dW_hy = np.zeros_like(W_hy)
            db = np.zeros_like(b)
            db_prime = np.zeros_like(b_prime)

            # print('Starting batch on {0} characters...'.format(len(input_chars)))

            for i in range(len(input_chars)):

                # print('Data point {0}'.format(i))

                char = input_chars[i]
                target_char = target_chars[i]
                x = np.zeros(_vocab_size,)
                index = _char_index_dict[char]
                target_index = _char_index_dict[target_char]
                x[index] = 1

                #forward pass
                hidden, y, probs = forward_pass(x, hidden_prev, W_xh, W_hh, W_hy, b, b_prime)

                #gradients - back prop
                dy = cp.deepcopy(probs)
                dy[target_index] -= 1

                dW_hy += np.matmul(dy[:, np.newaxis], hidden[:, np.newaxis].T)
                db_prime += dy

                dh = np.matmul(W_hy.T, dy[:, np.newaxis]).reshape(hidden.shape[0],)

                d_h_hraw = 1-np.square(hidden)

                dh_raw = np.matmul(np.diag(d_h_hraw), dh).reshape(hidden.shape[0],)

                #tier1
                dW_xh += np.matmul(dh_raw[:, np.newaxis], x[:, np.newaxis].T)
                dW_hh += np.matmul(dh_raw[:, np.newaxis], hidden_prev[:, np.newaxis].T)
                db += dh_raw

                dh_prev = np.matmul(W_hh.T, dh_raw)

                #tier2
                dW_xh += np.matmul(np.diag(dh_prev), d_hprev_Wxh)
                dW_hh += np.matmul(np.diag(dh_prev), d_hprev_Whh)
                db += np.matmul(np.diag(dh_prev), d_hprev_b)

                # print(dh_prev, d_hprev_Wxh, dW_xh)


                #save for next iter
                d_h_Wxh = np.matmul(d_h_hraw[:, np.newaxis], x[:, np.newaxis].T)
                d_h_Whh = np.matmul(d_h_hraw[:, np.newaxis], hidden_prev[:, np.newaxis].T)
                d_h_b = d_h_hraw
                d_hprev_Wxh = cp.deepcopy(d_h_Wxh)
                d_hprev_Whh = cp.deepcopy(d_h_Whh)
                d_hprev_b = cp.deepcopy(d_h_b)

                #update loss
                loss += -np.log(probs[target_index])

                #update hidden
                hidden_prev = cp.deepcopy(hidden)

            # if loss < 10 and _learning_rate >= .0001:
            #     _learning_rate *= .9

            input_indices = [_char_index_dict[char] for char in input_chars]
            # print(counter, np.sum(dW_xh), loss)

            for dparam in [dW_xh, dW_hh, dW_hy, db, db_prime]:
                np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients

            #update parameters by adagrad update
            for param, dparam, mem in zip([W_xh, W_hh, W_hy, b, b_prime],
                                          [dW_xh, dW_hh, dW_hy, db, db_prime],
                                          [mW_xh, mW_hh, mW_hy, mb, mb_prime]):
                mem += dparam * dparam
                param += -_learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update
                # param += -_learning_rate * dparam

            #print loss for batch
            # if counter % 10 == 0:
            # print('Loss: {0}'.format(loss))
            # print(W_xh, W_xh)

            # counter += 1

            #see if next batch exists
            start_index = start_index + _batch_size
            end_index = end_index + _batch_size
            if start_index >= len(_input_text)-1:
                cont = False
            elif end_index >= len(_input_text):
                end_index = len(_input_text)-1
                if start_index == end_index:
                    cont = False

        print('End of epoch {0}'.format(j+1))

    return W_xh, W_hh, W_hy, b, b_prime, loss


def main(_input_file, _batch_size, _hidden_layer_size, _learning_rate):

    input_text, chars = read_file_return_text(_input_file)
    vocab_size = len(chars)

    print('Vocab size: {0}'.format(len(input_text)))

    char_index_dict = {char: i for i, char in enumerate(chars)}
    index_char_dict = {v: k for k, v in char_index_dict.items()}

    W_xh, W_hh, W_hy, b, b_prime, loss = train(input_text, char_index_dict, vocab_size, _batch_size,
                                         _hidden_layer_size, _learning_rate)

    args = {'W_xh': W_xh, 'W_hh': W_hh, 'W_hy': W_hy, 'b': b, 'b_prime': b_prime, 'hidden': np.zeros(_hidden_layer_size,)}

    sample_text = sample_from_model(vocab_size, index_char_dict, 200, **args)

    print(loss)
    print(sample_text)

    pass


def parse_arguments():
    """
    :return: parsed arguments
    """

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--f', type=str, dest='input_file')
    arg_parser.add_argument('--b', type=int, dest='batch_size')
    arg_parser.add_argument('--h', type=int, dest='hidden_layer_size')
    arg_parser.add_argument('--lr', type=float, dest='learning_rate')

    return arg_parser.parse_args()


if __name__ == "__main__":

    #parse arguments
    arguments = parse_arguments()
    main(arguments.input_file, arguments.batch_size, arguments.hidden_layer_size, arguments.learning_rate)












