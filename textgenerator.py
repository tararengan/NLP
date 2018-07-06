"""
Author: Tara Rengarajan
Read sample text,run an RNN and generate text
"""

#imports
import numpy as np
import argparse
import copy as cp
import math
import time
import sys


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

    # b = np.zeros(_hidden_layer_size,)
    b = np.zeros((_hidden_layer_size, 1))
    # b_prime = np.zeros(_vocab_size,)
    b_prime = np.zeros((_vocab_size, 1))

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
    y = np.matmul(_W_hy, hidden) + _b_prime
    probs = np.exp(y)/np.sum(np.exp(y))

    return hidden, y, probs


def sample_from_model(_vocab_size, _index_char_dict, _text_size, **kwargs):


    first_char_index  = np.random.randint(0, _vocab_size)
    first_char = _index_char_dict[first_char_index]
    print('First char: {0}'.format(first_char))
    x = np.zeros((_vocab_size, 1))
    x[first_char_index, 0] = 1
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
        gen_char_index = np.random.choice(_vocab_size, p=probs.reshape(probs.shape[0],))
        gen_char = _index_char_dict[gen_char_index]

        gen_text = ''.join([gen_text, gen_char])
        x = np.zeros((_vocab_size, 1))
        x[gen_char_index, 0] = 1

    return gen_text


def train_and_sample(_input_text, _char_index_dict, _index_char_dict, _vocab_size, _batch_size, _hidden_layer_size, _learning_rate):

    """

    :param _input_file:
    :param _batch_size:
    :param _hidden_layer_size:
    :param _learning_rate:
    :return:
    """

    W_xh, W_hh, W_hy, b, b_prime = initialize(_vocab_size, _hidden_layer_size)

    mW_xh, mW_hh, mW_hy = np.zeros_like(W_xh), np.zeros_like(W_hh), np.zeros_like(W_hy)
    mb, mb_prime = np.zeros_like(b), np.zeros_like(b_prime)  # memory variables for Adagrad

    counter = -1
    cont = True

    start_index = 0
    end_index = start_index+_batch_size
    hidden_prev = np.zeros((_hidden_layer_size, 1))

    while cont:

        starttime = time.time()

        #next batch
        counter  += 1

        if (end_index >= len(_input_text) - 1 ) or (counter == 0):
            start_index, end_index = 0, _batch_size
            hidden_prev = np.zeros((_hidden_layer_size, 1))
            d_hprev_Wxh, d_hprev_Whh, d_hprev_b = np.zeros(W_xh.shape), np.zeros(W_hh.shape), np.zeros(b.shape)


        # spit out text
        if counter % 10 == 0:
            args = {'W_xh': W_xh, 'W_hh': W_hh, 'W_hy': W_hy, 'b': b, 'b_prime': b_prime,
                    'hidden': np.zeros((_hidden_layer_size, 1))}
            sample_text = sample_from_model(_vocab_size, _index_char_dict, 200, **args)
            print(sample_text)

        input_indices = [_char_index_dict[char] for char in _input_text[start_index: end_index]]
        target_indices = [_char_index_dict[char] for char in _input_text[start_index+1: end_index+1]]

        loss = 0

        dW_xh, dW_hh, dW_hy = np.zeros_like(W_xh), np.zeros_like(W_hh), np.zeros_like(W_hy)
        db, db_prime = np.zeros_like(b), np.zeros_like(b_prime)

        for i in range(len(input_indices)):

            # x = np.zeros(_vocab_size,)
            x = np.zeros((_vocab_size, 1))
            index = input_indices[i]
            target_index = target_indices[i]
            # x[index] = 1
            x[index, 0] = 1

            #forward pass
            hidden, y, probs = forward_pass(x, hidden_prev, W_xh, W_hh, W_hy, b, b_prime)

            #gradients - back prop
            dy = cp.deepcopy(probs)
            dy[target_index, 0] -= 1

            dW_hy += np.matmul(dy, hidden.T)
            db_prime += dy

            dh = np.matmul(W_hy.T, dy)
            d_h_hraw = 1-np.square(hidden)
            dh_raw = np.multiply(d_h_hraw, dh)

            #tier1
            dW_xh += np.matmul(dh_raw, x.T)
            dW_hh += np.matmul(dh_raw, hidden_prev.T)
            db += dh_raw

            dh_prev = np.matmul(W_hh.T, dh_raw)

            #tier2
            dW_xh += np.matmul(np.diag(dh_prev.flatten()), d_hprev_Wxh)
            dW_hh += np.matmul(np.diag(dh_prev.flatten()), d_hprev_Whh)
            db += np.multiply(dh_prev, d_hprev_b)

            #save for next iter
            d_hprev_Wxh = np.matmul(d_h_hraw, x.T)
            d_hprev_Whh = np.matmul(d_h_hraw, hidden_prev.T)
            d_hprev_b = d_h_hraw

            #update loss
            loss += -np.log(probs[target_index, 0])

            #update hidden
            hidden_prev = cp.deepcopy(hidden)

        print(counter, loss)

        for dparam in [dW_xh, dW_hh, dW_hy, db, db_prime]:
            np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients

        endtime = time.time()
        print('Time: {0}'.format(endtime - starttime))

        if loss < .1:
            break

        #update parameters by adagrad update
        for param, dparam, mem in zip([W_xh, W_hh, W_hy, b, b_prime],
                                      [dW_xh, dW_hh, dW_hy, db, db_prime],
                                      [mW_xh, mW_hh, mW_hy, mb, mb_prime]):
            mem += dparam * dparam
            param += -_learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update
            # param += -_learning_rate * dparam

        #next batch
        start_index = start_index + _batch_size
        end_index = end_index + _batch_size


    return W_xh, W_hh, W_hy, b, b_prime, loss


def main(_input_file, _batch_size, _hidden_layer_size, _learning_rate):

    input_text, chars = read_file_return_text(_input_file)
    vocab_size = len(chars)

    print('Vocab size: {0}'.format(len(input_text)))

    char_index_dict = {char: i for i, char in enumerate(chars)}
    index_char_dict = {v: k for k, v in char_index_dict.items()}

    W_xh, W_hh, W_hy, b, b_prime, loss = train_and_sample(input_text, char_index_dict, index_char_dict, vocab_size, _batch_size,
                                         _hidden_layer_size, _learning_rate)

    args = {'W_xh': W_xh, 'W_hh': W_hh, 'W_hy': W_hy, 'b': b, 'b_prime': b_prime,
            'hidden': np.zeros((_hidden_layer_size, 1))}

    sample_text = sample_from_model(vocab_size, index_char_dict, 300, **args)
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












