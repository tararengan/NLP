"""
Author: Tara Rengarajan
Read sample text,run an RNN and generate text
"""

#imports
import numpy as np
import argparse


def read_file_return_text(_input_file):
    """
    :param _input_file:
    :return:  chars in file
    """

    with open(_input_file, 'r') as f:
        text = f.read()

    chars = set([char for char in text])
    return text, chars


def initialize(_vocab_size, _hidden_layer_size):
    """
    :param _vocab_size:
    :param _hidden_layer_size:
    :return:
    """

    W_xh = np.zeros(_hidden_layer_size, _vocab_size)
    W_hh = np.zeros(_hidden_layer_size, _hidden_layer_size)
    W_hy = np.zeros(_vocab_size, _hidden_layer_size)

    b = np.zeros(_hidden_layer_size,)
    b_prime = np.zeros(_vocab_size,)

    return W_xh, W_hh, W_hy, b, b_prime


def forward_pass(_input, _hidden, _W_xh, _W_hh, _W_hy, _b, _b_prime):


    hidden = np.tanh(np.mm(_W_xh, _input) + np.mm(_W_hh, _hidden) + _b)
    y = np.mm(_W_hy, hidden) + _b_prime
    probs = np.exp(y)/np.sum(np.exp(y))

    return hidden, y, probs



def main(_input_file, _batch_size, _hidden_layer_size):


    input_text, chars = read_file_return_text(_input_file)
    vocab_size = len(chars)

    char_index_dict = {char:i for i, char in enumerate(chars)}
    index_char_dict = {v:k for k, v in char_index_dict.items()}

    W_xh, W_hh, W_hy, b, b_prime = initialize(vocab_size, _hidden_layer_size, _batch_size)


    cont = True
    hidden_prev = np.zeros(_hidden_layer_size,)
    start_index = 0


    while cont == True:

        input_chars = input_text[start_index: start_index+_batch_size]

        for i in range(len(input_chars)):
            char = input_chars[i]
            x = np.zeros(vocab_size,)
            index = char_index_dict[char]
            x[index] = 1

            #forward pass
            hidden, y, probs = forward_pass(input_chars, hidden_prev, W_xh, W_hh, W_hy, b, b_prime)

            #gradients - back prop
            dy = probs
            dy[index] += -1

            dW_hy = np.mm(dy, hidden.T)
            db_prime = dy

            dh = np.mm(W_hy.T, dy)

            dh_raw = np.dot(dh, 1-np.square(hidden))

            #tier1
            dW_xh = np.mm(dh_raw, x.T)
            dW_hh = np.mm(dh_raw, hidden_prev.T)
            db = dh_raw

            






        #save for next iter








def parse_arguments():
    """
    :return: parsed arguments
    """

    arg_parser = argparse.Argument_Parser()
    arg_parser.add_argument('--f', type=str, dest='input_file')
    arg_parser.add_argument('--b', type=int, dest='batch_size')
    arg_parser.add_argument('--h', type=int, dest='hidden_layer_size')

    return arg_parser.parse_args()


if __name__ == 'main':

    #parse arguments
    arguments = parse_arguments()
    main(arguments.input_file, arguments.batch_size, arguments.hidden_layer_size)












