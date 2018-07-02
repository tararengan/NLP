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



def main(_input_file, _batch_size, _hidden_layer_size):


    input_text, chars = read_file_return_text(_input_file)
    vocab_size = len(chars)

    W_xh, W_hh, W_hy, b, b_prime = initialize(vocab_size, _hidden_layer_size, _batch_size)




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












