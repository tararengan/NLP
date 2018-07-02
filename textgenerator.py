"""
Author: Tara Rengarajan
Read sample text,run an RNN and generate text
"""

#imports
import numpy as np
import argparse


def read_return_chars(_input_file):
    """
    :param _input_file:
    :return:  chars in file
    """

    with open(_input_file, 'r') as f:
        text = f.read()

    chars = set([char for char in text])
    return text, chars



def initialize(_vocab_size, _hidden_layer_size,)



def main(input_file, batch_size, hidden_layer_size):
    """
    :param input_file:
    :param batch_size:
    :param hidden_layer_size:
    :return:
    """





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












