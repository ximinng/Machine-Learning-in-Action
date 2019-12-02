# -*- coding: utf-8 -*-
"""
   Description :   Machine Learning in Action
   Author :        xxm
"""
import sys
import os
# import loguru

path = {
    'BASE_PATH': os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    'EXAMPLE_PATH': os.path.abspath(os.path.dirname(os.path.dirname(__file__))).__add__('/examples')
}

__all__ = ['cluster', 'linear_model', 'naive_bayes', 'neighbors', 'neural_network', 'svm', 'tree', 'utils', 'path',
           'metrics']

if __name__ == '__main__':
    pass
else:
    pass
