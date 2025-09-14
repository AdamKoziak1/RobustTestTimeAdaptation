# coding=utf-8
from alg.algs.ERM import ERM

ALGORITHMS = ['ERM']


def get_algorithm_class(algorithm_name="ERM"):
    if algorithm_name not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
