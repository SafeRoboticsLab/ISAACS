import argparse
import pickle


class Range(object):

  def __init__(self, start, end):
    self.start = start
    self.end = end

  def __eq__(self, other):
    return self.start <= other <= self.end

  def __contains__(self, item):
    return self.__eq__(item)

  def __iter__(self):
    yield self

  def __str__(self):
    return '[{0},{1}]'.format(self.start, self.end)


def bool_type(string):
  if string.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif string.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def save_obj(obj, filename):
  with open(filename + '.pkl', 'wb') as f:
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(filename):
  with open(filename + '.pkl', 'rb') as f:
    return pickle.load(f)
