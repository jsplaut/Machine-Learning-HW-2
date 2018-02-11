
"""
Utility routines for unzipping, reading and examining the MNIST dataset.

See sample.py file for an example of how to use this module and to understand
the format of the data.

Author: RR
"""

import cPickle as cPickle
import gzip
import numpy

def pretty_print(image_example):
    """ Pretty prints an MNIST training example.

    Parameters:
        image_example: a 1x784 numpy array corresponding to the features of
                       a single image.

    Returns:
        None.
    """
    print (numpy.array_str(image_example, precision=1, max_line_width=142))
    

def load_data():
    """ Returns the MNIST dataset in two pieces - a 60k training set and a
    10k test set.

    Returns:
        A tuple containing the training set and test set, each of which are
        a tuple of numpy matrices themselves.
    """
    data = gzip.open("mnist.pkl.gz", "rb")
    train_set, valid_set, test_set = cPickle.load(data)
    data.close()

    # Combine validation and train folds to recreate the master 60k set.
    new_images = numpy.concatenate((train_set[0], valid_set[0]))
    new_labels = numpy.concatenate((train_set[1], valid_set[1]))

    train_set = (new_images, new_labels)
    
    return (train_set, test_set)

