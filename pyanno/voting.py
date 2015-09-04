# Copyright (c) 2011, Enthought, Ltd.
# Authors: Pietro Berkes <pberkes@enthought.com>, Andrey Rzhetsky,
#          Bob Carpenter
# License: Modified BSD license (2-clause)

"""Utility functions."""

import numpy as np
from numpy import log


#: In annotations arrays, this is the value used to indicate missing values
MISSING_VALUE = -1


class PyannoError(ValueError):
    """ValueError subclass raised by pyAnno functions and methods.
    """
    pass


def labels_count(annotations, nclasses, missing_value_label=MISSING_VALUE):
    """Compute the total count of labels in observed annotations.

    Raises a PyannoValueError if there are no valid annotations.

    Arguments
    ---------
    annotations : array-like object, shape = (n_items, n_annotators)
        annotations[i,j] is the annotation made by annotator j on item i
    nclasses : int
        Number of label classes in `annotations`

    Returns
    -------
    count : list of length n_classes
        count[k] is the number of elements of class k in `annotations`
    """

    annotations = np.asarray(annotations)
    valid = annotations != missing_value_label
    nobservations = valid.sum()

    if nobservations == 0:
        # no valid observations
        raise PyannoValueError('No valid annotations')

    return list(np.bincount(annotations[valid], minlength=nclasses))


def majority_vote(annotations, missing_value_label=MISSING_VALUE):
    """Compute an estimate of the real class by majority vote.
    In case of ties, return the class with smallest number.
    If a row only contains invalid entries, return `MISSING_VALUE`.

    Arguments
    ---------
    annotations : array-like object, shape = (n_items, n_annotators)
        annotations[i,j] is the annotation made by annotator j on item i

    Return
    ------
    vote : list of length n_items
        vote[i] is the majority vote estimate for item i
    """

    annotations = np.asarray(annotations)
    nitems = annotations.shape[0]
    valid = annotations != missing_value_label

    vote = [0] * nitems

    for i in range(nitems):
        if not np.any(valid[i,:]):
            # no valid entries on this row
            vote[i] = missing_value_label
        else:
            count = np.bincount(annotations[i, valid[i,:]])
            vote[i] = count.argmax()

    return vote


def labels_frequency(annotations, nclasses):
    """Compute the total frequency of labels in observed annotations.

    Example:
    >>> labels_frequency([[1, 1, 2], [-1, 1, 2]], 4)
    array([ 0. ,  0.6,  0.4,  0. ])

    Arguments
    ---------
    annotations : array-like object, shape = (n_items, n_annotators)
        annotations[i,j] is the annotation made by annotator j on item i
    nclasses : int
        Number of label classes in `annotations`

    Returns
    -------
    freq : ndarray, shape = (n_classes, )
        freq[k] is the frequency of elements of class k in `annotations`, i.e.
        their count over the number of total of observed (non-missing) elements
    """
    return np.histogram(annotations, bins=range(nclasses+1), normed=True)[0]
    #annotations = np.array(annotations)
    #freq = lambda x: np.sum(annotations == x)
    #tot = np.sum(np.array(annotations >= 0)
    #return np.array(map(freq, range(nclasses)))/float(tot)
