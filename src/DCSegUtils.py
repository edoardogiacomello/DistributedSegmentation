

import numpy as np

ID_TO_LABEL = {16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse'}
LABEL_TO_ID = {'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19}
CHANNEL_ORDER = [0, 16, 17, 18, 19] # Order of channels in output segmentation and corresponding dataset labels
CHANNEL_NAMES = [ID_TO_LABEL[i] if i!=0 else 'other' for i in CHANNEL_ORDER]
ALL_LABELS = [ID_TO_LABEL[i]  for i in CHANNEL_ORDER if i!=0]
AGENT_NAMES = ['ag_{}'.format(l) for l in ALL_LABELS]

def softmax(X, theta = 1.0, axis = None):
    """
    from: https://nolanbconaway.github.io/blog/2017/softmax-numpy
    
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

# Feature scaling for visualization
def feature_scaling(x, A, B):
    ''' Perform linear scaling of an array of data between x.min() and x.max() to a new range [A,B]'''
    return A + (B-A)*(x-x.min())/(x.max()-x.min())




def numpy_to_pandas_series(data, index_prefix=None, index_values=None):
    '''Convert a multidimensional numpy array into a panda series having as many indices as the data dimensions and a single value column.
    Either index_prefix (a list of strings of length dim) or index_values (see MultiIndex.from_product()) should be defined. '''
    assert index_prefix or index_values
    assert not (index_prefix and index_values)
    if index_values:
        builder = index_values
    else:
        builder = [[pref + str(i) for i in range(dim) ] for dim, pref in zip(data.shape, index_prefix)]
            
    assert len(data.shape) == len(builder), "Data is shape " + str(data.shape) + " but index builder is long " + str(len(builder))
    
    import pandas as pd
    indices = pd.MultiIndex.from_product(builder)
    return pd.Series(data=data.flatten(), index=indices)
    
    
    