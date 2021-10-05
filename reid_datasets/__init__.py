from __future__ import absolute_import

from .market1501 import Market1501
from .dukemtmcreid import DukeMTMCreID


__factory = {
    'market1501': Market1501,
    'dukemtmcreid': DukeMTMCreID
}

def names():
    return sorted(__factory.keys())

def create(name, root, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name.
    root : str
        The root path of all dataset directories.
    """
    if name not in __factory:
        raise KeyError("unknown dataset:", name)
    return __factory[name](root, **kwargs)

def get_dataset(name, root):
    print("get_dataset() is deprecated. use create() instead.")
    return create(name, root)
