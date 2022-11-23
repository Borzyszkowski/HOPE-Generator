""" General purpose utility functions to limit duplication of code """

import json
import logging
from collections import OrderedDict
from copy import copy
from pathlib import Path

import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()


def set_random_seed():
    """ Fix random seeds for reproducibility """
    logging.warning("Using Random Seed for reproducibility. Please remove it for the real experiment!")
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        logging.warning("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        logging.warning(f"The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are available.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))

    if n_gpu_use > 0:
        torch.cuda.empty_cache()
        gpu_brand = torch.cuda.get_device_name(0)
        logging.info(f'Using {gpu_brand} for training')
    else:
        logging.info(f'Using CPU for training')

    return device, list_ids


def append2dict(source, data):
    for k in data.keys():
        if isinstance(data[k], list):
            source[k] += data[k].astype(np.float32)
        else:
            source[k].append(data[k].astype(np.float32))


def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = np.array(array.todencse(), dtype=dtype)
    elif torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    return array


def DotDict(in_dict):
    out_dict = copy(in_dict)
    for k, v in out_dict.items():
        if isinstance(v, dict):
            out_dict[k] = DotDict(v)
    return dotdict(out_dict)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def parse_npz(npz, allow_pickle=True):
    npz = np.load(npz, allow_pickle=allow_pickle)
    npz = {k: npz[k].item() for k in npz.files}
    return DotDict(npz)


def params2torch(params, dtype=torch.float32):
    return {k: torch.from_numpy(v).type(dtype) for k, v in params.items()}


def prepare_params(params, frame_mask, dtype=np.float32):
    return {k: v[frame_mask].astype(dtype) for k, v in params.items()}


def euler(rots, order='xyz', units='deg'):
    rots = np.asarray(rots)
    single_val = False if len(rots.shape) > 1 else True
    rots = rots.reshape(-1, 3)
    rotmats = []

    for xyz in rots:
        if units == 'deg':
            xyz = np.radians(xyz)
        r = np.eye(3)
        for theta, axis in zip(xyz, order):
            c = np.cos(theta)
            s = np.sin(theta)
            if axis == 'x':
                r = np.dot(np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), r)
            if axis == 'y':
                r = np.dot(np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]), r)
            if axis == 'z':
                r = np.dot(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]), r)
        rotmats.append(r)
    rotmats = np.stack(rotmats).astype(np.float32)
    if single_val:
        return rotmats[0]
    else:
        return rotmats
