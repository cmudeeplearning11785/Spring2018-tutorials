import io
import os
import pickle
from collections import OrderedDict

import torch


def make_chunks(inpath, outpath, chunk_size=1024 * 1024 * 20):
    """
    Split file into several chunks
    :param inpath: input path
    :param outpath: output path
    :param chunk_size: max chunk size in bytes
    :return: None
    """
    i = 0
    print("Input file: {}".format(inpath))
    with open(inpath, 'rb') as fin:
        while True:
            chunk = fin.read(chunk_size)
            if len(chunk) > 0:
                with open(outpath.format(i), 'wb') as fout:
                    fout.write(chunk)
                print("Output file: {}".format(outpath.format(i)))
                i += 1
            else:
                break


def read_chunks(inpath):
    """
    Read several chunks into a memory buffer
    :param inpath: format string for each chunk
    :return: Buffer
    """
    data = io.BytesIO()
    i = 0
    while os.path.exists(inpath.format(i)):
        with open(inpath.format(i), 'rb') as fin:
            data.write(fin.read())
        i += 1
    if i == 0:
        raise FileNotFoundError("missing file: {}".format(inpath.format(i)))
    data.seek(0)
    return data


def torch_to_numpy(inpath, outpath):
    """
    Convert torch save file to a pickle save file
    :param inpath: torch save path
    :param outpath: path for new pickle save
    :return: None
    """
    data = torch.load(inpath)
    cdata = OrderedDict([(k, w.cpu().numpy()) for k, w in data.items()])
    with open(outpath, 'wb') as f:
        pickle.dump(cdata, f)


def load_from_numpy(f):
    """
    Read data from a buffer and convert each element to torch.
    :param f: buffer
    :return: dictionary of torch tensors
    """
    cdata = pickle.load(f)
    data = OrderedDict([(k, torch.from_numpy(w)) for k, w in cdata.items()])
    return data


if __name__ == '__main__':
    # Example of usage
    model_path = '../output/model-v2/model-00000099.tar'  # saved torch state_dict
    numpy_path = model_path + '.npy'  # numpy dump of file
    chunk_path = model_path + '.npy.{}'  # format for each chunk

    # To write your model in chunks
    # Convert torch model to pickled numpy arrays
    torch_to_numpy(model_path, numpy_path)
    # Split pickled file into multiple chunks
    make_chunks(numpy_path, chunk_path)

    # To read your model in chunks
    # Read the chunks
    data = read_chunks(chunk_path)
    # Load the data
    state_dict = load_from_numpy(data)
    # Load dictionary into your model
    # model.load_state_dict(state_dict)
