# https://etsin.avointiede.fi/dataset/urn-nbn-fi-csc-kata20180603133722588355

import os
import urllib.request
import shutil
import tarfile
import hashlib
import numpy as np
import pandas as pd

valid_surfaces = ["100", "110", "111"]

_cu_tgz_url = "https://avaa.tdata.fi/openida/dl.jsp?pid=urn:nbn:fi:csc-ida-4x201806072015018412442s"
_cu_tgz_filename = "Cu_ANN_processed.tgz"
_cu_tgz_sha256 = "61aa30bfa00ccc31c951c42b37ecd4b96f788d40b4910d9d5805da80fbf34cba"


def _get_local_filename(storage_path):
    return os.path.join(storage_path, _cu_tgz_filename)


def _download(storage_path, verbose):
    cu_tgz_fp = open(_get_local_filename(storage_path), "wb+")
    if verbose:
        print("Downloading the Cu migration barrier data set")
    with urllib.request.urlopen(_cu_tgz_url) as response:
        shutil.copyfileobj(response, cu_tgz_fp)

    cu_tgz_fp.seek(0)
    h = hashlib.new("sha256")
    while True:
        data = cu_tgz_fp.read(2**20)
        if data:
            h.update(data)
        else:
            break
    if h.hexdigest() != _cu_tgz_sha256:
        raise RuntimeError("downloaded file has invalid hash")


def load_cu_migration_barriers(surface, storage_path="data", allow_download=True, verbose=False):
    if surface not in valid_surfaces:
        raise ValueError("surface must be one of {}".format(valid_surfaces))

    filename = _get_local_filename(storage_path)

    if not os.path.isfile(filename):
        if allow_download:
            _download(storage_path, verbose)
        else:
            raise RuntimeError("file {} not found and allow_download != True")

    fp_26d = tarfile.open(name=filename).extractfile(
        "Cu_ANN_processed/barriers/Cu_20170906_{}.26d".format(surface))

    df = pd.read_csv(fp_26d,
                     comment='#',
                     sep=r'\s+',
                     header=None,
                     dtype=np.float32)

    num_features = df.shape[1] - 1
    X = df.values[:, :num_features]
    y = df.values[:, num_features]

    return X, y
