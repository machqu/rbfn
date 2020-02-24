#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import pathlib
import shutil
import json

import util.dataset as dataset


def main(input_path, output_path, prefix):
    """
    Finds the best hyperparameter-optimized model for each surface type.

    :param input_path: Path that contains the raw tuning results.
    :param output_path: Path to which the result JSON will be saved.
    """
    for surface in dataset.valid_surfaces:
        best_score = float("inf")
        best_f = None
        p = pathlib.Path(input_path)
        for f in p.glob('{}-{}-*.json'.format(prefix, surface)):
            res = json.load(f.open())
            if res['avg_score'] < best_score:
                best_score = res['avg_score']
                best_f = f
        shutil.copyfile(
            best_f,
            os.path.join(output_path, "best-{}-{}.json".format(prefix, surface))
        )


if __name__ == "__main__":
    prefix = sys.argv[1]
    input_path = os.path.join("output", "{}_tuning_raw".format(prefix))
    output_path = os.path.join("output", "{}_tuning_combined".format(prefix))
    main(input_path, output_path, prefix)
