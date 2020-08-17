#!/usr/bin/env python

"""
Script to convert EXR files to Matlab MAT files.

E.g.: exr_to_mat.py *.exr

Can use `--output_dir` to specify where to save the MAT files.
"""

import argparse
import pathlib

import numpy as np
import scipy.io

import OpenEXR


def run():

    parser = argparse.ArgumentParser()

    parser.add_argument("exr_filenames", nargs="+")

    parser.add_argument("--output_dir", required=False)

    args = parser.parse_args()

    convert_files(exr_filenames=args.exr_filenames, output_dir=args.output_dir)


def convert_files(exr_filenames, output_dir):

    for exr_filename in exr_filenames:

        exr_path = pathlib.Path(exr_filename).resolve().absolute()

        if not exr_path.exists():
            print(f"{exr_path} doesn't exist; skipping")
            continue

        if exr_path.suffix != ".exr":
            print(f"{exr_path} doesn't end in .exr; skipping")
            continue

        if output_dir is None:
            output_path = exr_path.parent
        else:
            output_path = pathlib.Path(output_dir)

        mat_path = output_path / (exr_path.stem + ".mat")

        if mat_path.exists():
            print(f"Output path {mat_path} exists; not overwriting")
            continue

        convert_file(exr_filename=exr_path, output_filename=mat_path)


def convert_file(exr_filename, output_filename):

    img = read_exr(exr_path=str(exr_filename))

    scipy.io.savemat(file_name=output_filename, mdict={"img": img})


def read_exr(exr_path, squeeze=True, channel_order=None):
    """Read an EXR file and return a numpy array.

    Parameters
    ----------
    exr_path: string
        Path to the EXR file to read.
    squeeze: boolean, optional
        Whether to remove single channel dimensions.
    channel_order: collection of strings, optional
        Explicitly specify the channel name ordering.

    """

    exr_file = OpenEXR.InputFile(exr_path)

    header = exr_file.header()

    channels = list(header["channels"].keys())

    data_window = header["dataWindow"]

    img_size = (
        data_window.max.x - data_window.min.x + 1,
        data_window.max.y - data_window.min.y + 1,
    )

    if channel_order is None:

        if all([k in channels for k in ["Y", "A"]]):
            channel_order = ["Y", "A"]
        elif all([k in channels for k in ["R", "G", "B"]]):
            channel_order = ["R", "G", "B"]
        elif all([k in channels for k in ["X", "Y", "Z"]]):
            channel_order = ["X", "Y", "Z"]
        else:
            channel_order = channels

    type_lut = {"HALF": np.float16, "FLOAT": np.float32}

    img = np.full(
        (img_size[1], img_size[0], len(channels)),
        np.nan,
        dtype=type_lut[str(header["channels"][channel_order[0]].type)],
    )

    for (i_channel, curr_channel) in enumerate(channel_order):

        pixel_type = header["channels"][curr_channel].type

        channel_str = exr_file.channel(curr_channel, pixel_type)
        channel_img = np.frombuffer(channel_str, dtype=type_lut[str(pixel_type)])
        channel_img.shape = (img_size[1], img_size[0])
        img[..., i_channel] = channel_img

    assert np.sum(np.isnan(img)) == 0

    exr_file.close()

    if squeeze:
        img = np.squeeze(img)

    return img


if __name__ == "__main__":
    run()
