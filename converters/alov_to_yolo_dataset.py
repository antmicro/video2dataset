#!/usr/bin/env python3

"""
ALOV-to-YOLO detection dataset converter.

Converts the video sequences in the tracking dataset in ALOV format to
detection dataset in the format used for training YOLO architectures.

The tracking dataset directory structure should look similarly to:

    dataset_dir:
      sequence-1
        00000001.jpg
        00000002.jpg
        00000003.jpg
        00000004.jpg
        00000005.jpg
        00000006.jpg
        ...
        annotations.ann
      sequence-2
        00000001.jpg
        00000002.jpg
        00000003.jpg
        ...
        annotations.ann
      sequence-3
        00000001.jpg
        00000002.jpg
        ...
        annotations.ann

The script also requires a `sequence_object_classes` file.
This file describes to which class the tracked object in a given sequence
belongs. Each line should look as:

    <sequence-directory-name> <class-name>

For the above sequences, this file may look as follows:

    sequence-1 dog
    sequence-2 cat
    sequence-3 dog

This means that the final detection dataset will have two classes.

In the end, the directory will contain:

    data/
        train/
            00000001.jpg
            00000001.txt
            00000002.jpg
            00000002.txt
            00000003.jpg
            00000003.txt
            00000004.jpg
            00000004.txt
            ...
        valid/
            00000001.jpg
            00000001.txt
            00000002.jpg
            00000002.txt
            00000003.jpg
            00000003.txt
            00000004.jpg
            00000004.txt
            ...
    data.names
    train.txt
    test.txt
    data.data
    backup/

Where:

* `data` - contains `train` and `valid` directories.
  Each of them contain pairs of `JPEG` and `txt` files, the input image and
  annotations.
* `data.names` - contains class names. Names' order determines the class ID
  (starting from 0) in the `txt` files
* `train.txt` - list of files for training
* `test.txt` - list of files for validation
* `data.data` - file with number of classes, and paths to the above files
* `backup` - directory for darknet training that will contain the weights.
"""

from pathlib import Path
import argparse
from collections import namedtuple
import random
import sys
import math
import shutil
from skimage import io
from typing import List


Sequence = namedtuple('Sequence', ['directory', 'class_id'])


def convert_alov_entry_to_yolo(
        alov_sequence : Sequence,
        alov_annotation_line : List[str],
        frame_width : int,
        frame_height : int) -> List:
    """
    Converts entry from .ann file to a YOLO detection entry.

    Each line in ALOV dataset consists of (assuming the bounding box is not
    rotated in any way):
    
        frame-id x1 y1 x2 y1 x1 y2 x2 y2

    This function takes (x1,y1) and (x2,y2) points and returns the single entry
    of detection bounding box in YOLO dataset annotation format:
    
        class-id x y width height

    The class-id is taken from the `sequence_object_classes` file.
    (x, y) is the center of the bounding box, and (width, height) is the size
    of the bounding box.

    The coordinates in the ALOV annotations are not normalized, and the
    coordinates in the YOLO annotations are normalized.

    Parameters
    ----------
    alov_sequence : Sequence
        Pair (directory, class_id), where directory is a directory with a
        single video sequence, and class_id is the ID of the class of the
        tracked object
    alov_annotation_line : List[str]
        Split line from annotation file
    frame_width : int
        Width of the frame
    frame_height : int
        Height of the frame

    Returns
    -------
    List : list containing class_id, (x,y) coordinates and (width, height)
    """
    #                   0  1  2  3  4  5  6  7  8
    # alov line: frame-id x1 y1 x2 y1 x1 y2 x2 y2
    width = (float(alov_annotation_line[3]) - float(alov_annotation_line[1])) / frame_width
    height = (float(alov_annotation_line[6]) - float(alov_annotation_line[2])) / frame_height
    x_center = float(alov_annotation_line[1]) / frame_width + width / 2
    y_center = float(alov_annotation_line[2]) / frame_height + height / 2

    assert x_center - width / 2 >= 0
    assert y_center - height / 2 >= 0
    assert x_center + width / 2 <= 1.0
    assert y_center + height / 2 <= 1.0

    return (alov_sequence.class_id, x_center, y_center, width, height)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "input",
        help="Directory containing directories with ALOV video frames and annotations",
        type=Path)
    parser.add_argument(
        "sequence_object_classes",
        help="\
A file containing classes of objects in video sequences.\
Each line of the file contains ALOV dataset's sequence\
subdirectory name and the class that's being tracked",
        type=Path)
    parser.add_argument(
        "output",
        help="Directory for images and YOLO annotations",
        type=Path)
    parser.add_argument(
        "--directory-with-empty-frames",
        help="Directory containing frames without the objects",
        type=Path)
    parser.add_argument(
        "--train-frames-percentage",
        help="Percentage of frames used for training from each video sequence (0.0-1.0)",
        default=0.3,
        type=float)
    parser.add_argument(
        "--validation-frames-percentage",
        help="Percentage of frames used for validation from each video sequence (0.0-1.0)",
        default=0.1,
        type=float)
    parser.add_argument(
        "--use-every",
        help="Use every X frame. From every video sequence every X-th frame is used for random shuffling. Allowed values are in range (1-120)",
        type=int,
        choices=range(1,120),
        default=1,
        metavar='')
    parser.add_argument(
        "--seed",
        help="Seed for the random shuffling",
        type=int)

    args = parser.parse_args()

    if args.seed is None:
        seed = random.randrange(sys.maxsize)
        rng = random.Random(seed)
    else:
        seed = args.seed
        rng = random.Random(seed)

    print(f'The seed is:  {seed}')

    if not args.input.is_dir():
        print(f'{args.input} is not a directory')
        return 1

    try:
        if not args.output.exists():
            args.output.mkdir()
    except FileNotFoundError:
        print(f'Parent for {args.output} does not exist')
        return 1

    if not args.output.is_dir():
        print(f'{args.output} is not a directory')
        return 1

    if len(list(args.output.rglob('*'))) != 0:
        print(f'{args.output} is not empty. Use empty directory to store the results')
        return 1

    if not args.sequence_object_classes.is_file():
        print(f'{args.sequence_object_classes} is not a file.')
        return 1

    # prepare output directory structure
    (args.output / 'data').mkdir()
    outimagedirt = args.output / 'data' / 'train'
    outimagedirt.mkdir()
    outimagedirv = args.output / 'data' / 'valid'
    outimagedirv.mkdir()

    sequences = []

    classes = dict()
    classidtoclassname = dict()
    max_class_id = 0

    with open(args.sequence_object_classes, 'r') as file_with_sequences:
        for sequence in file_with_sequences:
            sequence_subdir, classname = sequence.split(' ')
            sequence_dir = args.input / sequence_subdir
            if not sequence_dir.is_dir():
                print(f'{sequence_dir} does not exist or is not a directory')
                return 1
            if classname not in classes:
                classes[classname] = max_class_id
                classidtoclassname[max_class_id] = classname
                max_class_id += 1
            sequences.append(Sequence(sequence_dir, classes[classname]))

    outputimageid = 0
    trainfilelist = []
    validfilelist = []

    for sequence in sequences:
        # load annotation file
        annotationsfile = list(sequence.directory.rglob('*.ann'))
        if len(annotationsfile) != 1:
            print(f'There should be only one annotation file in {sequence.directory}')
        annotationsfile = annotationsfile[0]
        entries = []

        # read annotations
        with open(annotationsfile, 'r') as annotations:
            entries = [(line.strip()).split(' ') for line in annotations]
        if len(entries) == 0:
            print(f'No entries in the {annotationsfile}, skipping...')
        # shuffle annotations
        entries = entries[::args.use_every]
        rng.shuffle(entries)

        # split annotations into train set and validation set
        train_count = math.ceil(args.train_frames_percentage * len(entries))
        validation_count = min(math.ceil(args.validation_frames_percentage * len(entries)), len(entries) - train_count)
        train = entries[:train_count]
        validation = entries[train_count:train_count + validation_count]

        # store train samples
        for alov_annotation_line in train:
            framepath = (sequence.directory / alov_annotation_line[0].zfill(8)).with_suffix('.jpg')
            frame = io.imread(framepath)
            frame_height, frame_width, _ = frame.shape
            yolodata = convert_alov_entry_to_yolo(sequence, alov_annotation_line, frame_width, frame_height)
            fileid = str(outputimageid).zfill(8)
            outimagepath = outimagedirt / f'{fileid}.jpg'
            shutil.copyfile(framepath, outimagepath)
            print(f'TRAIN: {framepath} => {outimagepath}, {" ".join([str(q) for q in yolodata])}')
            with open(outimagedirt / f'{fileid}.txt', 'w') as yolofile:
                yolofile.write(' '.join([str(q) for q in yolodata]))
            trainfilelist.append(outimagedirt / f'{fileid}.jpg')
            outputimageid += 1

        # store validation samples
        for alov_annotation_line in validation:
            framepath = (sequence.directory / alov_annotation_line[0].zfill(8)).with_suffix('.jpg')
            frame = io.imread(framepath)
            frame_height, frame_width, _ = frame.shape
            yolodata = convert_alov_entry_to_yolo(sequence, alov_annotation_line, frame_width, frame_height)
            fileid = str(outputimageid).zfill(8)
            outimagepath = outimagedirv / f'{fileid}.jpg'
            shutil.copyfile(framepath, outimagepath)
            print(f'VALID: {framepath} => {outimagepath}, {" ".join([str(q) for q in yolodata])}')
            with open(outimagedirv / f'{fileid}.txt', 'w') as yolofile:
                yolofile.write(' '.join([str(q) for q in yolodata]))
            validfilelist.append(outimagedirv / f'{fileid}.jpg')
            outputimageid += 1

    if args.directory_with_empty_frames is not None:
        empty_images = list(args.directory_with_empty_frames.rglob('*.jpg'))
        rng.shuffle(empty_images)
        train_count = math.ceil(args.train_frames_percentage * len(empty_images))
        validation_count = min(math.ceil(args.validation_frames_percentage * len(empty_images)), len(empty_images) - train_count)
        train = empty_images[:train_count]
        validation = empty_images[train_count:train_count + validation_count]
        for image in train:
            fileid = str(outputimageid).zfill(8)
            outimagepath = outimagedirt / f'{fileid}.jpg'
            print(f'EMPTY TRAIN: {image} => {outimagepath}')
            shutil.copyfile(image, outimagepath)
            with open(outimagedirt / f'{fileid}.txt', 'w'):
                pass
            trainfilelist.append(outimagedirt / f'{fileid}.jpg')
            outputimageid += 1
        for image in validation:
            fileid = str(outputimageid).zfill(8)
            outimagepath = outimagedirv / f'{fileid}.jpg'
            print(f'EMPTY VALID: {image} => {outimagepath}')
            shutil.copyfile(image, outimagepath)
            with open(outimagedirv / f'{fileid}.txt', 'w'):
                pass
            validfilelist.append(outimagedirv / f'{fileid}.jpg')
            outputimageid += 1


    with open(args.output / 'data.names', 'w') as names:
        for clsid in range(len(classidtoclassname)):
            names.write(f'{classidtoclassname[clsid]}')
    
    with open(args.output / 'train.txt', 'w') as trainfile:
        for framename in trainfilelist:
            trainfile.write(f'{framename}\n')

    with open(args.output / 'test.txt', 'w') as validfile:
        for framename in validfilelist:
            validfile.write(f'{framename}\n')

    backupdir = args.output / 'backup'
    backupdir.mkdir()

    with open(args.output / 'data.data', 'w') as datafile:
        datafile.write(f'classes= {len(classidtoclassname)}\n')
        datafile.write(f'train= train.txt\n')
        datafile.write(f'valid= test.txt\n')
        datafile.write(f'names= data.names\n')
        datafile.write(f'backup= backup/')

if __name__ == '__main__':
    main()
