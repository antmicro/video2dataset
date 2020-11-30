# ALOV dataset creator from video sequences with GOTURN aiding

This repository contains a tool for annotating videos for single object detection and tracking and converting the videos and annotations into ALOV dataset format.
The consecutive bounding boxes are proposed by the GOTURN tracker.

## Requirements

Follow the requirements in the [GOTURN install dependencies](https://github.com/davheld/GOTURN#install-dependencies).

## Building

Run the following command in the root directory of the project:

    mkdir build
    cd build
    cmake ..
    make -j`nproc`

## Obtaining weights

To download weights, run:

    wget http://cs.stanford.edu/people/davheld/public/GOTURN/trained_model/tracker.caffemodel -O nets/tracker.caffemodel

## Usage

To get the information about available flags, run:

    ./alov-dataset-creator -h

To start a new ALOV dataset in the from the video file, run:

    cd ./build
    mkdir dataset-dir/
    ./alov-dataset-creator --input-video video-file.mp4 dataset-dir/

This will extract frames from `video-file.mp4` and save them as JPG files.

If the frames are already available as JPG, the `--input-video` flag can be omitted.

After the GOTURN model loads, the GUI with the first frame will appear.

The GUI is as follows:

- green border of the window means the frame is the first in sequence for the ALOV dataset sequence,
- blue border of the window means the frome is the last in sequence for the ALOV dataset sequence,
- red bounding box means unstaged bounding box for the current frame,
- white bounding box means staged bounding box for the current frame (this bounding box will be saved).

Controls for GUI:

- `H` - displays help for GUI in the terminal with available key controls,
- `ESC` - quits the application without saving,
- `SPACE` - toggles pause/playing,
- `J` - move one frame backward,
- `K` - move one frame forward,
- `1` - stage unstaged (red) bounding box in the current frame,
- `A` - stage all unstaged bounding boxes,
- `C` - toggle automatically staging all consecutive bounding boxes,
- `2` - reset unstaged bounding box to the staged bounding box for the current frame (the tracker is reinitialzed),
- `R` - reset all unstaged bounding boxes to match staged bounding boxes (the tracker for the current frame is reinitialized),
- `S` - save the annotations,
- `(` - set the current frame to be the first frame in the ALOV sequence,
- `)` - set the current frame to be the last frame in the ALOV sequence,
- `+` - speed up playing the video sequence two times (up to 1x speed),
- `-` - slow down playing the video sequence two times,
- `I` - initialize the tracker with the current unstaged bounding box,
- `O` - initialize the tracker with the current staged bounding box,
- `Q` - toggle using tracker for consecutive frames,
- `&` - go to the first frame,
- `*` - go to the last frame.

In the beginning, select the object to track with mouse - the first bounding box will be marked as unstaged (red bounding box).
After this, press `SPACE` to automatically track the object with GOTURN tracker.
To pause, press `SPACE`.
You can use `J` and `K` to move across frames.
You can change the bounding box by re-selecting the bounding box with the mouse.
This will automatically reinitialize the tracker for the current frame.
To temporarily turn off the tracker (this will stop bounding box proposals and reinitialization), press `Q`.

To stage the bounding box for the current frame press `1`.
To stage all bounding boxes within the first and last frame press `A`.

To save annotations between the first and the last frame, all bounding boxes must be staged.
Only the staged bounding boxes will be saved, the unstaged bounding boxes will be ignored.
To save the annotations file, press `S`.
This will save annotations as `dataset-dir/annotations<first-frame-id>-<last-frame-id>.ann`.
After saving the annotations, close the application by pressing `ESC`.

To review the annotations files for a given `dataset-dir/annotations<first-frame-id>-<last-frame-id>.ann`, run:

    ./alov-dataset-creator dataset-dir/ --first-frame <first-frame-id> --last-frame <last-frame-id> --input-annotations dataset-dir/annotations<first-frame-id>-<last-frame-id>.ann

## Licensing

The sources are published under Apache 2.0 License, except for files located in the `third-party/` directory.
For those files the license is either enclosed in the file header or in a separate LICENSE file.
