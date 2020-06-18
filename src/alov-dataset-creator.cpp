#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstdio>
#include <vector>
#include "helper/bounding_box.h"
#include "train/tracker_trainer.h"
#include <sys/stat.h>
#include <cxxopts.hpp>
#include <dirent.h>
#include <fstream>
#include <algorithm>

cv::Mat3b canvas;
bool toogleplay;
cv::Point start;
bool selected;

bool paused = true;
bool nextframe = false;
bool manual = false;
bool autostage = false;

Tracker *tracker;
Regressor *regressor;

BoundingBox _bbox;

std::vector<std::string> frames;
std::vector<BoundingBox> staged;
std::vector<BoundingBox> unstaged;
std::vector<int> movieid;
cv::Mat frame;
int currframe;

int firstframe = 0;
int lastframe = -1;

std::string videoname = "";
std::string framesdir = "";
std::string outputdir = "";

int waitkeyduration = 1;

bool toggletracking = true;

bool fileAccessible(std::string filename)
{
    std::ifstream file(filename);
    return file.good();
}

bool checkExtension(const char *name, const char *ext)
{
    return (strlen(name) >= strlen(ext)) && (!strcmp(name + strlen(name) - strlen(ext), ext));
}

int getFiles(std::string directory,
                std::vector<std::string> &files, const char *ext)
{
    DIR *dp;
    struct dirent *ep;
    dp = opendir(directory.c_str());
    if (dp != NULL)
    {
        ep = readdir(dp);
        while (ep != NULL)
        {
            if (!ext || checkExtension(ep->d_name, ext))
            {
                if (strcmp(".", ep->d_name) != 0 &&
                    strcmp("..", ep->d_name) != 0)
                        files.push_back(directory + ep->d_name);
            }
            ep = readdir(dp);
        }
        closedir(dp);
    }
    else return -ENOENT;
    return 0;
}

int saveVideo()
{
    for (int i = firstframe; i < lastframe; i++)
    {
        if (staged[i].x1_ == 0 && staged[i].x2_ == 0 && staged[i].y1_ == 0 && staged[i].y2_ == 0)
        {
            printf("Not all frames within range are staged\n");
            return 1;
        }
    }
    std::string annotationsfile = outputdir + "annotations" + std::to_string(firstframe) + "-" + std::to_string(lastframe) + ".ann";
    std::ofstream annotations(annotationsfile);
    int count = 1;
    for (int i = firstframe; i < lastframe; i++)
    {
        annotations << count << " "
            << staged[i].x1_ + 1 << " " << staged[i].y1_ + 1 << " "
            << staged[i].x2_ + 1 << " " << staged[i].y1_ + 1 << " "
            << staged[i].x1_ + 1 << " " << staged[i].y2_ + 1 << " "
            << staged[i].x2_ + 1 << " " << staged[i].y2_ + 1 << std::endl;
        std::ostringstream path;
        path << outputdir;
        path << std::setfill('0') << std::setw(8) << count;
        path << ".jpg";
        cv::Mat tosave = cv::imread(frames[i]);
        cv::imwrite(path.str(), tosave);
        count++;
    }
    annotations.close();

    printf("Annotations saved to %s\n", annotationsfile.c_str());

    return 0;
}

void interpolateStagedFrames(int from, int to)
{
    printf("Interpolating from %d to %d\n", from, to);
    if (!(from < to && to - from > 1))
        return;
    double stepx1 = (staged[to].x1_ - staged[from].x1_) / (double(to-from));
    double stepx2 = (staged[to].x2_ - staged[from].x2_) / (double(to-from));
    double stepy1 = (staged[to].y1_ - staged[from].y1_) / (double(to-from));
    double stepy2 = (staged[to].y2_ - staged[from].y2_) / (double(to-from));

    for (int i = from + 1; i < to; i++)
    {
        staged[i].x1_ = staged[from].x1_ + (i - from) * stepx1;
        staged[i].x2_ = staged[from].x2_ + (i - from) * stepx2;
        staged[i].y1_ = staged[from].y1_ + (i - from) * stepy1;
        staged[i].y2_ = staged[from].y2_ + (i - from) * stepy2;
    }
}

int loadAnnotations(std::string inputannotations)
{
    std::ifstream annotations(inputannotations);

    if (!annotations.good())
    {
        printf("Annotations file not available\n");
        return 1;
    }

    int currid = firstframe;
    int previd = firstframe;
    int prevannid = -1;

    int annid;
    int step = 1;
    double Ax, Ay, Bx, By, Cx, Cy, Dx, Dy;

    while (currid < lastframe)
    {
        printf("Processing frame %d\n", currid);
        if (annotations >> annid >> Ax >> Ay >> Bx >> By >> Cx >> Cy >> Dx >> Dy)
        {
            printf("%d %f %f %f %f %f %f %f %f\n", annid, Ax, Ay, Bx, By, Cx, Cy, Dx, Dy);
            if (prevannid == -1) currid = firstframe;
            else
            {
                int step = annid - prevannid;
                currid += step;
            }
            staged[currid].x1_ = std::min(Ax, std::min(Bx, std::min(Cx, Dx))) - 1;
            staged[currid].y1_ = std::min(Ay, std::min(By, std::min(Cy, Dy))) - 1;
            staged[currid].x2_ = std::max(Ax, std::max(Bx, std::max(Cx, Dx))) - 1;
            staged[currid].y2_ = std::max(Ay, std::max(By, std::max(Cy, Dy))) - 1;
            if (currid - previd > 1) interpolateStagedFrames(previd, currid);
            previd = currid;
            prevannid = annid;
        }
        else
        {
            printf("Finished loading annotations\n");
            annotations.close();
            return 0;
        }
    }
    return 0;
}

void callbackfunc(int event, int x, int y, int flags, void* userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        toogleplay = false;
        start = cv::Point(x,y);
    }
    if (event == cv::EVENT_LBUTTONUP)
    {
        cv::Point end = cv::Point(x,y);
        std::vector<float> bbox(4);
        if (end.x < start.x)
        {
	        bbox[0] = end.x;
	        bbox[2] = start.x;
        }
        else
        {
	        bbox[0] = start.x;
	        bbox[2] = end.x;
        }
        if (end.y < start.y)
        {
	        bbox[1] = end.y;
	        bbox[3] = start.y;
        }
        else
        {
	        bbox[1] = start.y;
            bbox[3] = end.y;
        }
        _bbox.x1_ = bbox[0];
        _bbox.y1_ = bbox[1];
        _bbox.x2_ = bbox[2];
        _bbox.y2_ = bbox[3];
        printf("Initializing tracking... ");
        tracker->Init(frame, _bbox, regressor);
        printf("Initialized.\n");
        toogleplay = true;
        selected = true;
        nextframe = true;
    }
}

bool keyboardControl(int key)
{
    switch (key)
    {
    case 27: // ESC - quit
        return false;
    case 32: // SPACE - toggle pause/play
        paused = !paused;
        break;
    case 106: // J - move backwards
        if (paused)
        {
            if (currframe > 0) currframe--;
            nextframe = true;
        }
        break;
    case 107: // K - move forward
        if (paused)
        {
            if (currframe < frames.size() - 1) currframe++;
            nextframe = true;
        }
        break;
    case 49: // 1 - stage single
        if (paused)
            staged[currframe] = unstaged[currframe];
        break;
    case 97: // A - stage all unstaged
        if (paused)
        {
            for (int i = 0; i < staged.size(); i++)
            {
                staged[i] = unstaged[i];
            }
        }
        break;
    case 99: // C - toggle continuos stage
        autostage = !autostage;
        break;
    case 50: // 2 - reset single
        if (paused)
        {
            unstaged[currframe] = staged[currframe];
            tracker->Init(frame, staged[currframe], regressor);
        }
        break;
    case 114: // R - set all unstaged to stage (reset)
        if (paused)
        {
            for (int i = 0; i < staged.size(); i++)
            {
                unstaged[i] = staged[i];
            }
            tracker->Init(frame, staged[currframe], regressor);
        }
        break;
    case 115: // S - save the annotations
        saveVideo();
        break;
    case 40: // ( - set frame as the beginning
        if (currframe != lastframe) firstframe = currframe;
        break;
    case 41: // ) - set frame as the ending
        if (currframe != firstframe) lastframe = currframe;
        break;
    case 43: // + - speed up two times (up to 1x speed)
        if (waitkeyduration > 1) waitkeyduration /= 2;
        printf("Time for frame:  %dms\n", waitkeyduration);
        break;
    case 105: // I - initialize with current unstaged
        tracker->Init(frame, unstaged[currframe], regressor);
        break;
    case 111: // O - initialize with current staged
        tracker->Init(frame, staged[currframe], regressor);
        break;
    case 45: // - - slow down two times
        waitkeyduration *= 2;
        printf("Time for frame:  %dms\n", waitkeyduration);
        break;
    case 113: // Q - toggle tracker usage
        toggletracking = !toggletracking;
        printf("Tracking turned %s\n", toggletracking ? "on" : "off");
        break;
    case 38: // & - move to first frame
        currframe = firstframe;
        break;
    case 42: // * - move to last frame
        currframe = lastframe;
        break;
    case 104: // H - show help
        printf("\n=============================================================\n");
        printf("H     - help\n");
        printf("ESC   - quit\n");
        printf("SPACE - toggle pause/play\n");
        printf("J     - move backwards\n");
        printf("K     - move forward\n");
        printf("1     - stage single\n");
        printf("A     - stage all unstaged\n");
        printf("C     - toggle continuos staging\n");
        printf("2     - reset single\n");
        printf("R     - reset all unstaged to staged\n");
        printf("S     - save the annotations\n");
        printf("(     - set frame as the beginning\n");
        printf(")     - set frame as the ending\n");
        printf("+     - speed up movie two times (up to 1x speed\n");
        printf("-     - slow down move two times\n");
        printf("I     - initialize tracker with current unstaged bounding box\n");
        printf("O     - initialize tracker with current staged bounding box\n");
        printf("Q     - toggle tracker usage\n");
        printf("&     - go to the first frame\n");
        printf("*     - go to the last frame\n");
        printf("=============================================================\n");
    }
    return true;
}

bool tryLoading(const char *datadir)
{
    return true;
}

int main(int argc, char *argv[])
{
    cxxopts::Options options("Dataset creator tool", "Tool for creating bounding boxes for objects in video frames for the tracking tasks, classification tasks (within bounding boxes) and detection tasks (single object per image)");

    std::string inputannotations;

    options.add_options()
        ("input-video", "Input video to extract labels from", cxxopts::value(videoname))
        ("frames-directory", "The directory containing frames from input video", cxxopts::value(framesdir))
        ("output-directory", "The directory containing labeled frames and annotations", cxxopts::value(outputdir))
        ("first-frame", "The id of the first frame (0-based)", cxxopts::value(firstframe))
        ("last-frame", "The id of the last frame (0-based)", cxxopts::value(lastframe))
        ("input-annotations", "Input .ann file containing the annotations from frames from first-frame to last-frame", cxxopts::value(inputannotations))
        ("h,help", "Prints help for the application")
    ;

    options.parse_positional({"frames-directory", "output-directory"});

    auto result = options.parse(argc, argv);

    if (result.count("help") != 0)
    {
        printf("%s\n", options.help().c_str());
    }

    if (framesdir == "")
    {
        printf("--frames-directory is a required argument, for storing frames from input video, or loading frames from a previous session\n");
        return 1;
    }

    if (framesdir[framesdir.size() - 1] != '/') framesdir += "/";
    if (outputdir[outputdir.size() - 1] != '/') framesdir += "/";

    printf("Starting program...\n");

    if (videoname != "" && framesdir != "")
    {
        if (getFiles(framesdir.c_str(), frames, "jpg") != 0)
        {
            printf("%s directory does not exist or you have not right permissions\n", framesdir.c_str());
            return 1;
        }
        else if (frames.size() > 0)
        {
            printf("%s directory is not empty, run the application without input video or clear this directory\n", framesdir.c_str());
            return 1;
        }
        printf("Converting video to jpg images...\n");

        cv::VideoCapture cap(videoname, cv::CAP_FFMPEG);

        if (!cap.isOpened())
        {
            printf("Error opening video stream or file\n");
            return -1;
        }

        int i = 0;
        while (true)
        {
            printf("Frame:  %d\n", i);
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) break;
            cv::resize(frame, frame, cv::Size(1024,576));
            std::ostringstream path;
            path << framesdir;
            path << std::setfill('0') << std::setw(8) << i;
            path << ".jpg";
            cv::imwrite(path.str(), frame);
            frames.push_back(path.str());
            BoundingBox bbox;
            bbox.x1_ = 0;
            bbox.x2_ = 0;
            bbox.y1_ = 0;
            bbox.y2_ = 0;
            staged.push_back(bbox);
            unstaged.push_back(bbox);
            movieid.push_back(0);
            i++;
        }
        cap.release();
    }
    else if (videoname == "" && framesdir != "")
    {
        if (getFiles(framesdir.c_str(), frames, "jpg") != 0)
        {
            printf("%s directory does not exist or you have not right permissions\n", framesdir.c_str());
            return 1;
        }
        else if (frames.size() == 0)
        {
            printf("%s directory is empty\n", framesdir.c_str());
            return 1;
        }
        std::sort(frames.begin(), frames.end());
        for (int i = 0; i < frames.size(); i++)
        {
            BoundingBox bbox;
            bbox.x1_ = 0;
            bbox.x2_ = 0;
            bbox.y1_ = 0;
            bbox.y2_ = 0;
            staged.push_back(bbox);
            unstaged.push_back(bbox);
        }
    }

    if (lastframe == -1) lastframe = frames.size() - 1;

    caffe::Caffe::SetDevice(0);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    printf("Set GPU Caffe mode\n");
    tracker = new Tracker(false);
    regressor = new Regressor("../nets/tracker.prototxt", "../nets/models/pretrained_model/tracker.caffemodel", 0, false);
    printf("Prepared tracker structures\n");
    toogleplay = true;
    selected = false;

    if (inputannotations != "")
    {
        if (loadAnnotations(inputannotations) != 0)
        {
            printf("Error loading annotations file\n");
            return 1;
        }
    }
    printf("Sucessfully loaded annotations\n");

    currframe = 0;

    cv::namedWindow("Frame", cv::WINDOW_NORMAL);
    cv::setMouseCallback("Frame",callbackfunc);
    frame = cv::imread(frames[currframe]);
    if (!frame.data)
    {
        printf("Frame not valid:  %s\n", frames[currframe].c_str());
        return 1;
    }
    printf("%s %d %d\n", frames[currframe].c_str(), frame.rows, frame.cols);
    canvas = cv::Mat3b(frame.rows, frame.cols, cv::Vec3b(0,0,0));
    frame.copyTo(canvas);

    BoundingBox fullframe({0, 0, (float)frame.cols, (float)frame.rows});

    while (true)
    {
        frame = cv::imread(frames[currframe]);
        frame.copyTo(canvas);
        if (toogleplay && !paused)
        {
            if (currframe + 1 < frames.size()) currframe++;
            frame.copyTo(canvas);
            nextframe = true;
        }
        if (selected && nextframe)
        {
            printf("Frame:  %d\n", currframe);
            if (toggletracking) tracker->Track(frame, regressor, &_bbox);
            else _bbox = unstaged[currframe];
            nextframe = false;
            _bbox.Draw(255,0,0,&canvas);
            unstaged[currframe] = _bbox;
            if (autostage) staged[currframe] = unstaged[currframe];
        }
        unstaged[currframe].Draw(255,0,0,&canvas);
        staged[currframe].Draw(255,255,255,&canvas);

        if (firstframe == currframe) fullframe.Draw(0,255,0,&canvas);
        if (lastframe == currframe) fullframe.Draw(0,0,255,&canvas);

        cv::imshow("Frame", canvas);
        if (!keyboardControl(cv::waitKey(waitkeyduration))) break;
    }
    delete tracker;
    delete regressor;
    return 0;
}
