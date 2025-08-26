/********************************************************************************
  Project: Sport Video Analisis
  Author: Pooya Nasiri (Student ID: 2071437)
  Course: Computer Vision — University of Padova
  Instructor: Prof. Stefano Ghidoni
  Notes: Original work by the author. Built with C++17 and OpenCV on the official Virtual Lab.
         No external source code beyond standard libraries and OpenCV.
********************************************************************************/
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <vector>
namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <labels_dir> <video_path> [out_csv=yolo.csv]\n";
        return 1;
    }
    fs::path labels_dir = argv[1];
    std::string video_path = argv[2];
    std::string out_csv = (argc >= 4) ? argv[3] : "yolo.csv";

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        std::cerr << "Cannot open video\n";
        return 1;
    }
    const double W = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const double H = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    std::ofstream out(out_csv);
    out << "frame,x1,y1,x2,y2\n";

    std::regex num_re("(\\d+)");
    std::vector<fs::path> files;
    for (auto &p : fs::directory_iterator(labels_dir))
        if (p.is_regular_file() && p.path().extension() == ".txt")
            files.push_back(p.path());
    std::sort(files.begin(), files.end());

    for (auto &p : files)
    {
        std::smatch m;
        std::string fname = p.filename().string();
        if (!std::regex_search(fname, m, num_re))
            continue;
        int frame = std::stoi(m[1].str());

        std::ifstream in(p);
        std::string line;
        while (std::getline(in, line))
        {
            if (line.empty())
                continue;
            std::istringstream ss(line);
            int cls;
            double xc, yc, w, h;
            ss >> cls >> xc >> yc >> w >> h;
            if (cls != 0)
                continue; // keep 'person' only
            double x1 = (xc - w / 2.0) * W;
            double y1 = (yc - h / 2.0) * H;
            double x2 = (xc + w / 2.0) * W;
            double y2 = (yc + h / 2.0) * H;
            out << frame << "," << x1 << "," << y1 << "," << x2 << "," << y2 << "\n";
        }
    }
    std::cout << "Wrote " << out_csv << "\n";
    return 0;
}