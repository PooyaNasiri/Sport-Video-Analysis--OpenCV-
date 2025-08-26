/********************************************************************************
  Project: Sport Video Analisis
  Author: Pooya Nasiri (Student ID: 2071437)
  Course: Computer Vision — University of Padova
  Instructor: Prof. Stefano Ghidoni
  Notes: Original work by the author. Built with C++17 and OpenCV on the official Virtual Lab.
         No external source code beyond standard libraries and OpenCV.
********************************************************************************/
#ifndef HEATMAP_H
#define HEATMAP_H
#include <opencv2/opencv.hpp>
#include <vector>
class Heatmap{
    cv::Mat accum; cv::Mat first; std::vector<cv::Scalar> colors;
public:
    Heatmap();
    void update(const cv::Mat &frame,const std::vector<std::pair<cv::Rect,int> > &classified);
    void saveAndShow();
};
#endif
