/********************************************************************************
  Project: Sport Video Analisis
  Author: Pooya Nasiri (Student ID: 2071437)
  Course: Computer Vision — University of Padova
  Instructor: Prof. Stefano Ghidoni
  Notes: Original work by the author. Built with C++17 and OpenCV on the official Virtual Lab.
         No external source code beyond standard libraries and OpenCV.
********************************************************************************/
#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H
#include <opencv2/opencv.hpp>
#include <vector>
std::vector<std::pair<cv::Rect,int> > classifyPlayers(const cv::Mat &frame,const std::vector<cv::Rect> &boxes);
#endif
