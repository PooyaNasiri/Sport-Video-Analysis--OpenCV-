/********************************************************************************
  Project: Sport Video Analisis
  Author: Pooya Nasiri (Student ID: 2071437)
  Course: Computer Vision — University of Padova
  Instructor: Prof. Stefano Ghidoni
  Notes: Original work by the author. Built with C++17 and OpenCV on the official Virtual Lab.
         No external source code beyond standard libraries and OpenCV.
********************************************************************************/
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <iostream>
#include "detection.h"
#include "classification.h"
#include "heatmap.h"

int main(int argc,char **argv){
    if(argc<2){ std::cerr<<"Usage: "<<argv[0]<<" <video_file>\n"; return -1; }
    cv::VideoCapture cap(argv[1]); if(!cap.isOpened()){ std::cerr<<"Error: could not open "<<argv[1]<<"\n"; return -1; }
    std::ofstream det("ours.csv"); det<<"frame,x1,y1,x2,y2,team\n";
    cv::Ptr<cv::BackgroundSubtractor> bg=cv::createBackgroundSubtractorMOG2(500,16,false);
    double fps=cap.get(cv::CAP_PROP_FPS); int delay=fps>0?(int)(1000.0/fps):30;
    cv::Mat frame; int idx=0; Heatmap hm;
    std::vector<cv::Scalar> teamColors; teamColors.push_back(cv::Scalar(0,0,255)); teamColors.push_back(cv::Scalar(255,0,0)); teamColors.push_back(cv::Scalar(0,255,0));
    while(cap.read(frame)){
        std::vector<cv::Rect> boxes=detectPlayers(frame,bg);
        std::vector<std::pair<cv::Rect,int> > cls=classifyPlayers(frame,boxes);
        for(size_t i=0;i<cls.size();i++){
            cv::Rect b=cls[i].first; int t=cls[i].second;
            det<<idx<<","<<b.x<<","<<b.y<<","<<(b.x+b.width)<<","<<(b.y+b.height)<<","<<t<<"\n";
        }
        for(size_t i=0;i<cls.size();i++){
            cv::Rect b=cls[i].first; int t=cls[i].second; int cidx=(t==0||t==1)?t:2;
            cv::rectangle(frame,b,teamColors[cidx],2);
            cv::putText(frame,(t==0)?"Team A":(t==1)?"Team B":"Unknown",b.tl()+cv::Point(0,-5),cv::FONT_HERSHEY_SIMPLEX,0.5,teamColors[cidx],1);
        }
        hm.update(frame,cls);
        idx++;
        cv::imshow("Football Player Detection",frame);
        char k=(char)cv::waitKey(delay); if(k==27||k=='q') break;
    }
    hm.saveAndShow();
    cv::waitKey(0);
    det.close(); cap.release(); cv::destroyAllWindows(); return 0;
}
