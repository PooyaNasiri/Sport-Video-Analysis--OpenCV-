/********************************************************************************
  Project: Sport Video Analisis
  Author: Pooya Nasiri (Student ID: 2071437)
  Course: Computer Vision — University of Padova
  Instructor: Prof. Stefano Ghidoni
  Notes: Original work by the author. Built with C++17 and OpenCV on the official Virtual Lab.
         No external source code beyond standard libraries and OpenCV.
********************************************************************************/
#include "heatmap.h"

Heatmap::Heatmap(){ colors.push_back(cv::Scalar(0,0,255)); colors.push_back(cv::Scalar(255,0,0)); colors.push_back(cv::Scalar(0,255,0)); }

void Heatmap::update(const cv::Mat &frame,const std::vector<std::pair<cv::Rect,int> > &classified){
    if(accum.empty()){ accum=cv::Mat::zeros(frame.size(),CV_32FC3); first=frame.clone(); }
    for(size_t i=0;i<classified.size();i++){
        cv::Mat t=cv::Mat::zeros(frame.size(),CV_32FC3);
        int team=classified[i].second;
        if(team<0||team>=(int)colors.size()) team=2; // Unknown -> green
        cv::Point c=(classified[i].first.tl()+classified[i].first.br())*0.5;
        cv::circle(t,c,20,colors[team],-1,cv::LINE_AA);
        accum+=t;
    }
}

void Heatmap::saveAndShow(){
    if(accum.empty()) return;
    cv::Mat blr,hm8,ov;
    cv::GaussianBlur(accum,blr,cv::Size(0,0),15);
    cv::normalize(blr,blr,0,255,cv::NORM_MINMAX);
    blr.convertTo(hm8,CV_8UC3);
    cv::addWeighted(first,0.5,hm8,0.5,0,ov);
    cv::imshow("Combined Heatmap",hm8);
    cv::imshow("Heatmap Overlay",ov);
    cv::imwrite("combined_heatmap.png",hm8);
    cv::imwrite("heatmap_overlay.png",ov);
}
