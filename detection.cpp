/********************************************************************************
  Project: Sport Video Analisis
  Author: Pooya Nasiri (Student ID: 2071437)
  Course: Computer Vision — University of Padova
  Instructor: Prof. Stefano Ghidoni
  Notes: Original work by the author. Built with C++17 and OpenCV on the official Virtual Lab.
         No external source code beyond standard libraries and OpenCV.
********************************************************************************/
#include "detection.h"

static cv::Mat maskGreenField(const cv::Mat &hsv){
    cv::Mat mask,dilated,eroded,out;
    cv::inRange(hsv,cv::Scalar(40,40,40),cv::Scalar(90,255,255),mask);
    cv::Mat k=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(5,5));
    cv::dilate(mask,dilated,k);
    cv::erode(dilated,eroded,k);
    cv::erode(eroded,eroded,k);
    cv::erode(eroded,eroded,k);
    cv::erode(eroded,eroded,k);
    std::vector<std::vector<cv::Point> > cts;
    cv::findContours(eroded,cts,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
    out=cv::Mat::zeros(mask.size(),CV_8UC1);
    for(size_t i=0;i<cts.size();i++){
        if(cv::contourArea(cts[i])>1000.0) cv::drawContours(out,cts,(int)i,cv::Scalar(255),cv::FILLED);
    }
    cv::imshow("Green Field Mask",out);
    return out;
}

static cv::Mat maskGreenPlayers(const cv::Mat &hsvFieldMaskedBgr){
    cv::Mat hsv; cv::cvtColor(hsvFieldMaskedBgr,hsv,cv::COLOR_BGR2HSV);
    cv::Mat greenMask,blackMask,mask;
    cv::inRange(hsv,cv::Scalar(40,40,40),cv::Scalar(90,255,255),greenMask);
    cv::inRange(hsv,cv::Scalar(0,0,0),cv::Scalar(10,10,10),blackMask);
    cv::bitwise_or(greenMask,blackMask,mask);
    cv::bitwise_not(mask,mask);
    int d=5;
    cv::Mat elem=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(2*d+1,2*d+1),cv::Point(d,d));
    cv::dilate(mask,mask,elem);
    cv::Mat result;
    hsvFieldMaskedBgr.copyTo(result,mask);
    cv::imshow("Players",result);
    return mask;
}

static std::vector<cv::Rect> mergeBoxes(const std::vector<cv::Rect> &inputBoxes){
    std::vector<cv::Rect> merged; std::vector<bool> used(inputBoxes.size(),false);
    for(size_t i=0;i<inputBoxes.size();i++){
        if(used[i]) continue;
        cv::Rect cur=inputBoxes[i]; bool changed;
        do{
            changed=false;
            for(size_t j=0;j<inputBoxes.size();j++){
                if(i==j||used[j]) continue;
                cv::Rect o=inputBoxes[j];
                if((cur&o).area()>0||cur.contains(o.tl())||cur.contains(o.br())||o.contains(cur.tl())||o.contains(cur.br())){
                    cur=cur|o; used[j]=true; changed=true;
                }
            }
        }while(changed);
        merged.push_back(cur); used[i]=true;
    }
    std::vector<cv::Rect> cleaned;
    for(size_t i=0;i<merged.size();i++){
        bool inside=false;
        for(size_t j=0;j<merged.size();j++){
            if(i==j) continue;
            if(merged[j].contains(merged[i].tl())&&merged[j].contains(merged[i].br())){ inside=true; break; }
        }
        if(!inside) cleaned.push_back(merged[i]);
    }
    return cleaned;
}

std::vector<cv::Rect> detectPlayers(const cv::Mat &frame, cv::Ptr<cv::BackgroundSubtractor> &bgSub){
    cv::Mat fgMask,hsv,fieldMask,fieldMaskedBgr,playersMask,combined;
    bgSub->apply(frame,fgMask,0.01);
    cv::cvtColor(frame,hsv,cv::COLOR_BGR2HSV);
    fieldMask=maskGreenField(hsv);
    fieldMaskedBgr=cv::Mat::zeros(frame.size(),frame.type());
    frame.copyTo(fieldMaskedBgr,fieldMask);
    playersMask=maskGreenPlayers(fieldMaskedBgr);
    cv::bitwise_and(fgMask,playersMask,combined);
    std::vector<std::vector<cv::Point> > contours; std::vector<cv::Rect> boxes;
    cv::findContours(combined,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
    for(size_t i=0;i<contours.size();i++){
        double area=cv::contourArea(contours[i]); if(area<30) continue;
        cv::Rect b=cv::boundingRect(contours[i]);
        if(b.width<10||b.height<20||b.width>100||b.height>200) continue;
        boxes.push_back(b);
    }
    return mergeBoxes(boxes);
}
