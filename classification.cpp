/********************************************************************************
  Project: Sport Video Analisis
  Author: Pooya Nasiri (Student ID: 2071437)
  Course: Computer Vision — University of Padova
  Instructor: Prof. Stefano Ghidoni
  Notes: Original work by the author. Built with C++17 and OpenCV on the official Virtual Lab.
         No external source code beyond standard libraries and OpenCV.
********************************************************************************/
#include "classification.h"
#include <map>

static std::vector<cv::Mat> teamFeatureAnchors;
static int anchorCount=0;
static const int MAX_ANCHOR_FRAMES=10;
static bool teamAnchorsInitialized=false;
static std::map<int,std::pair<cv::Rect,int> > lastFrameBoxes;
static int nextID=0;
static const int teamsCount=2;

static cv::Vec3f avgNonGreenLab(const cv::Mat &roi){
    cv::Mat hsv; cv::cvtColor(roi,hsv,cv::COLOR_BGR2HSV);
    cv::Mat g; cv::inRange(hsv,cv::Scalar(35,40,40),cv::Scalar(90,255,255),g);
    cv::Mat lab; cv::cvtColor(roi,lab,cv::COLOR_BGR2Lab); lab.convertTo(lab,CV_32F);
    cv::Mat r=lab.reshape(1,lab.rows*lab.cols);
    std::vector<cv::Vec3f> v; v.reserve(r.rows);
    for(int i=0;i<r.rows;i++) if(g.at<uchar>(i)==0) v.push_back(r.at<cv::Vec3f>(i));
    if(v.empty()) return cv::Vec3f(0,0,0);
    std::sort(v.begin(),v.end(),[](const cv::Vec3f &a,const cv::Vec3f &b){return cv::norm(a)>cv::norm(b);});
    int n=(int)std::min<size_t>(v.size(),500);
    cv::Vec3f s(0,0,0); for(int i=0;i<n;i++) s+=v[i];
    return s*(1.0f/n);
}

static int findClosestBox(const cv::Rect &cur,const std::map<int,std::pair<cv::Rect,int> > &last){
    int best=-1; double dmin=50.0;
    for(std::map<int,std::pair<cv::Rect,int> >::const_iterator it=last.begin();it!=last.end();++it){
        cv::Rect p=it->second.first;
        cv::Point2f c1=(cur.tl()+cur.br())*0.5f, c2=(p.tl()+p.br())*0.5f;
        double d=cv::norm(c1-c2);
        if(d<dmin){ dmin=d; best=it->first; }
    }
    return best;
}

std::vector<std::pair<cv::Rect,int> > classifyPlayers(const cv::Mat &frame,const std::vector<cv::Rect> &boxes){
    std::vector<cv::Vec3f> feats; feats.reserve(boxes.size());
    for(size_t i=0;i<boxes.size();i++){
        cv::Rect sb=boxes[i]&cv::Rect(0,0,frame.cols,frame.rows);
        if(sb.area()<=0){ feats.push_back(cv::Vec3f(0,0,0)); continue; }
        cv::Mat roi=frame(sb); cv::resize(roi,roi,cv::Size(32,64));
        feats.push_back(avgNonGreenLab(roi));
    }
    if(feats.empty()) return std::vector<std::pair<cv::Rect,int> >();
    cv::Mat X((int)feats.size(),3,CV_32F);
    for(int i=0;i<X.rows;i++){ X.at<float>(i,0)=feats[i][0]; X.at<float>(i,1)=feats[i][1]; X.at<float>(i,2)=feats[i][2]; }
    if(X.rows<teamsCount) return std::vector<std::pair<cv::Rect,int> >();
    cv::Mat labels,centers;
    cv::kmeans(X,teamsCount,labels,cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT,10,1.0),5,cv::KMEANS_PP_CENTERS,centers);
    if(anchorCount<MAX_ANCHOR_FRAMES){
        if(teamFeatureAnchors.empty()){ for(int i=0;i<centers.rows;i++) teamFeatureAnchors.push_back(centers.row(i).clone()); }
        else{
            for(int i=0;i<centers.rows;i++){
                if(teamFeatureAnchors[i].size()!=centers.row(i).size()||teamFeatureAnchors[i].type()!=centers.row(i).type())
                    teamFeatureAnchors[i]=centers.row(i).clone();
                else
                    teamFeatureAnchors[i]=teamFeatureAnchors[i]*0.9f+centers.row(i)*0.1f;
            }
        }
        anchorCount++; if(anchorCount==MAX_ANCHOR_FRAMES) teamAnchorsInitialized=true;
    }
    std::vector<int> mapLab(teamsCount,-1); std::vector<bool> used(teamsCount,false);
    for(int fixed=0;fixed<teamsCount;fixed++){
        float md=FLT_MAX; int bj=-1;
        for(int i=0;i<teamsCount;i++){
            if(used[i]) continue;
            float d=cv::norm(teamFeatureAnchors[fixed]-centers.row(i));
            if(d<md){ md=d; bj=i; }
        }
        if(bj!=-1){ mapLab[bj]=fixed; used[bj]=true; }
    }
    std::vector<std::pair<cv::Rect,int> > out; out.reserve(boxes.size());
    std::map<int,std::pair<cv::Rect,int> > now;
    for(size_t i=0;i<boxes.size();i++){
        int raw=labels.at<int>((int)i);
        int team=mapLab[raw];
        int mid=findClosestBox(boxes[i],lastFrameBoxes);
        if(mid!=-1){
            if(lastFrameBoxes.at(mid).second!=team) team=lastFrameBoxes.at(mid).second;
            now[mid]=std::make_pair(boxes[i],team);
        }else{
            now[nextID++]=std::make_pair(boxes[i],team);
        }
        out.push_back(std::make_pair(boxes[i],team));
    }
    lastFrameBoxes=now;
    return out;
}
