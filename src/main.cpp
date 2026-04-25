#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
// Vehicle
// ─────────────────────────────────────────────────────────────────────────────
struct Vehicle {
    int                                  id;
    cv::Point2f                          centroid;
    cv::Point2f                          smoothCentroid;
    cv::Point2f                          prevSmooth;
    cv::Rect                             bbox;
    std::vector<cv::Point>               contour;
    std::vector<std::vector<cv::Point>>  subContours;
    int                                  framesLost;
    int                                  framesConfirmed;
    int                                  framesInsideTotal; // frames spent inside zone
    bool                                 confirmed;

    enum class ZoneState { OUTSIDE, ENTERING, INSIDE, EXITING };
    ZoneState   zoneState;
    int         zoneFrames;
    bool        counted;
};

// ─────────────────────────────────────────────────────────────────────────────
// Geometry
// ─────────────────────────────────────────────────────────────────────────────
bool pointInPolygon(const cv::Point2f& p,
                    const std::vector<cv::Point>& poly)
{
    int n=(int)poly.size(), crossings=0;
    for(int i=0;i<n;i++){
        cv::Point a=poly[i], b=poly[(i+1)%n];
        if(((a.y<=p.y)&&(b.y>p.y))||((b.y<=p.y)&&(a.y>p.y))){
            float t=(p.y-a.y)/(float)(b.y-a.y);
            if(p.x<a.x+t*(b.x-a.x)) crossings++;
        }
    }
    return (crossings%2)==1;
}

bool crossedLine(const cv::Point2f& prev, const cv::Point2f& curr,
                 const cv::Point& A, const cv::Point& B)
{
    auto cross2d=[](cv::Point2f o,cv::Point2f a,cv::Point2f b){
        return (a.x-o.x)*(b.y-o.y)-(a.y-o.y)*(b.x-o.x);
    };
    cv::Point2f a(A.x,A.y),b(B.x,B.y);
    float d1=cross2d(a,b,prev),d2=cross2d(a,b,curr);
    float d3=cross2d(prev,curr,a),d4=cross2d(prev,curr,b);
    return (((d1>0&&d2<0)||(d1<0&&d2>0))&&
            ((d3>0&&d4<0)||(d3<0&&d4>0)));
}

float euclidean(const cv::Point2f& a, const cv::Point2f& b)
{
    float dx=a.x-b.x,dy=a.y-b.y;
    return std::sqrt(dx*dx+dy*dy);
}

float iou(const cv::Rect& a, const cv::Rect& b)
{
    cv::Rect i=a&b;
    if(i.area()==0) return 0.0f;
    return i.area()/(float)(a.area()+b.area()-i.area());
}

float matchScore(const Vehicle& v,
                 const cv::Point2f& cen, const cv::Rect& box)
{
    return euclidean(v.smoothCentroid,cen)*(1.0f-iou(v.bbox,box)*0.5f);
}

// ─────────────────────────────────────────────────────────────────────────────
// Merge nearby boxes — expandPx=15, only same-vehicle fragments
// ─────────────────────────────────────────────────────────────────────────────
struct MergedBlob {
    cv::Rect                             bbox;
    std::vector<cv::Point>               contour;
    std::vector<std::vector<cv::Point>>  subContours;
};

std::vector<MergedBlob> mergeBoxes(
    const std::vector<cv::Rect>&               rawBoxes,
    const std::vector<std::vector<cv::Point>>& rawContours,
    int expandPx=15)
{
    if(rawBoxes.empty()) return {};
    auto expand=[&](const cv::Rect& r){
        return cv::Rect(r.x-expandPx,r.y-expandPx,
                        r.width+2*expandPx,r.height+2*expandPx);
    };
    std::vector<bool> merged(rawBoxes.size(),false);
    std::vector<MergedBlob> result;
    for(int i=0;i<(int)rawBoxes.size();i++){
        if(merged[i]) continue;
        cv::Rect cur=rawBoxes[i];
        std::vector<cv::Point> pts=rawContours[i];
        std::vector<std::vector<cv::Point>> subs={rawContours[i]};
        bool changed=true;
        while(changed){
            changed=false;
            for(int j=0;j<(int)rawBoxes.size();j++){
                if(merged[j]||j==i) continue;
                if((expand(cur)&expand(rawBoxes[j])).area()>0){
                    cur=cur|rawBoxes[j];
                    pts.insert(pts.end(),
                               rawContours[j].begin(),
                               rawContours[j].end());
                    subs.push_back(rawContours[j]);
                    merged[j]=true;
                    changed=true;
                }
            }
        }
        merged[i]=true;
        MergedBlob blob;
        blob.bbox=cur;
        cv::convexHull(pts,blob.contour);
        blob.subContours=subs;
        result.push_back(blob);
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Drawing state — single window zone editor
// ─────────────────────────────────────────────────────────────────────────────
struct DrawState {
    std::vector<cv::Point> points;
    bool                   done=false;
    cv::Point              mouse={0,0};
};
static DrawState gDraw;
static cv::Mat   gCurrentFrame;

void drawOverlay(cv::Mat& img)
{
    int n=(int)gDraw.points.size();
    if(n>=3){
        std::vector<std::vector<cv::Point>> poly={gDraw.points};
        cv::Mat layer=img.clone();
        cv::fillPoly(layer,poly,cv::Scalar(0,220,220));
        cv::addWeighted(layer,0.25,img,0.75,0,img);
        cv::polylines(img,poly,true,cv::Scalar(0,255,255),2);
    }
    for(int i=0;i+1<n;i++)
        cv::line(img,gDraw.points[i],gDraw.points[i+1],
                 cv::Scalar(0,255,255),2);
    if(n>0&&n<4)
        cv::line(img,gDraw.points.back(),gDraw.mouse,
                 cv::Scalar(80,255,80),1);
    for(int i=0;i<n;i++){
        cv::circle(img,gDraw.points[i],7,cv::Scalar(0,0,255),-1);
        cv::putText(img,std::to_string(i+1),
                    gDraw.points[i]+cv::Point(10,-8),
                    cv::FONT_HERSHEY_SIMPLEX,0.6,
                    cv::Scalar(255,255,0),2);
    }
    std::string hint;
    if     (n==0) hint="Click point 1";
    else if(n==1) hint="Click point 2  (ENTER = confirm as line)";
    else if(n==2) hint="Click point 3 for surface  —  or ENTER for line";
    else if(n==3) hint="Click point 4 to complete surface";
    cv::rectangle(img,cv::Rect(0,img.rows-50,img.cols,50),
                  cv::Scalar(0,0,0),-1);
    cv::putText(img,hint+"   |   RIGHT CLICK: undo   |   ESC: quit",
                cv::Point(15,img.rows-15),
                cv::FONT_HERSHEY_SIMPLEX,0.65,
                cv::Scalar(255,255,255),2);
}

void onMouse(int event, int x, int y, int, void*)
{
    if(gDraw.done) return;
    gDraw.mouse={x,y};
    if(event==cv::EVENT_MOUSEMOVE){
        cv::Mat tmp=gCurrentFrame.clone(); drawOverlay(tmp);
        cv::imshow("Vehicle Tracker",tmp);
    } else if(event==cv::EVENT_LBUTTONDOWN){
        if((int)gDraw.points.size()<4){
            gDraw.points.push_back({x,y});
            if((int)gDraw.points.size()==4) gDraw.done=true;
        }
        cv::Mat tmp=gCurrentFrame.clone(); drawOverlay(tmp);
        cv::imshow("Vehicle Tracker",tmp);
    } else if(event==cv::EVENT_RBUTTONDOWN){
        if(!gDraw.points.empty()) gDraw.points.pop_back();
        cv::Mat tmp=gCurrentFrame.clone(); drawOverlay(tmp);
        cv::imshow("Vehicle Tracker",tmp);
    }
}

int main()
{
    cv::VideoCapture cap("../video.mp4");
    if(!cap.isOpened()){std::cerr<<"Cannot open video.\n";return -1;}

    int frameW=(int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frameH=(int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout<<"Resolution: "<<frameW<<"x"<<frameH<<"\n";

    cv::Ptr<cv::BackgroundSubtractorMOG2> pMOG2=
        cv::createBackgroundSubtractorMOG2(300,40,true);
    pMOG2->setShadowThreshold(0.5);
    pMOG2->setNMixtures(5);

    std::vector<cv::Point> roadROI={
        {450,400},{1920,400},{1920,1080},{0,1080},{0,600}
    };
    cv::Mat roadMask=cv::Mat::zeros(frameH,frameW,CV_8UC1);
    std::vector<std::vector<cv::Point>> roiVec={roadROI};
    cv::fillPoly(roadMask,roiVec,cv::Scalar(255));

    // Warmup
    std::cout<<"Warming up";
    cv::Mat wf,wg,wb;
    for(int i=0;i<60;i++){
        cap>>wf; if(wf.empty()) break;
        cv::cvtColor(wf,wg,cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(wg,wb,cv::Size(5,5),0);
        pMOG2->apply(wb,wg,0.01);
        if(i%10==0){std::cout<<".";std::cout.flush();}
    }
    std::cout<<" done.\n";
    cap.set(cv::CAP_PROP_POS_FRAMES,0);

    cv::Mat firstFrame;
    cap>>firstFrame;
    if(firstFrame.empty()){std::cerr<<"Empty.\n";return -1;}
    cap.set(cv::CAP_PROP_POS_FRAMES,0);

    cv::namedWindow("Vehicle Tracker",cv::WINDOW_NORMAL);
    cv::resizeWindow("Vehicle Tracker",1280,720);
    cv::setMouseCallback("Vehicle Tracker",onMouse);
    gCurrentFrame=firstFrame.clone();
    {
        cv::Mat tmp=gCurrentFrame.clone();
        drawOverlay(tmp);
        cv::imshow("Vehicle Tracker",tmp);
    }
    while(!gDraw.done){
        int key=cv::waitKey(30);
        if(key==27) return 0;
        if(key==13&&(int)gDraw.points.size()==2)
            gDraw.done=true;
    }

    bool lineMode=((int)gDraw.points.size()==2);
    std::vector<cv::Point>& imageZone=gDraw.points;

    std::map<int,Vehicle> vehicles;
    int    nextID           = 0;
    float  MAX_DIST         = 150.0f;
    int    MAX_LOST         = 25;
    int    MAX_LOST_IN_ZONE = 60;
    int    MIN_CONFIRMED    = 2;

    // ── Size filters ──────────────────────────────────────────────────────────
    double MIN_AREA = 200.0;
    double MAX_AREA = 300000.0;
    int    MIN_W    = 10;
    int    MIN_H    = 6;
    int    MAX_W    = 800;
    int    MAX_H    = 600;

    const int   FRAMES_TO_ENTER   = 1;
    const int   FRAMES_TO_EXIT    = 2;
    const int   MIN_FRAMES_INSIDE = 8; // must spend this many frames inside
                                       // before exit or lost-inside counts
                                       // prevents counting clips and ghosts
    const float EMA_ALPHA         = 0.6f;

    int entryCount=0;
    int exitCount =0;

    cv::Mat kernelClose=cv::getStructuringElement(
        cv::MORPH_ELLIPSE,cv::Size(17,17));
    cv::Mat kernelOpen=cv::getStructuringElement(
        cv::MORPH_ELLIPSE,cv::Size(5,5));

    cv::Mat frame,gray,blurred,fgMask,maskedFg;

    while(true)
    {
        cap>>frame;
        if(frame.empty()) break;
        gCurrentFrame=frame.clone();

        // Stage A
        cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray,blurred,cv::Size(5,5),0);

        // Stage B
        pMOG2->apply(blurred,fgMask,0.005);
        cv::bitwise_and(fgMask,roadMask,maskedFg);
        cv::morphologyEx(maskedFg,maskedFg,cv::MORPH_CLOSE,kernelClose);
        cv::morphologyEx(maskedFg,maskedFg,cv::MORPH_OPEN, kernelOpen);

        // Stage C — contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(maskedFg,contours,
                         cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);

        std::vector<cv::Rect>                rawBoxes;
        std::vector<std::vector<cv::Point>>  rawContours;

        for(const auto& c:contours){
            double area=cv::contourArea(c);
            if(area<MIN_AREA||area>MAX_AREA) continue;
            cv::Rect bbox=cv::boundingRect(c);
            if(bbox.width <MIN_W||bbox.height<MIN_H) continue;
            if(bbox.width >MAX_W||bbox.height>MAX_H) continue;
            cv::Point2f cen(bbox.x+bbox.width/2.0f,
                            bbox.y+bbox.height/2.0f);
            if(!pointInPolygon(cen,roadROI)) continue;
            rawBoxes.push_back(bbox);
            rawContours.push_back(c);
        }

        std::vector<MergedBlob> blobs=mergeBoxes(rawBoxes,rawContours,15);

        std::vector<cv::Point2f>                         detCentroids;
        std::vector<cv::Rect>                            detBoxes;
        std::vector<std::vector<cv::Point>>              detContours;
        std::vector<std::vector<std::vector<cv::Point>>> detSubContours;

        for(const auto& blob:blobs){
            if(blob.bbox.width <MIN_W||blob.bbox.height<MIN_H) continue;
            if(blob.bbox.width >MAX_W||blob.bbox.height>MAX_H) continue;
            cv::Point2f cen(blob.bbox.x+blob.bbox.width/2.0f,
                            blob.bbox.y+blob.bbox.height/2.0f);
            if(!pointInPolygon(cen,roadROI)) continue;
            detCentroids.push_back(cen);
            detBoxes.push_back(blob.bbox);
            detContours.push_back(blob.contour);
            detSubContours.push_back(blob.subContours);
        }

        // Matching
        std::map<int,bool> matched;
        for(auto& [id,v]:vehicles) matched[id]=false;
        std::vector<bool> detMatched(detCentroids.size(),false);

        for(auto& [id,v]:vehicles){
            int   bestDet  =-1;
            float bestScore=MAX_DIST;
            for(int i=0;i<(int)detCentroids.size();i++){
                if(detMatched[i]) continue;
                float s=matchScore(v,detCentroids[i],detBoxes[i]);
                if(s<bestScore){bestScore=s;bestDet=i;}
            }
            if(bestDet>=0){
                detMatched[bestDet]=true;
                v.prevSmooth      =v.smoothCentroid;
                v.centroid        =detCentroids[bestDet];
                v.smoothCentroid.x=EMA_ALPHA*v.centroid.x
                                  +(1-EMA_ALPHA)*v.smoothCentroid.x;
                v.smoothCentroid.y=EMA_ALPHA*v.centroid.y
                                  +(1-EMA_ALPHA)*v.smoothCentroid.y;
                v.bbox            =detBoxes[bestDet];
                v.contour         =detContours[bestDet];
                v.subContours     =detSubContours[bestDet];
                v.framesLost      =0;
                v.framesConfirmed++;
                if(v.framesConfirmed>=MIN_CONFIRMED) v.confirmed=true;
                matched[id]=true;
            }
        }

        for(int i=0;i<(int)detCentroids.size();i++){
            if(detMatched[i]) continue;
            Vehicle v;
            v.id=nextID++;
            v.centroid=v.smoothCentroid=v.prevSmooth=detCentroids[i];
            v.bbox=detBoxes[i];
            v.contour=detContours[i];
            v.subContours=detSubContours[i];
            v.framesLost=0; v.framesConfirmed=1; v.confirmed=false;
            v.framesInsideTotal=0;
            v.zoneState=Vehicle::ZoneState::OUTSIDE;
            v.zoneFrames=0; v.counted=false;
            vehicles[v.id]=v; matched[v.id]=true;
        }

        // ── Age out ───────────────────────────────────────────────────────────
        for(auto it=vehicles.begin();it!=vehicles.end();){
            int id=it->first;
            if(!matched.count(id)||!matched[id]){
                it->second.framesLost++;
                using ZS=Vehicle::ZoneState;
                auto& v=it->second;

                int allowedLost=(v.zoneState==ZS::INSIDE||
                                 v.zoneState==ZS::ENTERING)
                                ? MAX_LOST_IN_ZONE
                                : MAX_LOST;

                if(v.framesLost>allowedLost){
                    // Only count lost-inside if vehicle spent enough
                    // frames inside — prevents ghost/clip false counts
                    if(v.confirmed&&
                       (v.zoneState==ZS::INSIDE||
                        v.zoneState==ZS::EXITING)&&
                       !v.counted&&
                       v.framesInsideTotal>=MIN_FRAMES_INSIDE)
                    {
                        exitCount++;
                        std::cout<<"ID "<<id
                                 <<" lost inside. Exits:"
                                 <<exitCount<<"\n";
                    }
                    it=vehicles.erase(it);
                } else {
                    // Coast using last known velocity
                    cv::Point2f vel=v.smoothCentroid-v.prevSmooth;
                    v.prevSmooth    =v.smoothCentroid;
                    v.smoothCentroid=v.smoothCentroid+vel*0.5f;
                    ++it;
                }
            } else ++it;
        }

        // ── Zone state machine ────────────────────────────────────────────────
        //
        // OUTSIDE → ENTERING → INSIDE → EXITING → OUTSIDE
        //
        // Entry counted: ENTERING → INSIDE
        // Exit counted:  EXITING  → OUTSIDE  (only if framesInsideTotal >= MIN)
        // Lost counted:  erased while INSIDE  (only if framesInsideTotal >= MIN)
        //
        int inZoneCount=0;

        for(auto& [id,v]:vehicles){
            if(!v.confirmed) continue;

            bool raw;
            if(lineMode)
                raw=crossedLine(v.prevSmooth,v.smoothCentroid,
                                imageZone[0],imageZone[1]);
            else
                raw=pointInPolygon(v.smoothCentroid,imageZone);

            using ZS=Vehicle::ZoneState;
            switch(v.zoneState){
                case ZS::OUTSIDE:
                    if(raw){
                        v.zoneState=ZS::ENTERING;
                        v.zoneFrames=1;
                    }
                    break;

                case ZS::ENTERING:
                    if(raw){
                        v.zoneFrames++;
                        if(v.zoneFrames>=FRAMES_TO_ENTER){
                            v.zoneState=ZS::INSIDE;
                            v.zoneFrames=0;
                            v.framesInsideTotal=0; // reset counter on entry
                            v.counted=false;
                            entryCount++;
                            std::cout<<"ID "<<id<<" entered. Entries:"
                                     <<entryCount<<"\n";
                        }
                    } else {
                        v.zoneState=ZS::OUTSIDE;
                        v.zoneFrames=0;
                    }
                    break;

                case ZS::INSIDE:
                    v.framesInsideTotal++; // accumulate time inside
                    if(!raw){
                        v.zoneState=ZS::EXITING;
                        v.zoneFrames=1;
                    }
                    break;

                case ZS::EXITING:
                    if(!raw){
                        v.zoneFrames++;
                        if(v.zoneFrames>=FRAMES_TO_EXIT){
                            v.zoneState=ZS::OUTSIDE;
                            v.zoneFrames=0;
                            // Only count if vehicle spent enough frames inside
                            // prevents counting a vehicle that just clipped
                            // the zone boundary and immediately left
                            if(!v.counted&&
                               v.framesInsideTotal>=MIN_FRAMES_INSIDE)
                            {
                                exitCount++;
                                v.counted=true;
                                std::cout<<"ID "<<id<<" exited. Exits:"
                                         <<exitCount<<"\n";
                            }
                        }
                    } else {
                        // Flickered back in — still inside
                        v.zoneState=ZS::INSIDE;
                        v.zoneFrames=0;
                    }
                    break;
            }

            bool visuallyIn=(v.zoneState==ZS::INSIDE||
                             v.zoneState==ZS::ENTERING);
            if(visuallyIn) inZoneCount++;

            // Draw vehicle shape in green — only when freshly detected
            // framesLost==0 prevents ghosting of old positions
            if(visuallyIn&&!v.subContours.empty()&&v.framesLost==0){
                cv::Mat carLayer=frame.clone();
                cv::fillPoly(carLayer,v.subContours,cv::Scalar(0,255,0));
                cv::addWeighted(carLayer,0.45,frame,0.55,0,frame);
                cv::polylines(frame,v.subContours,true,
                              cv::Scalar(0,220,0),2);
            }
        }

        // Draw zone
        if(lineMode){
            cv::line(frame,imageZone[0],imageZone[1],
                     cv::Scalar(0,255,255),3);
            cv::circle(frame,imageZone[0],7,cv::Scalar(0,200,255),-1);
            cv::circle(frame,imageZone[1],7,cv::Scalar(0,200,255),-1);
        } else {
            cv::Mat zoneLayer=frame.clone();
            std::vector<std::vector<cv::Point>> zoneVec={imageZone};
            cv::fillPoly(zoneLayer,zoneVec,cv::Scalar(0,220,220));
            cv::addWeighted(zoneLayer,0.20,frame,0.80,0,frame);
            cv::polylines(frame,zoneVec,true,cv::Scalar(0,255,255),2);
        }

        // HUD
        int hudH=lineMode?70:100;
        cv::Mat hud=frame.clone();
        cv::rectangle(hud,cv::Rect(10,10,250,hudH),
                      cv::Scalar(0,0,0),-1);
        cv::addWeighted(hud,0.45,frame,0.55,0,frame);

        cv::putText(frame,"In Zone : "+std::to_string(inZoneCount),
                    cv::Point(20,40),cv::FONT_HERSHEY_SIMPLEX,0.65,
                    cv::Scalar(0,255,0),2);
        if(!lineMode){
            cv::putText(frame,"Entered : "+std::to_string(entryCount),
                        cv::Point(20,68),cv::FONT_HERSHEY_SIMPLEX,0.65,
                        cv::Scalar(0,200,255),2);
            cv::putText(frame,"Exited  : "+std::to_string(exitCount),
                        cv::Point(20,96),cv::FONT_HERSHEY_SIMPLEX,0.65,
                        cv::Scalar(0,255,255),2);
        } else {
            cv::putText(frame,"Crossed : "+std::to_string(exitCount),
                        cv::Point(20,68),cv::FONT_HERSHEY_SIMPLEX,0.65,
                        cv::Scalar(0,255,255),2);
        }

        cv::imshow("Vehicle Tracker",frame);
        if(cv::waitKey(30)==27) break;
    }
    return 0;
}