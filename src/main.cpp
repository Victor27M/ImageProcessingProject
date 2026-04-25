#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
// Data Structures
// ─────────────────────────────────────────────────────────────────────────────

struct Vehicle {
    int         id;
    cv::Point2f centroid;
    cv::Rect    bbox;
    bool        inZone;
    int         framesLost;
};

// ─────────────────────────────────────────────────────────────────────────────
// Stage C Helper — Ray Casting Algorithm
// Cast a horizontal ray from P and count polygon edge crossings.
// Odd  => inside   |   Even => outside
// ─────────────────────────────────────────────────────────────────────────────

bool pointInPolygon(const cv::Point2f& p,
                    const std::vector<cv::Point>& polygon)
{
    int crossings = 0;
    int n = (int)polygon.size();
    for (int i = 0; i < n; i++) {
        cv::Point a = polygon[i];
        cv::Point b = polygon[(i + 1) % n];
        if (((a.y <= p.y) && (b.y > p.y)) ||
            ((b.y <= p.y) && (a.y > p.y)))
        {
            float t = (p.y - a.y) / (float)(b.y - a.y);
            if (p.x < a.x + t * (b.x - a.x))
                crossings++;
        }
    }
    return (crossings % 2) == 1;
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage C Helper — Euclidean Distance
// d = sqrt((x2-x1)^2 + (y2-y1)^2)
// ─────────────────────────────────────────────────────────────────────────────

float euclidean(const cv::Point2f& a, const cv::Point2f& b)
{
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main()
{
    cv::VideoCapture cap("video.mp4"); // replace with 0 for webcam
    if (!cap.isOpened()) {
        std::cerr << "Error: cannot open video source.\n";
        return -1;
    }

    // ── Stage 1: MOG2 Initialisation ─────────────────────────────────────────
    // history=500 frames for robust initial background model
    cv::Ptr<cv::BackgroundSubtractorMOG2> pMOG2 =
        cv::createBackgroundSubtractorMOG2(500, 16, true);

    // ── Betting Zone polygon (adjust to your video frame) ────────────────────
    std::vector<cv::Point> bettingZone = {
        {200, 150}, {600, 150}, {600, 450}, {200, 450}
    };

    // ── Tracking parameters ───────────────────────────────────────────────────
    std::map<int, Vehicle> vehicles;
    int   nextID         = 0;
    float MAX_DIST       = 80.0f;  // max px distance for ID association
    double MIN_AREA      = 500.0;  // min contour area — filters pedestrians/debris
    int   MAX_LOST       = 10;     // frames before a lost vehicle is removed

    cv::Mat frame, gray, blurred, fgMask;

    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;

        // ── Stage A: Pre-processing & Noise Mitigation ────────────────────────

        // Grayscale — reduces computation 3x, preserves structural edges
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Gaussian blur — 5x5 kernel suppresses high-frequency sensor noise
        // prevents "false motion" in subtraction
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

        // ── Stage B: Motion-Based Segmentation ───────────────────────────────

        // MOG2 — pixel labelled foreground if |I(x,y) - B(x,y)| > threshold
        // background updated adaptively: B(t+1) = a*I(t) + (1-a)*B(t)
        pMOG2->apply(blurred, fgMask);

        // Morphological refinement
        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(5, 5));

        // Closing = Dilation after Erosion → fills interior gaps
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, kernel);
        // Opening = Erosion after Dilation → removes isolated noise clusters
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, kernel);

        // ── Stage C: Contour Extraction & Filtering ───────────────────────────

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(fgMask, contours,
                         cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<cv::Point2f> detectedCentroids;
        std::vector<cv::Rect>    detectedBoxes;

        for (const auto& contour : contours)
        {
            // cv2.contourArea — rejects blobs below pixel-area threshold
            if (cv::contourArea(contour) < MIN_AREA) continue;

            cv::Rect bbox = cv::boundingRect(contour);
            cv::Point2f centroid(
                bbox.x + bbox.width  / 2.0f,
                bbox.y + bbox.height / 2.0f);

            detectedCentroids.push_back(centroid);
            detectedBoxes.push_back(bbox);
        }

        // ── Stage C: Centroid Association (Euclidean ID matching) ─────────────

        std::map<int, bool> matched;
        for (auto& [id, v] : vehicles) matched[id] = false;

        for (int i = 0; i < (int)detectedCentroids.size(); i++)
        {
            int   bestID   = -1;
            float bestDist = MAX_DIST;

            for (auto& [id, v] : vehicles)
            {
                float d = euclidean(detectedCentroids[i], v.centroid);
                if (d < bestDist) {
                    bestDist = d;
                    bestID   = id;
                }
            }

            if (bestID >= 0) {
                // Update existing vehicle
                vehicles[bestID].centroid   = detectedCentroids[i];
                vehicles[bestID].bbox       = detectedBoxes[i];
                vehicles[bestID].framesLost = 0;
                matched[bestID]             = true;
            } else {
                // Register new vehicle with stable ID
                Vehicle v;
                v.id         = nextID++;
                v.centroid   = detectedCentroids[i];
                v.bbox       = detectedBoxes[i];
                v.inZone     = false;
                v.framesLost = 0;
                vehicles[v.id]  = v;
                matched[v.id]   = true;
            }
        }

        // Age out vehicles not seen for MAX_LOST frames
        for (auto it = vehicles.begin(); it != vehicles.end(); )
        {
            int id = it->first;
            if (!matched.count(id) || !matched[id]) {
                it->second.framesLost++;
                if (it->second.framesLost > MAX_LOST)
                    it = vehicles.erase(it);
                else
                    ++it;
            } else {
                ++it;
            }
        }

        // ── Stage C: Ray Casting Zone Logic ──────────────────────────────────

        int vehiclesInZone = 0;
        for (auto& [id, v] : vehicles)
        {
            v.inZone = pointInPolygon(v.centroid, bettingZone);
            if (v.inZone) vehiclesInZone++;
        }

        // ── Odds Calculation: Odds = 1 / Probability ─────────────────────────

        int    total       = std::max((int)vehicles.size(), 1);
        double probability = (double)vehiclesInZone / total;
        double odds        = (probability > 0.0) ? 1.0 / probability : 99.99;

        // ── Draw betting zone ─────────────────────────────────────────────────

        std::vector<std::vector<cv::Point>> zoneVec = {bettingZone};
        cv::polylines(frame, zoneVec, true, cv::Scalar(0, 255, 255), 2);

        // ── Draw vehicles ─────────────────────────────────────────────────────

        for (auto& [id, v] : vehicles)
        {
            // Green = outside zone, Red = inside zone
            cv::Scalar color = v.inZone
                ? cv::Scalar(0, 0, 255)
                : cv::Scalar(0, 255, 0);

            cv::rectangle(frame, v.bbox, color, 2);
            cv::circle(frame, v.centroid, 4, cv::Scalar(255, 0, 0), -1);
            cv::putText(frame,
                        "ID:" + std::to_string(v.id),
                        cv::Point(v.bbox.x, v.bbox.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        }

        // ── HUD Overlay (semi-transparent) ────────────────────────────────────
        // cv2.addWeighted blends the dark panel at 40% opacity

        cv::Mat overlay = frame.clone();
        cv::rectangle(overlay,
                      cv::Rect(10, 10, 270, 110),
                      cv::Scalar(0, 0, 0), -1);
        cv::addWeighted(overlay, 0.4, frame, 0.6, 0, frame);

        cv::putText(frame,
                    "Vehicles : " + std::to_string(vehicles.size()),
                    cv::Point(20, 38),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65,
                    cv::Scalar(255, 255, 255), 2);

        cv::putText(frame,
                    "In Zone  : " + std::to_string(vehiclesInZone),
                    cv::Point(20, 65),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65,
                    cv::Scalar(0, 255, 255), 2);

        char oddsStr[32];
        std::snprintf(oddsStr, sizeof(oddsStr), "Odds     : %.2f", odds);
        cv::putText(frame, oddsStr,
                    cv::Point(20, 92),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65,
                    cv::Scalar(0, 200, 255), 2);

        cv::imshow("Vehicle Tracker", frame);
        cv::imshow("FG Mask",         fgMask);

        if (cv::waitKey(30) == 27) break; // ESC to quit
    }

    return 0;
}