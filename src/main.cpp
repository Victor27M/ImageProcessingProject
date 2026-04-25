#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap("video.mp4"); // or 0 for webcam
    if (!cap.isOpened()) {
        std::cerr << "Error: cannot open video source\n";
        return -1;
    }

    // Stage 1 — MOG2 Background Subtractor
    cv::Ptr<cv::BackgroundSubtractorMOG2> pMOG2 =
        cv::createBackgroundSubtractorMOG2(500, 16, true);

    cv::Mat frame, gray, blurred, fgMask;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Stage A — Pre-processing
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

        // Stage B — Segmentation
        pMOG2->apply(blurred, fgMask);

        // Stage B — Morphological refinement
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, kernel);

        // Stage C — Contour extraction
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(fgMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area < 500) continue; // filter out pedestrians/noise

            cv::Rect bbox = cv::boundingRect(contour);
            cv::Point centroid(bbox.x + bbox.width / 2,
                               bbox.y + bbox.height / 2);

            cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);
            cv::circle(frame, centroid, 4, cv::Scalar(0, 0, 255), -1);
        }

        // HUD overlay placeholder
        cv::Mat overlay = frame.clone();
        cv::rectangle(overlay, cv::Rect(10, 10, 200, 50),
                      cv::Scalar(0, 0, 0), -1);
        cv::addWeighted(overlay, 0.4, frame, 0.6, 0, frame);
        cv::putText(frame, "Vehicles: " + std::to_string(contours.size()),
                    cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX,
                    0.7, cv::Scalar(255, 255, 255), 2);

        cv::imshow("Vehicle Tracker", frame);
        cv::imshow("FG Mask", fgMask);

        if (cv::waitKey(30) == 27) break; // ESC to quit
    }

    return 0;
}