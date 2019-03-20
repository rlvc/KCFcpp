#include <iostream>
#include "img_series_reader.h"
#include "linemod_if.h"
#include "opencv2/opencv.hpp"
#include "lotus_common.h"
#include "my_timer.h"
#include "BoxExtractor.h"
using namespace std;
using namespace cv;



void linemod_recon(const string &strConfigFile)
{
    VideoWriter videowriter;
    videowriter.open("./test.avi", CV_FOURCC('D', 'I', 'V', 'X'), 25.0, Size(640, 480));

    // Create KCFTracker object and ROI selector
    bool HOG = true;
    bool FIXEDWINDOW = false;
    bool MULTISCALE = true;
    bool SILENT = true;
    bool LAB = false;
    KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
    BoxExtractor box;

    Timer match_timer;
    float matching_threshold = 80.0f;
    cv::Ptr<cv::linemod::Detector> detector = readLinemod(strConfigFile + string("/linemod_templates.yml"));
    std::vector<String> ids = detector->classIds();
    int num_classes = detector->numClasses();
    printf("Loaded %s with %d classes and %d templates\n",
        strConfigFile.c_str(), num_classes, detector->numTemplates());
    int num_modalities = (int)detector->getModalities().size();
    printf("num_modalities = %d \n", num_modalities);


    CImgSeriesReader reader;
    if (!reader.Init(CImgSeriesReader::ESrcType(1), "1"))
    {
        cout << "initial image reader failed!" << endl;
        return;
    }
    cv::Mat color;
    
    bool b_init_kcf = true;
    Rect kcf_roi;
    Mat current_template;
    while (reader.GetNextImage(color))
    {
        if (b_init_kcf)
        {
            Rect2d roi = box.extract("tracker", color); 
            if (roi.width == 0 || roi.height == 0) return;
            tracker.init(roi, color);
            //rectangle(color, roi, Scalar(0, 255, 255), 1, 8);
            b_init_kcf = false;
            continue;
        }
        kcf_roi = tracker.update(color);
        Mat display = color.clone();
        std::vector<cv::Mat> sources;
        std::vector<cv::Mat> masks;
        cv::Mat mask = Mat::zeros(color.size(), CV_8UC1);
        mask(kcf_roi).setTo(255);
        masks.push_back(mask);
        sources.push_back(color);

        std::vector<cv::linemod::Match> matches;
        std::vector<String> class_ids;
        std::vector<cv::Mat> quantized_images;
        match_timer.start();
        detector->match(sources, matching_threshold, matches, class_ids, quantized_images, masks);

        int classes_visited = 0;
        std::set<std::string> visited;

        for (int i = 0; (i < (int)matches.size()) && (classes_visited < num_classes); ++i)
        {
            cv::linemod::Match m = matches[i];

            if (visited.insert(m.class_id).second)
            {
                ++classes_visited;
                printf("matches.size()%d\n", matches.size());
                printf("Similarity: %5.1f%%; x: %3d; y: %3d; class: %s; template: %3d\n",
                    m.similarity, m.x, m.y, m.class_id.c_str(), m.template_id);
                string current_template_path = strConfigFile + string("/gray/") + to_string(m.template_id) + string(".png");
                current_template = imread(current_template_path);
                // Draw matching template 
                const std::vector<cv::linemod::Template>& templates = detector->getTemplates(m.class_id, m.template_id);
                drawResponse(templates, num_modalities, display, cv::Point(m.x, m.y), detector->getT(0), current_template);
            }
        }
        match_timer.stop();
        printf("Matching: %.2fs\n", match_timer.time());
        rectangle(display, Point(kcf_roi.x, kcf_roi.y), Point(kcf_roi.x + kcf_roi.width, kcf_roi.y + kcf_roi.height), Scalar(0, 255, 255), 1, 8);
        cv::imshow("color", display);
        videowriter << display;
        waitKey(1);
    }
    //videowriter.release();
    //cv::Mat color = imread(strConfigFile + string("/gray/3330.png"));
    
    system("pause");
}