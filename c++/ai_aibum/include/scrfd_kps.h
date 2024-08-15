
#include "net.h"
#include <opencv2/core/core.hpp>
#ifndef SCRFD_KPS_H
#define SCRFD_KPS_H

struct FaceObject
{
    cv::Rect_<float> rect;
    cv::Point2f landmark[5];
    float prob;
};


class Detect
{
public:
    ncnn::Net scrfd;
    int target_size = 640;
    float prob_threshold = 0.6f;
    float nms_threshold = 0.7f;

public:
    Detect()
    {
    };
    Detect(int target_size);
    Detect(int target_size,
           float prob_threshold,
           float nms_threshold);
    Detect::~Detect();
    static inline float intersection_area(const FaceObject& a, const FaceObject& b);
    void qsort_descent_inplace(std::vector<FaceObject>& faceobjects, int left, int right);
    void qsort_descent_inplace(std::vector<FaceObject>& faceobjects);
    void nms_sorted_bboxes(const std::vector<FaceObject>& faceobjects, std::vector<int>& picked,
                           float nms_threshold);
    ncnn::Mat generate_anchors(int base_size, const ncnn::Mat& ratios, const ncnn::Mat& scales);
    void generate_proposals(const ncnn::Mat& anchors, int feat_stride, const ncnn::Mat& score_blob,
                            const ncnn::Mat& bbox_blob, const ncnn::Mat& landmark_blob,
                            float prob_threshold,
                            std::vector<FaceObject>& faceobjects);
    void extract_and_generate_proposals(ncnn::Extractor& ex, const std::string& score_layer,
                                        const std::string& bbox_layer, const std::string& landmark_layer, int base_size,
                                        int feat_stride, float prob_threshold, std::vector<FaceObject>& faceproposals);
    int detect_init(const std::string& param_model_path, const std::string& bin_model_path);
    //人脸检测
    int detect_scrfd(const cv::Mat& bgr, std::vector<FaceObject>& faceobjects);
};

#endif //SCRFD_KPS_H
