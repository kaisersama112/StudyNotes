//
// Created by Administrator on 27/7/2024.
//
#include "scrfd_kps.h"

#ifndef ARCFACE_H
#define ARCFACE_H


class Recognition
{
public:
    ncnn::Net arcface;
    ncnn::Net Identify;

public:
    Recognition()
    {
    };
    ~Recognition();
    int recognition_init(const std::string& param_model_path, const std::string& bin_model_path);
    int identify_init(const std::string& param_model_path, const std::string& bin_model_path);
    inline void normalize(std::vector<float>& feature);
    cv::Mat similarTransform(cv::Mat src, cv::Mat dst);
    void arcface_forward(const cv::Mat& aligned, std::vector<float>& feature);
    cv::Mat similarity_transform(float src[5][2], float dst[5][2]);
    void arcface_alignment(const cv::Mat& bgr, const FaceObject& face, cv::Mat& aligned);
    //人脸特征提取
    void arcface_get_feature(const cv::Mat& bgr,
                             const FaceObject& face,
                             std::vector<float>& feature);

    //人脸比对
    float calculate_similarity(const std::vector<float>& feature1,
                               const std::vector<float>& feature2);
    // 相似度比对
    double calculSimilar(std::vector<float>& feature1, std::vector<float>& feature2);
};


#endif //ARCFACE_H
