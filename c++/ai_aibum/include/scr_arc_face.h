//
// Created by Administrator on 27/7/2024.
//


#include "arcface.h"
#include "scrfd_kps.h"


struct faceObject
{
    // 图片路径
    std::string image_path;
    // 相似度值
    float similarity;
};

struct imageQueueObject
{
    std::string imagePath;
    cv::Mat image;
};

extern "C++" {
std::vector<float> get_feature(Detect* detect,
                               Recognition* recognition,
                               const cv::String& image_path
);

std::vector<faceObject> face_detection_consumer_producer(
    Detect* detect,
    Recognition* recognition,
    std::vector<float> feature_face,
    const std::vector<std::string>& imagePaths,
    int queueLength,
    int producerThreadNum,
    int consumerThreadNum
);
/*
    检测模型初始化
    const std::string& param_model_path params路径
    const std::string& bin_model_path bin路径
*/
int detect_init(Detect* detect, const std::string& param_model_path, const std::string& bin_model_path);
/*
    识别模型初始化
    const std::string& param_model_path params路径
    const std::string& bin_model_path bin路径
 */
int recognition_init(Recognition* recognition, const std::string& param_model_path,
                     const std::string& bin_model_path);

//重设图像大小
cv::Mat resize_image(const cv::Mat& feature, int max_side = 640);
//人脸检测返回人脸数量
int detcet_scrfd(Detect* detect, cv::Mat& image, int max_side = 640);
//人脸检测返回人脸区域
std::vector<cv::Mat> detcet_scrfd_mat(Detect* detect, cv::Mat& image, int max_side = 640);

//人脸检测返回人脸faceobjects(关键点)
std::vector<FaceObject> detcet_scrfd_faceobjects(Detect* detect, cv::Mat& image, int max_side = 640);

//人脸特征提取
std::vector<float> arcface_get_feature(Recognition* recognition,
                                       cv::Mat& detect_image,
                                       FaceObject& faceobject);
//特征比对
float calculate(Recognition* recognition, std::vector<float>& feature1, std::vector<float>& feature2);
//人脸特征提取比对
float calculate_similarity(Detect* detect,
                           Recognition* recognition, const cv::Mat& feature1, const cv::Mat& feature2);

float calculate_similarity1(Detect* detect,
                            Recognition* recognition,
                            const cv::Mat& feature1,
                            const cv::Mat& feature2);
// 返回人脸相似度值/批量图片
std::vector<faceObject> calculate_similarity_batch(Detect* detect,
                                                   Recognition* recognition,
                                                   const std::vector<std::string>& imageList,
                                                   const std::string& botFeature
);
}




