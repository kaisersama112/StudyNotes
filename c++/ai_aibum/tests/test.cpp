//
// Created by Administrator on 27/7/2024.
//


#include "iostream"
#include "face_de/include/scr_arc_face.h"
#include "face_de/include/scrfd_kps.h"
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
namespace fs = std::filesystem;

std::vector<std::string> get_image_paths(const std::string& folderPath)
{
    std::vector<std::string> imagePaths;
    for (const auto& entry : fs::directory_iterator(folderPath))
    {
        if (entry.is_regular_file() &&
            (
                entry.path().extension() == ".jpg" ||
                entry.path().extension() == ".png" ||
                entry.path().extension() == ".JPG" ||
                entry.path().extension() == ".PNG"
                // entry.path().extension() == ".heic" ||
                // entry.path().extension() == ".HEIC"
            ))
        {
            imagePaths.push_back(entry.path().string());
        }
    }
    return imagePaths;
}

// 生成唯一文件名的辅助函数
std::string generate_unique_filename(const std::string& baseDir, const std::string& prefix)
{
    static int counter = 0;
    std::ostringstream filename;
    filename << baseDir << "/" << prefix << "_" << std::setw(5) << std::setfill('0') << counter++ << ".jpg";
    return filename.str();
}


void test_producer_consumer()
{
    Detect* detect = new Detect;
    Recognition* recognition = new Recognition;
    detect_init(detect,
                "../face_de/model/scrfd_500m_kps.param",
                "../face_de/model/scrfd_500m_kps.bin");
    recognition_init(recognition,
                     "../face_de/model/w600k_mbf.param",
                     "../face_de/model/w600k_mbf.bin");
    auto start = std::chrono::high_resolution_clock::now();
    std::string folderPath = "F:\\DataSet\\baby";
    // std::string folderPath = "F:\\image1";
    std::string bottomShot = "../face_res/img.png";
    float similarity = 0.6f;
    std::vector<std::string> imagePaths = get_image_paths(folderPath);
    cv::Mat image1 = cv::imread(bottomShot);
    int numbers = 0;
#pragma omp parallel for
    for (int i = 0; i < imagePaths.size(); ++i)
    {
        cv::Mat image2 = cv::imread(imagePaths[i]);

        float max_similarity = calculate_similarity(detect, recognition, image1, image2);
        // std::cout << "results:" << max_similarity << std::endl;
        if (max_similarity > 0.57)
        {
            numbers++;
            std::cout << "max_similarity:" << max_similarity << ",results:" << imagePaths[i] << std::endl;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedSeconds = end - start;
    std::cout << "Total time taken: " << elapsedSeconds.count() << " seconds,"
        << "numbers" << numbers << std::endl;
    delete detect;
    delete recognition;
}

void test_producer_consumer1()
{
    Detect* detect = new Detect;
    Recognition* recognition = new Recognition;
    detect_init(detect,
                "../face_de/model/scrfd_500m_kps.param",
                "../face_de/model/scrfd_500m_kps.bin");
    recognition_init(recognition,
                     "../face_de/model/w600k_mbf.param",
                     "../face_de/model/w600k_mbf.bin");
    auto start = std::chrono::high_resolution_clock::now();
    std::string folderPath = "F:\\DataSet\\baby";
    // std::string folderPath = "F:\\image1";
    std::string bottomShot = "../face_res/img.png";
    cv::Mat image = cv::imread(bottomShot);
    cv::Mat image_face = resize_image(image);
    std::vector<FaceObject> faceobjects;
    detect->detect_scrfd(image_face, faceobjects);
    std::vector<float> feature_face;
    if (faceobjects.size())
    {
        recognition->arcface_get_feature(image_face, faceobjects[0], feature_face);
    }

    std::vector<std::string> imagePaths = get_image_paths(folderPath);
    cv::Mat image1 = cv::imread(bottomShot);
    std::vector<faceObject> faceObj = face_detection_consumer_producer(
        detect,
        recognition,
        feature_face,
        imagePaths,
        100,
        8,
        4);

    delete detect;
    delete recognition;
}

int main()
{
    /*
    const char* imagepath = "../face_res/B202302026025pz_170010750894.jpg";

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<FaceObject> faceobjects;
    detect_scrfd(m, faceobjects);
    std::cout << "faceobjects:" << faceobjects.size() << std::endl;
    draw_faceobjects(m, faceobjects);
    */
    auto start = std::chrono::high_resolution_clock::now();
    test_producer_consumer1();
    // test_loadHeic();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "test_loadHeic Time taken to load HEIC image: " << duration.count() << " seconds" << std::endl;

    return 0;
}

