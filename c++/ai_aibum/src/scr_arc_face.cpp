

#include "queue"
#include "opencv2/opencv.hpp"
#include <functional>
#include <future>
#include <thread>
#include "../include/scr_arc_face.h"


int MAX_QUEUE_SIZE = 100; // 队列的最大大小
int NUM_PRODUCER_THREADS = 4; // 生产者线程数量
int NUM_CONSUMER_THREADS = 1; // 消费者线程数量
// 图像队列和同步变量
std::queue<imageQueueObject> imageQueue;
std::mutex queueMutex;
std::condition_variable conditionVar;

std::vector<faceObject> results; // 用于存储消费者结果

bool done = false; // 表示生产是否完成
void producer_worker(const std::vector<std::string>& imagePaths, int start, int end)
{
    for (int i = start; i < end; ++i)
    {
        const auto& imagePath = imagePaths[i];
        const int max_side = 320;
        cv::Mat image = cv::imread(imagePath);
        float long_side = std::max(image.cols, image.rows);
        float scale = max_side / long_side;
        cv::Mat img_scale;
        if (image.cols <= max_side || image.rows <= max_side)
        {
            img_scale = image;
        }
        else
        {
            cv::resize(image, img_scale, cv::Size(image.cols * scale, image.rows * scale));
        }

        if (img_scale.empty())
        {
            std::cerr << "Failed to load image: " << imagePath << std::endl;
            continue;
        }

        std::unique_lock<std::mutex> lock(queueMutex);
        conditionVar.wait(lock, [] { return imageQueue.size() < MAX_QUEUE_SIZE; });
        imageQueueObject imgObj;
        imgObj.imagePath = imagePath;
        imgObj.image = img_scale;
        imageQueue.push(imgObj);
        // imageQueue.push(img_scale);
        // imageQueue.push(std::make_pair(imagePath, img_scale));
        // std::cout << "imageQueue:=>" << imageQueue.size() << ",imread:=> " << img_scale.size() << std::endl;
        lock.unlock();
        // if (imageQueue.size() == MAX_QUEUE_SIZE)
        // {
        conditionVar.notify_all(); // 通知消费者有新的图像
        // }
    }
    // conditionVar.notify_all(); // 通知消费者有新的图像
}


void producer(const std::vector<std::string>& imagePaths)
{
    int numImages = imagePaths.size();
    int imagesPerThread = (numImages + NUM_PRODUCER_THREADS - 1) / NUM_PRODUCER_THREADS;

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_PRODUCER_THREADS; ++i)
    {
        int start = i * imagesPerThread;
        int end = std::min(start + imagesPerThread, numImages);
        if (start < end)
        {
            threads.emplace_back(producer_worker, std::ref(imagePaths), start, end);
        }
    }

    for (auto& thread : threads)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }
    done = true;
    conditionVar.notify_all();
}

void consumer_worker(Detect* detect, Recognition* recognition, const std::vector<float>& feature_face)
{
    while (true)
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        conditionVar.wait(lock, [] { return !imageQueue.empty() || done; });

        if (imageQueue.empty() && done)
        {
            break;
        }

        auto [imagePath, image] = imageQueue.front();
        imageQueue.pop();
        lock.unlock();
        conditionVar.notify_all();

        std::vector<FaceObject> faceobjects;
        detect->detect_scrfd(image, faceobjects);
        float max_similarity = -1;
        if (faceobjects.size())
        {
#pragma omp parallel for
            for (int i = 0; i < faceobjects.size(); i++)
            {
                std::vector<float> feature2_rec;
                recognition->arcface_get_feature(image, faceobjects[i], feature2_rec);
                if (feature2_rec.empty()) continue;
                float similarity = recognition->calculate_similarity(feature_face, feature2_rec);
                if (similarity > max_similarity) max_similarity = similarity;
            }

            if (!faceobjects.empty())
            {
                faceObject faceobject{imagePath, max_similarity};
                results.push_back(faceobject);
                std::cout << "Consumer thread " << std::this_thread::get_id()
                    << " ,finished processing image: " << imagePath
                    << " ,similarity:" << faceobject.similarity << std::endl;
            }
        }
    }
}

void consumer(Detect* detect,
              Recognition* recognition,
              std::vector<float> feature_face)
{
    std::vector<std::future<void>> futures;
    for (int i = 0; i < NUM_CONSUMER_THREADS; ++i)
    {
        futures.push_back(std::async(std::launch::async, consumer_worker, detect, recognition,
                                     std::cref(feature_face)));
    }
    for (auto& future : futures)
    {
        future.get();
    }
}


extern "C++" {
std::vector<float> get_feature(Detect* detect,
                               Recognition* recognition,
                               const std::string& image_path
)
{
    cv::Mat image = cv::imread(image_path);
    cv::Mat image_face = resize_image(image);
    //获取人脸底照向量
    std::vector<FaceObject> faceobjects_face;
    detect->detect_scrfd(image_face, faceobjects_face);
    //获取人脸向量
    std::vector<float> feature;
    if (faceobjects_face.size() == 1)
    {
        recognition->arcface_get_feature(image_face, faceobjects_face[0], feature);
    }
    return feature;
}

//人脸批量检测识别
std::vector<faceObject> face_detection_consumer_producer(
    Detect* detect,
    Recognition* recognition,
    std::vector<float> feature_face,
    const std::vector<std::string>& imagePaths,
    int queueLength,
    int producerThreadNum,
    int consumerThreadNum
)
{
    results.clear();
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        while (!imageQueue.empty())
        {
            imageQueue.pop();
        }
    }
    done = false;
    MAX_QUEUE_SIZE = queueLength;
    NUM_PRODUCER_THREADS = producerThreadNum;
    NUM_CONSUMER_THREADS = consumerThreadNum;
    std::thread producerThread(producer, imagePaths);
    std::thread consumerThread(consumer, detect, recognition, std::ref(feature_face));
    producerThread.join();
    consumerThread.join();
    return results;
}

//人脸检测初始化
int detect_init(Detect* detect,
                const std::string& param_model_path,
                const std::string& bin_model_path)
{
    return detect->detect_init(param_model_path, bin_model_path);
}

//人脸识别初始化
int recognition_init(Recognition* recognition,
                     const std::string& param_model_path,
                     const std::string& bin_model_path)
{
    // return recognition->identify_init("../face_de/model/mobilefacenet.param",
    //                                   "../face_de/model/mobilefacenet.bin");
    return recognition->recognition_init(param_model_path, bin_model_path);
}


//人脸检测返回人脸数量
int detcet_scrfd(Detect* detect, cv::Mat& image, const int max_side)
{
    cv::Mat img_scale = resize_image(image, max_side);
    std::vector<FaceObject> faceobjects;
    detect->detect_scrfd(image, faceobjects);
    return faceobjects.size();
}

//人脸检测返回人脸区域
std::vector<cv::Mat> detcet_scrfd_mat(Detect* detect, cv::Mat& image, int max_side)
{
    float long_side = std::max(image.cols, image.rows);
    float scale = max_side / long_side;
    cv::Mat img_scale = resize_image(image, max_side);
    std::vector<FaceObject> faceobjects;
    detect->detect_scrfd(image, faceobjects);
    std::vector<cv::Mat> faceMats;
    for (int i = 0; i < faceobjects.size(); i++)
    {
        cv::Rect face_rect = faceobjects[i].rect;
        cv::Mat face = image(face_rect);
        faceMats.push_back(face);
    }
    return faceMats;
}

//人脸检测返回人脸faceobjects(关键点)区域
std::vector<FaceObject> detcet_scrfd_faceobjects(Detect* detect, cv::Mat& image, int max_side)
{
    cv::Mat img_scale = resize_image(image, max_side);
    std::vector<FaceObject> faceobjects;
    detect->detect_scrfd(image, faceobjects);
    return faceobjects;
}

cv::Mat resize_image(const cv::Mat& feature, int max_side)
{
    cv::Mat feature_cvt;
    cv::cvtColor(feature, feature_cvt, cv::COLOR_RGBA2BGR);
    float long_side = std::max(feature_cvt.cols, feature_cvt.rows);
    float scale = max_side / long_side;
    cv::Mat img_scale;
    if (feature_cvt.cols > max_side || feature_cvt.rows > max_side)
    {
        cv::resize(feature_cvt,
                   img_scale,
                   cv::Size(static_cast<int>(feature_cvt.cols * scale),
                            static_cast<int>(feature_cvt.rows * scale)
                   )
        );
    }
    else
    {
        img_scale = feature_cvt;
    }
    return img_scale;
}

//人脸特征提取
std::vector<float> arcface_get_feature(Recognition* recognition,
                                       cv::Mat& detect_image,
                                       FaceObject& faceobject)
{
    std::vector<float> feature_rec;
    recognition->arcface_get_feature(detect_image, faceobject, feature_rec);
    return feature_rec;
}

//特征比对
float calculate(Recognition* recognition,
                std::vector<float>& feature1,
                std::vector<float>& feature2)
{
    float similarity = recognition->calculate_similarity(feature1, feature2);
    return similarity;
}

//返回人脸相似度值
float calculate_similarity(
    Detect* detect,
    Recognition* recognition,
    const cv::Mat& feature1,
    const cv::Mat& feature2)
{
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat detect_image1 = resize_image(feature1);
    cv::Mat detect_image2 = resize_image(feature2);

    std::vector<FaceObject> faceobjects1;
    detect->detect_scrfd(detect_image1, faceobjects1);
    std::vector<FaceObject> faceobjects2;
    detect->detect_scrfd(detect_image2, faceobjects2);

    if (faceobjects1.empty() || faceobjects2.empty()) return -1;
    std::vector<float> feature1_rec;
    recognition->arcface_get_feature(detect_image1, faceobjects1[0], feature1_rec);
    if (feature1_rec.empty()) return -1;
    float max_similarity = -1;
#pragma omp parallel for
    for (int i = 0; i < faceobjects2.size(); i++)
    // for (const auto& faceobject2 : faceobjects2)
    {
        std::vector<float> feature2_rec;
        recognition->arcface_get_feature(detect_image2, faceobjects2[i], feature2_rec);
        if (feature2_rec.empty()) continue;
        float similarity = recognition->calculate_similarity(feature1_rec, feature2_rec);
        if (similarity > max_similarity) max_similarity = similarity;
    }
    auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsedSeconds = end - start;
    // auto elapsedMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(elapsedSeconds);
    // std::cout << "calculate_similarity: " << elapsedMilliseconds.count() << " seconds," << std::endl;
    return max_similarity;
}
}

