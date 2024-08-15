#include "net.h"

#include <opencv2/imgproc.hpp>
#include "../include/arcface.h"

#include <opencv2/highgui.hpp>

static cv::Mat meanAxis0(const cv::Mat& src)
{
    int num = src.rows;
    int dim = src.cols;

    // x1 y1
    // x2 y2

    cv::Mat output(1, dim, CV_32F);
    for (int i = 0; i < dim; i++)
    {
        float sum = 0;
        for (int j = 0; j < num; j++)
        {
            sum += src.at<float>(j, i);
        }
        output.at<float>(0, i) = sum / num;
    }
    return output;
}

static cv::Mat elementwiseMinus(const cv::Mat& A, const cv::Mat& B)
{
    cv::Mat output(A.rows, A.cols, A.type());

    assert(B.cols == A.cols);
    if (B.cols == A.cols)
    {
        for (int i = 0; i < A.rows; i++)
        {
            for (int j = 0; j < B.cols; j++)
            {
                output.at<float>(i, j) = A.at<float>(i, j) - B.at<float>(0, j);
            }
        }
    }
    return output;
}

static cv::Mat varAxis0(const cv::Mat& src)
{
    cv::Mat temp_ = elementwiseMinus(src, meanAxis0(src));
    cv::multiply(temp_, temp_, temp_);
    return meanAxis0(temp_);
}

static int MatrixRank(cv::Mat M)
{
    cv::Mat w, u, vt;
    cv::SVD::compute(M, w, u, vt);
    cv::Mat1b nonZeroSingularValues = w > 0.0001;
    int rank = countNonZero(nonZeroSingularValues);
    return rank;
}

int Recognition::recognition_init(const std::string& param_model_path,
                                  const std::string& bin_model_path)
{
    arcface.clear();
#if NCNN_VULKAN
    arcface.opt.num_threads = 2;
    arcface.opt.use_fp16_arithmetic = true; // 启用 FP16 计算
    arcface.opt.use_vulkan_compute = true; // 启用 Vulkan 加速
    arcface.opt.use_int8_inference = true; // 启用 INT8 推理

#endif
    if (arcface.load_param(param_model_path.c_str()) != 0)
        return false;
    if (arcface.load_model(bin_model_path.c_str()) != 0)
        return false;
    return true;
}

int Recognition::identify_init(const std::string& param_model_path,
                               const std::string& bin_model_path)
{
    Identify.clear();
#if NCNN_VULKAN
    Identify.opt.num_threads = 2;
    Identify.opt.use_fp16_arithmetic = true; // 启用 FP16 计算
    Identify.opt.use_vulkan_compute = true; // 启用 Vulkan 加速
    Identify.opt.use_int8_inference = true; // 启用 INT8 推理

#endif
    if (Identify.load_param(param_model_path.c_str()) != 0)
        return false;
    if (Identify.load_model(bin_model_path.c_str()) != 0)
        return false;
    return true;
}

Recognition::~Recognition()
{
    this->arcface.clear();
    this->Identify.clear();
}


cv::Mat Recognition::similarTransform(cv::Mat src, cv::Mat dst)
{
    int num = src.rows;
    int dim = src.cols;
    cv::Mat src_mean = meanAxis0(src);
    cv::Mat dst_mean = meanAxis0(dst);
    cv::Mat src_demean = elementwiseMinus(src, src_mean);
    cv::Mat dst_demean = elementwiseMinus(dst, dst_mean);
    cv::Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);
    cv::Mat d(dim, 1, CV_32F);
    d.setTo(1.0f);
    if (cv::determinant(A) < 0)
    {
        d.at<float>(dim - 1, 0) = -1;
    }
    cv::Mat T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
    cv::Mat U, S, V;
    cv::SVD::compute(A, S, U, V);
    // the SVD function in opencv differ from scipy .
    int rank = MatrixRank(A);
    if (rank == 0)
    {
        assert(rank == 0);
    }
    else if (rank == dim - 1)
    {
        if (cv::determinant(U) * cv::determinant(V) > 0)
        {
            T.rowRange(0, dim).colRange(0, dim) = U * V;
        }
        else
        {
            //            s = d[dim - 1]
            //            d[dim - 1] = -1
            //            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            //            d[dim - 1] = s
            int s = d.at<float>(dim - 1, 0) = -1;
            d.at<float>(dim - 1, 0) = -1;

            T.rowRange(0, dim).colRange(0, dim) = U * V;
            cv::Mat diag_ = cv::Mat::diag(d);
            cv::Mat twp = diag_ * V; //np.dot(np.diag(d), V.T)
            cv::Mat B = cv::Mat::zeros(3, 3, CV_8UC1);
            cv::Mat C = B.diag(0);
            T.rowRange(0, dim).colRange(0, dim) = U * twp;
            d.at<float>(dim - 1, 0) = s;
        }
    }
    else
    {
        cv::Mat diag_ = cv::Mat::diag(d);
        cv::Mat twp = diag_ * V.t(); //np.dot(np.diag(d), V.T)
        cv::Mat res = U * twp; // U
        T.rowRange(0, dim).colRange(0, dim) = -U.t() * twp;
    }
    cv::Mat var_ = varAxis0(src_demean);
    float val = cv::sum(var_).val[0];
    cv::Mat res;
    cv::multiply(d, S, res);
    float scale = 1.0 / val * cv::sum(res).val[0];
    T.rowRange(0, dim).colRange(0, dim) = -T.rowRange(0, dim).colRange(0, dim).t();
    cv::Mat temp1 = T.rowRange(0, dim).colRange(0, dim); // T[:dim, :dim]
    cv::Mat temp2 = src_mean.t(); //src_mean.T
    cv::Mat temp3 = temp1 * temp2; // np.dot(T[:dim, :dim], src_mean.T)
    cv::Mat temp4 = scale * temp3;
    T.rowRange(0, dim).colRange(dim, dim + 1) = -(temp4 - dst_mean.t());
    T.rowRange(0, dim).colRange(0, dim) *= scale;

    return T;
}

// L2 normalize
inline void Recognition::normalize(std::vector<float>& feature)
{
    float sum = 0;
    for (auto it = feature.begin(); it != feature.end(); it++)
        sum += (float)*it * (float)*it;
    sum = sqrt(sum);
    for (auto it = feature.begin(); it != feature.end(); it++)
        *it /= sum;
}

#include "iostream"

void Recognition::arcface_forward(const cv::Mat& aligned, std::vector<float>& feature)
{
    ncnn::Mat input = ncnn::Mat::from_pixels(aligned.data, ncnn::Mat::PIXEL_BGR2RGB, 112, 112);
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1 / 128.f, 1 / 128.f, 1 / 128.f};
    input.substract_mean_normalize(mean_vals, norm_vals);
    ncnn::Extractor ex = arcface.create_extractor();
    ex.input("input.1", input);
    ncnn::Mat output;
    output.create(512, 1, 1);
    ex.extract("feature", output);
    // std::cout << "output shape" << output.w << " " << output.h << " " << output.c << std::endl;
    auto dim = output.total();
    feature.resize(dim);
    for (int i = 0; i < feature.size(); ++i)
    {
        feature[i] = output[i];
    }
    normalize(feature);
}


//  a simple wrapper for function similarTransform
cv::Mat Recognition::similarity_transform(float src[5][2], float dst[5][2])
{
    cv::Mat srcmat(5, 2, CV_32FC1, src);
    memcpy(srcmat.data, src, 2 * 5 * sizeof(float));
    cv::Mat dstmat(5, 2, CV_32FC1, dst);
    memcpy(dstmat.data, dst, 2 * 5 * sizeof(float));

    return similarTransform(srcmat, dstmat);
}


void Recognition::arcface_alignment(const cv::Mat& bgr, const FaceObject& face, cv::Mat& aligned)
{
    float arcface_dst[5][2] = {
        {38.2946f, 51.6963f},
        {73.5318f, 51.5014f},
        {56.0252f, 71.7366f},
        {41.5493f, 92.3655f},
        {70.7299f, 92.2041f}
    };

    float landmark[5][2];
    for (int i = 0; i < 5; ++i)
    {
        landmark[i][0] = face.landmark[i].x;
        landmark[i][1] = face.landmark[i].y;
    }

    cv::Mat M = similarity_transform(landmark, arcface_dst);
    cv::warpPerspective(bgr, aligned, M, {112, 112});
}


void Recognition::arcface_get_feature(const cv::Mat& bgr, const FaceObject& face,
                                      std::vector<float>& feature)
{
    cv::Mat aligned;
    arcface_alignment(bgr, face, aligned);
    arcface_forward(aligned, feature);
}


float Recognition::calculate_similarity(const std::vector<float>& feature1,
                                        const std::vector<float>& feature2)
{
    float sim = 0.0;
    for (int i = 0; i < feature1.size(); ++i)
        sim += feature1[i] * feature2[i];
    return sim;
}
