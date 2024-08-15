# 模型移动端部署流程

## 整体流程概览

（1）训练模型
（2）将 `*.pth` 转换成 onnx， 优化 onnx 模型
（3）使用转换工具转换成可供 ncnn 使用的模型
（4）编译 ncnn 框架，并编写 c 代码调用上一步转换的模型，得到模型的输出结果，封装成可供调用的类
（5）使用 JNIC 调用上一步 C++ 封装的类，提供出接口
（6）在安卓端编写 java 代码再次封装一次，供应用层调用

## 将 `*.pth` 转换成 onnx

```cmd
# 1. 定义模型结构
# 2. 加载模型pth文件
# 3. 将模型weight参数加载到模型结构当中
# 4. 定义模型输入，输出
# 5. 模型导出为onnx
```

## 编译 NCNN 框架

### 1. windows平台使用教程

#### 1. 安装MinGW编译工具

1. 下载并安装MinGW：[Releases · niXman/mingw-builds-binaries (github.com)](https://github.com/niXman/mingw-builds-binaries/releases)，根据系统下载指定源码文件（）
2. 添加环境变量（/bin）

2. 测试是否安装成功：`gcc --version`

#### 2. 安装CMake工具

1. 下载安装CMake：[Download CMake](https://cmake.org/download/)，根据系统下载指定版本（[cmake-3.29.4-windows-x86_64.msi](https://github.com/Kitware/CMake/releases/download/v3.29.4/cmake-3.29.4-windows-x86_64.msi)）
2. 添加环境变量（/bin）

#### 3. ncnn源码下载

```cmd
git clone https://github.com/Tencent/ncnn.git
cd ncnn 
git submodule update --init
```

#### 4. 安装相关库

linux：

```cmd
sudo apt install build-essential git cmake libprotobuf-dev protobuf-compiler libvulkan-dev vulkan-utils libopencv-dev
```

windows：需要手动下载包并且编译

```cmd
# 1. Protobuf (Protocol Buffers) 
# 2. Vulkan
# 3. OpenCV
```

#####  4.1 Protobuf (Protocol Buffers) 

1. 下载链接：https://github.com/protocolbuffers/protobuf/releases?q=21&expanded=true（protoc-3.21.2）

2. 编译protobuf（3.21.2）(这里使用Visual Studio命令行工具进行编译)

```cmd
mkdir build_vs

cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%cd%/install -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_MSVC_STATIC_RUNTIME=OFF ../cmake

nmake

nmake install

```

##### 4.2 Vulkan

1. 下载链接：[LunarXchange (lunarg.com)](https://vulkan.lunarg.com/sdk/home)（**1.3.275.0**）
2. 同样添加环境变量（/bin）

##### 4.3 OpenCV（1）

1. 下载链接：[Releases - OpenCV](https://opencv.org/releases/)
2. 同样添加环境变量（\opencv\build\x64\vc16\bin）

##### opencv源码编译安装

1. 下载opencv（https://github.com/opencv/opencv/releases）源码以及opencv_contrib（https://github.com/opencv/opencv_contrib/releases/tag/4.8.1）源码
2. 使用cmake编译
3. 在cmake搜索框中搜索OPENCV_EXTRA_MODULES_PATH，并将其值设置成opencv_contrib文件夹中的modules，然后再点击configure

### 3. ncnn编译

1. 进入ncnn项目目录下面

   ```cmd
   mkdir build_vs
   
   cd build_vs
   
   cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%cd%/install -DProtobuf_INCLUDE_DIR=D:/cv/photo-album/package_install/protobuf-3.21.12/build/install/include -DProtobuf_LIBRARIES=D:/cv/photo-album/package_install/protobuf-3.21.12/build/install/lib/libprotobuf.lib -DProtobuf_PROTOC_EXECUTABLE=D:/cv/photo-album/package_install/protobuf-3.21.12/build/install/bin/protoc.exe -DNCNN_VULKAN=ON ..
   
   nmake 
   
   nmake install
   
   ```
   


## C++ 调用和封装

1. 打开vs2022，新建一个空项目：
2. 然后依次点击**视图=>其他窗口=>属性管理器**，在Release | x64（与上面编译过程中的参数对应）处右击进入属性界面。点击**VC++ 目录**，在**包含目录**中依次添加如下内容：(opencv，ncnn，protobuf)

```cmd
D:/cv/photo-album/package_install/opencv/build/include
D:/cv/photo-album/package_install/opencv/build/include/opencv
D:/cv/photo-album/package_install/opencv/build/include/opencv2
D:/cv/photo-album/ncnn/build_vs/install/include/ncnn
D:/cv/photo-album/package_install/protobuf-3.21.12/build/install/include

```

**下次编译依赖最好将依赖放在跟项目同一目录 ，引入采用相对路径比较好**

3. 在**库目录**中依次添加如下内容：

```cmd
D:/cv/photo-album/package_install/opencv/build/x64/vc16/lib
D:/cv/photo-album/ncnn/build_vs/install/lib
D:/cv/photo-album/package_install/protobuf-3.21.12/build/install/lib
```

4. 然后在属性界面选择**链接器=>输入**，在**附加依赖项**中依次添加如下内容：

   ![image-20240607105918187](https://raw.githubusercontent.com/kaisersama112/typora_image/master/image-20240607105918187.png)

```cmd
ncnn.lib
vulkan-1.lib
glslang.lib
SPIRV.lib
MachineIndependent.lib
OGLCompiler.lib
OSDependent.lib
libprotobuf.lib
opencv_world4100.lib
opencv_world4100d.lib
```

![图片](https://user-images.githubusercontent.com/3831847/158217075-4dd47ca5-8d28-4953-8733-138712ec49d2.png)

![image-20240607110010442](https://raw.githubusercontent.com/kaisersama112/typora_image/master/image-20240607110010442.png)

拷贝到C:\Windows\System32中

##  编写 JNI C++/Object-C

## 🥂人脸识别（源码调试）
基于ncnn、OpenCV、libheif 的人脸识别检测项目

### 一、环境依赖及其三方工具

1. Cmake Version: 3.28
2. C++ Version: C++ 17
3. 编译器：Visual Studio 2019
4. c++包管理工具：vcpkg
5. 操作系统：Windows 10

### 二、 三方依赖包(windows)
- [ ] ✅OpenCV 4.5.5
- [ ] ✅ncnn-20240410-windows-vs2019-shared
- [ ] ✅libheif Version: 1.12.0

### 三、 模型文件

path: /model

**不同后缀代表不同量化等级：**

| 模型                 | 说明                     |
| -------------------- | ------------------------ |
| scrfd_500m_kps.param | 检测模型（模型层级结构） |
| scrfd_500m_kps.bin   | 检测模型（模型参数信息） |
| w600k_mbf.param      | 识别模型（模型层级结构） |
| w600k_mbf.bin        | 识别模型（模型参数信息） |


### 四、项目执行流程

#### 💫 windows 平台源码调试
**CMake options**

```cmd
-DOpenCV_STATIC=ON -DCMAKE_TOOLCHAIN_FILE=E:/vcpkg/vcpkg/scripts/buildsystems/vcpkg.cmake
```
**CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.20)
project(ai-album)
set(CMAKE_CXX_STANDARD 17)
set(OpenCV_DIR "E:/opencv/opencv/build/x64/vc15/lib")
find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found!")
else ()
    message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
endif ()
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})
# ncnn
set(ncnn_DIR "${CMAKE_SOURCE_DIR}/ncnn-20240410-windows-vs2019-shared/x64/lib/cmake/ncnn")
find_package(ncnn REQUIRED)
include_directories(${ncnn_INCLUDE_DIRS})
# libheif
find_package(libheif REQUIRED)
set(libheif_LIBRARIES "heif")
include_directories(${libheif_INCLUDE_DIRS})
link_directories(${libheif_LIBRARY_DIRS})
add_executable(${PROJECT_NAME}
        include/scrfd_kps.h
        include/arcface.h
        include/scr_arc_face.h
        src/arcface.cpp
        src/scrfd_kps.cpp
        src/scr_arc_face.cpp
        main.cpp

)
if (MSVC)
    # Debug
    target_link_libraries(
            ${PROJECT_NAME}
            ${OpenCV_LIBS}
            ${libheif_LIBRARIES}
            ncnn
    )
else ()
    # Release 
    target_compile_options(
            ${PROJECT_NAME}
            ${OpenCV_LIBS}
            ${libheif_LIBRARIES}
            ncnn
    )
endif ()
```

#### 💫 android 平台源码调试

**CMakeLists.txt**

```cmake

cmake_minimum_required(VERSION 3.22.1)

project("facedetect")
set(CMAKE_SYSTEM_NAME Android)

set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/OpenCV-android-sdk/sdk/native/jni")
find_package(OpenCV REQUIRED)
set(ncnn_DIR "${CMAKE_SOURCE_DIR}/ncnn-20240102-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn")
find_package(ncnn REQUIRED)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/../../../export/${ANDROID_ABI}")

add_library(${CMAKE_PROJECT_NAME} SHARED
        include/arcface.h
        include/scrfd_kps.h
        include/scr_arc_face.h
        src/arcface.cpp
        src/scrfd_kps.cpp
        src/scr_arc_face.cpp
        jni_main.cpp
)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("${CMAKE_SOURCE_DIR}/ncnn-20240410-android-vulkan-shared/${ANDROID_ABI}/include")

set_source_files_properties(jni_main.cpp src/scr_arc_face.cpp PROPERTIES COMPILE_FLAGS "-frtti -fexceptions -DNCNN_OPENMP=OFF")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -marm -mfpu=neon -D__ARM_NEON")
#target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE -marm -mfpu=neon -D__ARM_NEON -march=armv7-a)
target_link_libraries(${CMAKE_PROJECT_NAME}
        ncnn
        ${OpenCV_LIBS}
        android
        log)
```

### 五、接口封装目录：

```c++

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
```

