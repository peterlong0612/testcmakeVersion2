#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <unistd.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <android/bitmap.h>
#include <android/log.h>
#include <android/asset_manager_jni.h>
#include "include/opencv2/flann/any.h"
#include "include/opencv2/stitching/detail/warpers.hpp"

#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, "error", __VA_ARGS__))
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, "debug", __VA_ARGS__))

using namespace cv;
extern "C"
{
void BitmapToMat2(JNIEnv *env, jobject &bitmap, Mat &mat, jboolean needUnPremultiplyAlpha) {
    AndroidBitmapInfo info;
    void *pixels = 0;
    Mat &dst = mat;

    try {
        LOGD("nBitmapToMat");
        CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
        CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
                  info.format == ANDROID_BITMAP_FORMAT_RGB_565);
        CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
        CV_Assert(pixels);
        dst.create(info.height, info.width, CV_8UC4);
        if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
            LOGD("nBitmapToMat: RGBA_8888 -> CV_8UC4");
            Mat tmp(info.height, info.width, CV_8UC4, pixels);
            if (needUnPremultiplyAlpha) cvtColor(tmp, dst, COLOR_mRGBA2RGBA);
            else tmp.copyTo(dst);
        } else {
            // info.format == ANDROID_BITMAP_FORMAT_RGB_565
            LOGD("nBitmapToMat: RGB_565 -> CV_8UC4");
            Mat tmp(info.height, info.width, CV_8UC2, pixels);
            cvtColor(tmp, dst, COLOR_BGR5652RGBA);
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return;
    } catch (const cv::Exception &e) {
        AndroidBitmap_unlockPixels(env, bitmap);
        LOGE("nBitmapToMat catched cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if (!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        AndroidBitmap_unlockPixels(env, bitmap);
        LOGE("nBitmapToMat catched unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {nBitmapToMat}");
        return;
    }
}

void BitmapToMat(JNIEnv *env, jobject &bitmap, Mat &mat) {
    BitmapToMat2(env, bitmap, mat, false);
}

void MatToBitmap2
        (JNIEnv *env, Mat &mat, jobject &bitmap, jboolean needPremultiplyAlpha) {
    AndroidBitmapInfo info;
    void *pixels = 0;
    Mat &src = mat;

    try {
        LOGD("nMatToBitmap");
        CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
        CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
                  info.format == ANDROID_BITMAP_FORMAT_RGB_565);
        CV_Assert(src.dims == 2 && info.height == (uint32_t) src.rows &&
                  info.width == (uint32_t) src.cols);
        CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3 || src.type() == CV_8UC4);
        CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
        CV_Assert(pixels);
        if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
            Mat tmp(info.height, info.width, CV_8UC4, pixels);
            if (src.type() == CV_8UC1) {
                LOGD("nMatToBitmap: CV_8UC1 -> RGBA_8888");
                cvtColor(src, tmp, COLOR_GRAY2RGBA);
            } else if (src.type() == CV_8UC3) {
                LOGD("nMatToBitmap: CV_8UC3 -> RGBA_8888");
                cvtColor(src, tmp, COLOR_RGB2RGBA);
            } else if (src.type() == CV_8UC4) {
                LOGD("nMatToBitmap: CV_8UC4 -> RGBA_8888");
                if (needPremultiplyAlpha)
                    cvtColor(src, tmp, COLOR_RGBA2mRGBA);
                else
                    src.copyTo(tmp);
            }
        } else {
            // info.format == ANDROID_BITMAP_FORMAT_RGB_565
            Mat tmp(info.height, info.width, CV_8UC2, pixels);
            if (src.type() == CV_8UC1) {
                LOGD("nMatToBitmap: CV_8UC1 -> RGB_565");
                cvtColor(src, tmp, COLOR_GRAY2BGR565);
            } else if (src.type() == CV_8UC3) {
                LOGD("nMatToBitmap: CV_8UC3 -> RGB_565");
                cvtColor(src, tmp, COLOR_RGB2BGR565);
            } else if (src.type() == CV_8UC4) {
                LOGD("nMatToBitmap: CV_8UC4 -> RGB_565");
                cvtColor(src, tmp, COLOR_RGBA2BGR565);
            }
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return;
    } catch (const cv::Exception &e) {
        AndroidBitmap_unlockPixels(env, bitmap);
        LOGE("nMatToBitmap catched cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if (!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        AndroidBitmap_unlockPixels(env, bitmap);
        LOGE("nMatToBitmap catched unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {nMatToBitmap}");
        return;
    }
}

void MatToBitmap(JNIEnv *env, Mat &mat, jobject &bitmap) {
    MatToBitmap2(env, mat, bitmap, false);
}
}

//白平衡
extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_matrixFromJNI(
        JNIEnv *env,
        jobject  instance,
        jobject bitmap/* this */) {
    //std::string hello = "Hello from C++";
    //return env->NewStringUTF(hello.c_str());
    LOGD("function used");

    Mat g_srcImage,dstImage;
    std::vector<Mat> g_vChannels;
    BitmapToMat(env,bitmap,g_srcImage);
    //waitKey(0);
    LOGD("function using");
    //分离通道
    split(g_srcImage,g_vChannels);
    Mat imageBlueChannel = g_vChannels.at(0);
    Mat imageGreenChannel = g_vChannels.at(1);
    Mat imageRedChannel = g_vChannels.at(2);

    double imageBlueChannelAvg=0;
    double imageGreenChannelAvg=0;
    double imageRedChannelAvg=0;

    //求各通道的平均值
    imageBlueChannelAvg = mean(imageBlueChannel)[0];
    imageGreenChannelAvg = mean(imageGreenChannel)[0];
    imageRedChannelAvg = mean(imageRedChannel)[0];

    //求出个通道所占增益
    double K = (imageRedChannelAvg+imageGreenChannelAvg+imageRedChannelAvg)/3;
    double Kb = K/imageBlueChannelAvg;
    double Kg = K/imageGreenChannelAvg;
    double Kr = K/imageRedChannelAvg;

    //更新白平衡后的各通道BGR值
    addWeighted(imageBlueChannel,Kb,0,0,0,imageBlueChannel);
    addWeighted(imageGreenChannel,Kg,0,0,0,imageGreenChannel);
    addWeighted(imageRedChannel,Kr,0,0,0,imageRedChannel);

    merge(g_vChannels,dstImage);//图像各通道合并
    MatToBitmap(env,dstImage,bitmap);
}

//亮度提升
extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_lightup(
        JNIEnv *env,
        jobject  instance,
        jobject bitmap/* this */)
{
    Mat scr,dst;
    BitmapToMat(env,bitmap,scr);
    if (!scr.data)  //判断图像是否被正确读取；
    {
        LOGE("输入图像有误");
        return ;
    }

    int row = scr.rows;
    int col = scr.cols;


    Mat ycc;                        //转换空间到YUV；
    cvtColor(scr, ycc, COLOR_RGB2YUV);

    std::vector<Mat> channels(3);        //分离通道，取channels[0]；
    split(ycc, channels);


    Mat Luminance(row, col, CV_32FC1);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            Luminance.at<float>(i, j) =(float)channels[0].at<uchar>(i, j)/ 255;
        }
    }


    double log_Ave = 0;
    double sum = 0;
    for (int i = 0; i < row; i++)                 //求对数均值
    {
        for (int j = 0; j < col; j++)
        {
            sum += log(0.001 + Luminance.at<float>(i, j));
        }
    }
    log_Ave = exp(sum / (row*col));

    double MaxValue, MinValue;      //获取亮度最大值为MaxValue；
    minMaxLoc(Luminance, &MinValue, &MaxValue);

    Mat hdr_L (row,col,CV_32FC1);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            hdr_L.at<float>(i, j) = log(1 + Luminance.at<float>(i, j) / log_Ave) / log(1 + MaxValue / log_Ave);


            if (channels[0].at<uchar>(i, j) == 0)
            {
                hdr_L.at<float>(i, j) = 0;
            }
            else
            {
                hdr_L.at<float>(i, j) /= Luminance.at<float>(i, j);
            }

        }
    }

    std::vector<Mat> rgb_channels;        //分别对RGB三个通道进行提升
    split(scr, rgb_channels);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            int r = rgb_channels[0].at<uchar>(i, j) *hdr_L.at<float>(i, j); if ( r> 255){r = 255; }
            rgb_channels[0].at<uchar>(i, j) = r;

            int g = rgb_channels[1].at<uchar>(i, j) *hdr_L.at<float>(i, j); if (g> 255){ g = 255; }
            rgb_channels[1].at<uchar>(i, j) = g;

            int b = rgb_channels[2].at<uchar>(i, j) *hdr_L.at<float>(i, j); if (b> 255){ b = 255; }
            rgb_channels[2].at<uchar>(i, j) = b;
        }
    }
    merge(rgb_channels, dst);
    MatToBitmap(env,dst,bitmap);

}

/*
//失焦模糊

//傅里叶变换
void fftshift(const Mat& inputImg, Mat& outputImg)
{
    outputImg = inputImg.clone();
    int cx = outputImg.cols / 2;
    int cy = outputImg.rows / 2;
    Mat q0(outputImg, Rect(0, 0, cx, cy));
    Mat q1(outputImg, Rect(cx, 0, cx, cy));
    Mat q2(outputImg, Rect(0, cy, cx, cy));
    Mat q3(outputImg, Rect(cx, cy, cx, cy));
    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}
//计算PSF
void calcPSF(Mat& outputImg, Size filterSize, int R)
{
    Mat h(filterSize, CV_32F, Scalar(0));
    Point point(filterSize.width / 2, filterSize.height / 2);
    circle(h, point, R, 255, -1, 8);
    Scalar summa = sum(h);
    outputImg = h / summa[0];
}
//维纳滤波
void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr)
{
    Mat h_PSF_shifted;
    fftshift(input_h_PSF, h_PSF_shifted);
    Mat planes[2] = { Mat_<float>(h_PSF_shifted.clone()), Mat::zeros(h_PSF_shifted.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI);
    split(complexI, planes);
    Mat denom;
    pow(abs(planes[0]), 2, denom);
    denom += nsr;
    divide(planes[0], denom, output_G);
}
//反模糊
void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H)
{
    Mat planes[2] = { Mat_<float>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
    LOGD("check");
    Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI, DFT_SCALE);
    Mat planesH[2] = { Mat_<float>(H.clone()), Mat::zeros(H.size(), CV_32F) };
    Mat complexH;
    merge(planesH, 2, complexH);
    Mat complexIH;
    mulSpectrums(complexI, complexH, complexIH, 0);
    idft(complexIH, complexIH);
    split(complexIH, planes);
    outputImg = planes[0];
}

//const Mat src;

extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_outoffocus(
        JNIEnv *env,
        jobject  instance,
        jobject bitmap){
    //BitmapToMat(env,bitmap,in);
    Mat src;
    src = imread("oof.jpeg");
    Mat imgOut;
    LOGD("matrix created");
    // 偶数处理，神级操作
    Rect roi = Rect(0, 0, src.cols & -2, src.rows & -2);
    LOGD("roi.x=%d, y=%d, w=%d, h=%d", roi.x, roi.y, roi.width, roi.height);

    // 生成PSF与维纳滤波器
    Mat Hw, h;
    //试验后发现R=10,snr=40时效果较好
    calcPSF(h, roi.size(), 10);
    LOGD("after PSF");
    calcWnrFilter(h, Hw, 1.0 / double(40));
    LOGD("after Wnr");
    // 反模糊
    filter2DFreq(src(roi), imgOut, Hw);
    LOGD("after freq");
    // 归一化显示
    //imgOut.convertTo(imgOut, CV_8U);

    normalize(imgOut, imgOut, 0, 255, NORM_MINMAX);
    Mat output;
    imwrite("oofout.jpg",imgOut);
    //imgOut.convertTo(output,CV_8UC4);
    //MatToBitmap(env,output,bitmap);
}*/


//图像修复
extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_repair(
        JNIEnv *env,
        jobject  instance,
        jobject bitmap){

    Mat imageSource = imread("sdcard/rp.png");
    if (!imageSource.data)
    {
        return ;
    }
    //imshow("原图", imageSource);
    Mat imageGray;
    //转换为灰度图
    cvtColor(imageSource, imageGray, COLOR_RGB2GRAY, 0);
    Mat imageMask = Mat(imageSource.size(), CV_8UC1, Scalar::all(0));

    //通过阈值处理生成Mask
    threshold(imageGray, imageMask, 240, 255, THRESH_BINARY);
    Mat Kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    //对Mask膨胀处理，增加Mask面积
    dilate(imageMask, imageMask, Kernel);

    //图像修复
    inpaint(imageSource, imageMask, imageSource, 5, INPAINT_NS);
    imwrite("sdcard/rpoutput.jpg",imageSource);
}

//去除红眼

//孔洞填充
void fillHoles(Mat &mask)
{
    Mat maskFloodfill = mask.clone();
    //漫水填充
    floodFill(maskFloodfill, cv::Point(0, 0), Scalar(255));
    Mat mask2;
    //反色
    bitwise_not(maskFloodfill, mask2);
    //或运算
    mask = (mask2 | mask);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_redeye(
        JNIEnv *env,
        jobject  instance,
        jobject bitmap,
        jstring cascadefilename){

    // Read image 读彩色图像
    Mat img  = imread("sdcard/redeyepic.jpg");
    //BitmapToMat(env,bitmap,img);

    // Output image 输出图像
    Mat imgOut = img.clone();
    const char* cascade_file_name = env->GetStringUTFChars(cascadefilename, NULL);
    /*CascadeClassifier *eyesCa;
    if( eyesCa == nullptr){
        eyesCa = new cv::CascadeClassifier(cascade_file_name);
    }
    LOGD("%s",cascade_file_name);
    // Load HAAR cascade 读取haar分类器
    //CascadeClassifier eyesCascade("/haarcascade_eye.xml");
    // Detect eyes 检测眼睛*/
    CascadeClassifier eyesCa;
    eyesCa.load("/sdcard/haarcascade_eye.xml");
    std::vector<Rect> eyes;

    //前四个参数：输入图像，眼睛结果，表示每次图像尺寸减小的比例，表示每一个目标至少要被检测到4次才算是真的
    //后两个参数：0 | CASCADE_SCALE_IMAGE表示不同的检测模式，最小检测尺寸
    eyesCa.detectMultiScale(img,eyes, 1.1, 3, 0);
    LOGD("check");
    Mat eyeOut;
    // For every detected eye 每只眼睛都进行处理
    for (size_t i = 0; i < eyes.size(); i++)
    {
        // Extract eye from the image. 提取眼睛图像
        Mat eye = img(eyes[i]);

        // Split eye image into 3 channels. 颜色分离
        std::vector<Mat>bgr(3);
        split(eye, bgr);

        // Simple red eye detector 红眼检测器，获得结果掩模
        Mat mask = (bgr[2] > 150) & (bgr[2] > (bgr[1] + bgr[0]));

        // Clean mask 清理掩模
        //填充孔洞
        fillHoles(mask);
        //扩充孔洞
        dilate(mask, mask, Mat(), Point(-1, -1), 3, 1, 1);

        // Calculate the mean channel by averaging the green and blue channels
        //计算b通道和g通道的均值
        Mat mean = (bgr[0] + bgr[1]) / 2;
        //用该均值图像覆盖原图掩模部分图像
        mean.copyTo(bgr[2], mask);
        mean.copyTo(bgr[0], mask);
        mean.copyTo(bgr[1], mask);

        // Merge channels
        Mat eyeOut;
        //图像合并
        cv::merge(bgr, eyeOut);

        // Copy the fixed eye to the output image.
        // 眼部图像替换
        eyeOut.copyTo(imgOut(eyes[i]));
    }

    // Display Result
    //MatToBitmap(env,imgOut,bitmap);
    LOGD("aftermat2bitmap");
    //cvtColor(imgOut,imgOut,COLOR_RGB2BGR);
    imwrite("sdcard/output.jpg",imgOut);
}

