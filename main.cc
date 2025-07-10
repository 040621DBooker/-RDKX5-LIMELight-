
// 注意: 此程序在RDK板端端运行
// Attention: This program runs on RDK board.

// D-Robotics *.bin 模型路径
// Path of D-Robotics *.bin model.
#define MODEL_PATH "/home/sunrise/my_path/LIME.bin"

// 视频文件路径
// Path of the video file for inference.
#define VIDEO_PATH "/home/sunrise/my_path/IMG_6005_RESIZE_LIME.mp4"

// 推理结果保存视频路径
// Path where the inference result video will be saved
#define OUTPUT_VIDEO_PATH "cpp_result_video.avi"

// 前处理方式选择, 0:Resize, 1:LetterBox
// Preprocessing method selection, 0: Resize, 1: LetterBox
#define RESIZE_TYPE 0
#define LETTERBOX_TYPE 1
#define PREPROCESS_TYPE LETTERBOX_TYPE

// 模型的类别数量, 默认80
// Number of classes in the model, default is 80
#define CLASSES_NUM 1

// NMS的阈值, 默认0.45
// Non-Maximum Suppression (NMS) threshold, default is 0.45
#define NMS_THRESHOLD 0.1

// 分数阈值, 默认0.25
// Score threshold, default is 0.25
#define SCORE_THRESHOLD 0.5

// NMS选取的前K个框数, 默认300
// Number of top-K boxes selected by NMS, default is 300
#define NMS_TOP_K 300

// 控制回归部分离散化程度的超参数, 默认16
// A hyperparameter that controls the discretization level of the regression part, default is 16
#define REG 16

// 绘制标签的字体尺寸, 默认1.0
// Font size for drawing labels, default is 1.0.
#define FONT_SIZE 1.0

// 绘制标签的字体粗细, 默认 1.0
// Font thickness for drawing labels, default is 1.0.
#define FONT_THICKNESS 1.0

// 绘制矩形框的线宽, 默认2.0
// Line width for drawing bounding boxes, default is 2.0.
#define LINE_SIZE 2.0

// C/C++ Standard Librarys
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

// Thrid Party Librarys
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

// RDK BPU libDNN API
#include "dnn/hb_dnn.h"
#include "dnn/hb_dnn_ext.h"
#include "dnn/plugin/hb_dnn_layer.h"
#include "dnn/plugin/hb_dnn_plugin.h"
#include "dnn/hb_sys.h"

// ！！！！！低照度增强算法(未开源)
#include "LIME.h"

#define RDK_CHECK_SUCCESS(value, errmsg)                                         \
    do                                                                           \
    {                                                                            \
        auto ret_code = value;                                                   \
        if (ret_code != 0)                                                       \
        {                                                                        \
            std::cout << "[ERROR] " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::cout << errmsg << ", error code:" << ret_code << std::endl;     \
            return ret_code;                                                     \
        }                                                                        \
    } while (0);

// COCO Names
std::vector<std::string> object_names = {"rebar"};

int main()
{
    // 0. 初始化视频输入输出
    cv::VideoCapture cap(VIDEO_PATH);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return -1;
    }

    // 获取视频属性
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    // 初始化视频写入器
    cv::VideoWriter video_writer(OUTPUT_VIDEO_PATH, 
                               cv::VideoWriter::fourcc('M','J','P','G'), 
                               fps, 
                               cv::Size(frame_width, frame_height));

    // 0. 加载bin模型
    auto begin_time = std::chrono::system_clock::now();

    hbPackedDNNHandle_t packed_dnn_handle;
    const char *model_file_name = MODEL_PATH;
    RDK_CHECK_SUCCESS(
        hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1),
        "hbDNNInitializeFromFiles failed");

    std::cout << "\033[31m Load D-Robotics Quantize model time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;

    // 1. 打印相关版本信息
    std::cout << "[INFO] OpenCV Version: " << CV_VERSION << std::endl;
    std::cout << "[INFO] MODEL_PATH: " << MODEL_PATH << std::endl;
    std::cout << "[INFO] CLASSES_NUM: " << CLASSES_NUM << std::endl;
    std::cout << "[INFO] NMS_THRESHOLD: " << NMS_THRESHOLD << std::endl;
    std::cout << "[INFO] SCORE_THRESHOLD: " << SCORE_THRESHOLD << std::endl;

    // 2. 打印模型信息
    const char **model_name_list;
    int model_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle),
        "hbDNNGetModelNameList failed");

    if (model_count > 1)
    {
        std::cout << "This model file have more than 1 model, only use model 0.";
    }
    const char *model_name = model_name_list[0];
    std::cout << "[model name]: " << model_name << std::endl;

    hbDNNHandle_t dnn_handle;
    RDK_CHECK_SUCCESS(
        hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name),
        "hbDNNGetModelHandle failed");

    int32_t input_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetInputCount(&input_count, dnn_handle),
        "hbDNNGetInputCount failed");

    hbDNNTensorProperties input_properties;
    RDK_CHECK_SUCCESS(
        hbDNNGetInputTensorProperties(&input_properties, dnn_handle, 0),
        "hbDNNGetInputTensorProperties failed");

    if (input_count > 1)
    {
        std::cout << "Your Model have more than 1 input, please check!" << std::endl;
        return -1;
    }

    if (input_properties.tensorType == HB_DNN_IMG_TYPE_NV12)
    {
        std::cout << "input tensor type: HB_DNN_IMG_TYPE_NV12" << std::endl;
    }
    else
    {
        std::cout << "input tensor type is not HB_DNN_IMG_TYPE_NV12, please check!" << std::endl;
        return -1;
    }

    std::cout << "input_properties.tensorType: " << input_properties.tensorType << std::endl;
    if (input_properties.tensorLayout == HB_DNN_LAYOUT_NCHW)
    {
        std::cout << "input tensor layout: HB_DNN_LAYOUT_NCHW" << std::endl;
    }
    else
    {
        std::cout << "input tensor layout is not HB_DNN_LAYOUT_NCHW, please check!" << std::endl;
        return -1;
    }

    int32_t input_H, input_W;
    if (input_properties.validShape.numDimensions == 4)
    {
        input_H = input_properties.validShape.dimensionSize[2];
        input_W = input_properties.validShape.dimensionSize[3];
        std::cout << "input tensor valid shape: (" << input_properties.validShape.dimensionSize[0];
        std::cout << ", " << input_properties.validShape.dimensionSize[1];
        std::cout << ", " << input_H;
        std::cout << ", " << input_W << ")" << std::endl;
    }
    else
    {
        std::cout << "input tensor validShape.numDimensions is not 4 such as (1,3,640,640), please check!" << std::endl;
        return -1;
    }

    int32_t output_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetOutputCount(&output_count, dnn_handle),
        "hbDNNGetOutputCount failed");

    if (output_count == 6)
    {
        for (int i = 0; i < 6; i++)
        {
            hbDNNTensorProperties output_properties;
            RDK_CHECK_SUCCESS(
                hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i),
                "hbDNNGetOutputTensorProperties failed");
            std::cout << "output[" << i << "] ";
            std::cout << "valid shape: (" << output_properties.validShape.dimensionSize[0];
            std::cout << ", " << output_properties.validShape.dimensionSize[1];
            std::cout << ", " << output_properties.validShape.dimensionSize[2];
            std::cout << ", " << output_properties.validShape.dimensionSize[3] << "), ";
            if (output_properties.quantiType == SHIFT)
                std::cout << "QuantiType: SHIFT" << std::endl;
            if (output_properties.quantiType == SCALE)
                std::cout << "QuantiType: SCALE" << std::endl;
            if (output_properties.quantiType == NONE)
                std::cout << "QuantiType: NONE" << std::endl;
        }
    }
    else
    {
        std::cout << "Your Model's outputs num is not 6, please check!" << std::endl;
        return -1;
    }

    int order[6] = {0, 1, 2, 3, 4, 5};
    int32_t H_8 = input_H / 8;
    int32_t H_16 = input_H / 16;
    int32_t H_32 = input_H / 32;
    int32_t W_8 = input_W / 8;
    int32_t W_16 = input_W / 16;
    int32_t W_32 = input_W / 32;
    int32_t order_we_want[6][3] = {
        {H_8, W_8, CLASSES_NUM},   // output[order[3]]: (1, H // 8,  W // 8,  CLASSES_NUM)
        {H_8, W_8, 64},            // output[order[0]]: (1, H // 8,  W // 8,  64)
        {H_16, W_16, CLASSES_NUM}, // output[order[4]]: (1, H // 16, W // 16, CLASSES_NUM)
        {H_16, W_16, 64},          // output[order[1]]: (1, H // 16, W // 16, 64)
        {H_32, W_32, CLASSES_NUM}, // output[order[5]]: (1, H // 32, W // 32, CLASSES_NUM)
        {H_32, W_32, 64},          // output[order[2]]: (1, H // 32, W // 32, 64)
    };
    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            hbDNNTensorProperties output_properties;
            RDK_CHECK_SUCCESS(
                hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, j),
                "hbDNNGetOutputTensorProperties failed");
            int32_t h = output_properties.validShape.dimensionSize[1];
            int32_t w = output_properties.validShape.dimensionSize[2];
            int32_t c = output_properties.validShape.dimensionSize[3];
            if (h == order_we_want[i][0] && w == order_we_want[i][1] && c == order_we_want[i][2])
            {
                order[i] = j;
                break;
            }
        }
    }

    if (order[0] + order[1] + order[2] + order[3] + order[4] + order[5] == 0 + 1 + 2 + 3 + 4 + 5)
    {
        std::cout << "Outputs order check SUCCESS, continue." << std::endl;
        std::cout << "order = {";
        for (int i = 0; i < 6; i++)
        {
            std::cout << order[i] << ", ";
        }
        std::cout << "}" << std::endl;
    }
    else
    {
        std::cout << "Outputs order check FAILED, use default" << std::endl;
        for (int i = 0; i < 6; i++)
            order[i] = i;
    }

    // 准备输入数据的hbDNNTensor内存空间
    hbDNNTensor input;
    input.properties = input_properties;
    hbSysAllocCachedMem(&input.sysMem[0], int(3 * input_H * input_W / 2));

    // 准备模型输出数据的空间
    hbDNNTensor *output = new hbDNNTensor[output_count];
    for (int i = 0; i < 6; i++)
    {
        hbDNNTensorProperties &output_properties = output[i].properties;
        hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i);
        int out_aligned_size = output_properties.alignedByteSize;
        hbSysMem &mem = output[i].sysMem[0];
        hbSysAllocCachedMem(&mem, out_aligned_size);
    }

    // 主循环处理视频帧
    cv::Mat frame;
    int frame_count = 0;
    hbDNNTaskHandle_t task_handle = nullptr;
    while (true) {

        // 初始化任务句柄
        task_handle = nullptr;

        auto frame_begin_time = std::chrono::system_clock::now();

        cap >> frame;  // 读取一帧
        if (frame.empty()) break;  // 视频结束

        frame_count++;
        std::cout << "\nProcessing frame " << frame_count << std::endl;

        // 使用当前帧作为输入
        cv::Mat img = frame.clone();

        //低照度增强
        img = limeEnhance(img);

        // 3.2 前处理
        float y_scale = 1.0;
        float x_scale = 1.0;
        int x_shift = 0;
        int y_shift = 0;
        cv::Mat resize_img;
        if (PREPROCESS_TYPE == LETTERBOX_TYPE) // letter box
        {
            begin_time = std::chrono::system_clock::now();
            x_scale = std::min(1.0 * input_H / img.rows, 1.0 * input_W / img.cols);
            y_scale = x_scale;
            if (x_scale <= 0 || y_scale <= 0)
            {
                throw std::runtime_error("Invalid scale factor.");
            }

            int new_w = img.cols * x_scale;
            x_shift = (input_W - new_w) / 2;
            int x_other = input_W - new_w - x_shift;

            int new_h = img.rows * y_scale;
            y_shift = (input_H - new_h) / 2;
            int y_other = input_H - new_h - y_shift;

            cv::Size targetSize(new_w, new_h);
            cv::resize(img, resize_img, targetSize);
            cv::copyMakeBorder(resize_img, resize_img, y_shift, y_other, x_shift, x_other, cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));

            std::cout << "\033[31m pre process (LetterBox) time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;
        }
        else if (PREPROCESS_TYPE == RESIZE_TYPE) // resize
        {
            begin_time = std::chrono::system_clock::now();

            cv::Size targetSize(input_W, input_H);
            cv::resize(img, resize_img, targetSize);

            y_scale = 1.0 * input_H / img.rows;
            x_scale = 1.0 * input_W / img.cols;
            y_shift = 0;
            x_shift = 0;

            std::cout << "\033[31m pre process (Resize) time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;
        }
        std::cout << "y_scale = " << y_scale << ", ";
        std::cout << "x_scale = " << x_scale << std::endl;
        std::cout << "y_shift = " << y_shift << ", ";
        std::cout << "x_shift = " << x_shift << std::endl;

        // 3.3 cv::Mat的BGR888格式转为YUV420SP格式
        begin_time = std::chrono::system_clock::now();
        cv::Mat img_nv12;
        cv::Mat yuv_mat;
        cv::cvtColor(resize_img, yuv_mat, cv::COLOR_BGR2YUV_I420);
        uint8_t *yuv = yuv_mat.ptr<uint8_t>();
        img_nv12 = cv::Mat(input_H * 3 / 2, input_W, CV_8UC1);
        uint8_t *ynv12 = img_nv12.ptr<uint8_t>();
        int uv_height = input_H / 2;
        int uv_width = input_W / 2;
        int y_size = input_H * input_W;
        memcpy(ynv12, yuv, y_size);
        uint8_t *nv12 = ynv12 + y_size;
        uint8_t *u_data = yuv + y_size;
        uint8_t *v_data = u_data + uv_height * uv_width;
        for (int i = 0; i < uv_width * uv_height; i++)
        {
            *nv12++ = *u_data++;
            *nv12++ = *v_data++;
        }
        std::cout << "\033[31m bgr8 to nv12 time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;

        // 3.4 将准备好的输入数据放入hbDNNTensor
        begin_time = std::chrono::system_clock::now();
        memcpy(input.sysMem[0].virAddr, ynv12, int(3 * input_H * input_W / 2));
        hbSysFlushMem(&input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

        // 5. 推理模型
        hbDNNInferCtrlParam infer_ctrl_param;
        HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
        hbDNNInfer(&task_handle, &output, &input, dnn_handle, &infer_ctrl_param);

        // 6. 等待任务结束
        hbDNNWaitTaskDone(task_handle, 0);
        std::cout << "\033[31m forward time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;

        // 7. YOLO11-Detect 后处理
        float CONF_THRES_RAW = -log(1 / SCORE_THRESHOLD - 1);
        std::vector<std::vector<cv::Rect2d>> bboxes(CLASSES_NUM);
        std::vector<std::vector<float>> scores(CLASSES_NUM);

        begin_time = std::chrono::system_clock::now();

        // 7.1 小目标特征图
        if (output[order[0]].properties.quantiType != NONE)
        {
            std::cout << "output[order[0]] QuantiType is not NONE, please check!" << std::endl;
            return -1;
        }
        if (output[order[1]].properties.quantiType != SCALE)
        {
            std::cout << "output[order[1]] QuantiType is not SCALE, please check!" << std::endl;
            return -1;
        }

        hbSysFlushMem(&(output[order[0]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
        hbSysFlushMem(&(output[order[1]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

        auto *s_cls_raw = reinterpret_cast<float *>(output[order[0]].sysMem[0].virAddr);
        auto *s_bbox_raw = reinterpret_cast<int32_t *>(output[order[1]].sysMem[0].virAddr);
        auto *s_bbox_scale = reinterpret_cast<float *>(output[order[1]].properties.scale.scaleData);
        for (int h = 0; h < H_8; h++)
        {
            for (int w = 0; w < W_8; w++)
            {
                float *cur_s_cls_raw = s_cls_raw;
                int32_t *cur_s_bbox_raw = s_bbox_raw;

                int cls_id = 0;
                for (int i = 1; i < CLASSES_NUM; i++)
                {
                    if (cur_s_cls_raw[i] > cur_s_cls_raw[cls_id])
                    {
                        cls_id = i;
                    }
                }

                if (cur_s_cls_raw[cls_id] < CONF_THRES_RAW)
                {
                    s_cls_raw += CLASSES_NUM;
                    s_bbox_raw += REG * 4;
                    continue;
                }

                float score = 1 / (1 + std::exp(-cur_s_cls_raw[cls_id]));

                float ltrb[4], sum, dfl;
                for (int i = 0; i < 4; i++)
                {
                    ltrb[i] = 0.;
                    sum = 0.;
                    for (int j = 0; j < REG; j++)
                    {
                        int index_id = REG * i + j;
                        dfl = std::exp(float(cur_s_bbox_raw[index_id]) * s_bbox_scale[index_id]);
                        ltrb[i] += dfl * j;
                        sum += dfl;
                    }
                    ltrb[i] /= sum;
                }

                if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0)
                {
                    s_cls_raw += CLASSES_NUM;
                    s_bbox_raw += REG * 4;
                    continue;
                }

                float x1 = (w + 0.5 - ltrb[0]) * 8.0;
                float y1 = (h + 0.5 - ltrb[1]) * 8.0;
                float x2 = (w + 0.5 + ltrb[2]) * 8.0;
                float y2 = (h + 0.5 + ltrb[3]) * 8.0;

                bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
                scores[cls_id].push_back(score);

                s_cls_raw += CLASSES_NUM;
                s_bbox_raw += REG * 4;
            }
        }

        // 7.2 中目标特征图
        if (output[order[2]].properties.quantiType != NONE)
        {
            std::cout << "output[order[2]] QuantiType is not NONE, please check!" << std::endl;
            return -1;
        }
        if (output[order[3]].properties.quantiType != SCALE)
        {
            std::cout << "output[order[3]] QuantiType is not SCALE, please check!" << std::endl;
            return -1;
        }

        hbSysFlushMem(&(output[order[2]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
        hbSysFlushMem(&(output[order[3]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

        auto *m_cls_raw = reinterpret_cast<float *>(output[order[2]].sysMem[0].virAddr);
        auto *m_bbox_raw = reinterpret_cast<int32_t *>(output[order[3]].sysMem[0].virAddr);
        auto *m_bbox_scale = reinterpret_cast<float *>(output[order[3]].properties.scale.scaleData);
        for (int h = 0; h < H_16; h++)
        {
            for (int w = 0; w < W_16; w++)
            {
                float *cur_m_cls_raw = m_cls_raw;
                int32_t *cur_m_bbox_raw = m_bbox_raw;

                int cls_id = 0;
                for (int i = 1; i < CLASSES_NUM; i++)
                {
                    if (cur_m_cls_raw[i] > cur_m_cls_raw[cls_id])
                    {
                        cls_id = i;
                    }
                }

                if (cur_m_cls_raw[cls_id] < CONF_THRES_RAW)
                {
                    m_cls_raw += CLASSES_NUM;
                    m_bbox_raw += REG * 4;
                    continue;
                }

                float score = 1 / (1 + std::exp(-cur_m_cls_raw[cls_id]));

                float ltrb[4], sum, dfl;
                for (int i = 0; i < 4; i++)
                {
                    ltrb[i] = 0.;
                    sum = 0.;
                    for (int j = 0; j < REG; j++)
                    {
                        int index_id = REG * i + j;
                        dfl = std::exp(float(cur_m_bbox_raw[index_id]) * m_bbox_scale[index_id]);
                        ltrb[i] += dfl * j;
                        sum += dfl;
                    }
                    ltrb[i] /= sum;
                }

                if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0)
                {
                    m_cls_raw += CLASSES_NUM;
                    m_bbox_raw += REG * 4;
                    continue;
                }

                float x1 = (w + 0.5 - ltrb[0]) * 16.0;
                float y1 = (h + 0.5 - ltrb[1]) * 16.0;
                float x2 = (w + 0.5 + ltrb[2]) * 16.0;
                float y2 = (h + 0.5 + ltrb[3]) * 16.0;

                bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
                scores[cls_id].push_back(score);

                m_cls_raw += CLASSES_NUM;
                m_bbox_raw += REG * 4;
            }
        }

        // 7.3 大目标特征图
        if (output[order[4]].properties.quantiType != NONE)
        {
            std::cout << "output[order[4]] QuantiType is not NONE, please check!" << std::endl;
            return -1;
        }
        if (output[order[5]].properties.quantiType != SCALE)
        {
            std::cout << "output[order[5]] QuantiType is not SCALE, please check!" << std::endl;
            return -1;
        }

        hbSysFlushMem(&(output[order[4]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
        hbSysFlushMem(&(output[order[5]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

        auto *l_cls_raw = reinterpret_cast<float *>(output[order[4]].sysMem[0].virAddr);
        auto *l_bbox_raw = reinterpret_cast<int32_t *>(output[order[5]].sysMem[0].virAddr);
        auto *l_bbox_scale = reinterpret_cast<float *>(output[order[5]].properties.scale.scaleData);
        for (int h = 0; h < H_32; h++)
        {
            for (int w = 0; w < W_32; w++)
            {
                float *cur_l_cls_raw = l_cls_raw;
                int32_t *cur_l_bbox_raw = l_bbox_raw;

                int cls_id = 0;
                for (int i = 1; i < CLASSES_NUM; i++)
                {
                    if (cur_l_cls_raw[i] > cur_l_cls_raw[cls_id])
                    {
                        cls_id = i;
                    }
                }

                if (cur_l_cls_raw[cls_id] < CONF_THRES_RAW)
                {
                    l_cls_raw += CLASSES_NUM;
                    l_bbox_raw += REG * 4;
                    continue;
                }

                float score = 1 / (1 + std::exp(-cur_l_cls_raw[cls_id]));

                float ltrb[4], sum, dfl;
                for (int i = 0; i < 4; i++)
                {
                    ltrb[i] = 0.;
                    sum = 0.;
                    for (int j = 0; j < REG; j++)
                    {
                        int index_id = REG * i + j;
                        dfl = std::exp(float(cur_l_bbox_raw[index_id]) * l_bbox_scale[index_id]);
                        ltrb[i] += dfl * j;
                        sum += dfl;
                    }
                    ltrb[i] /= sum;
                }

                if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0)
                {
                    l_cls_raw += CLASSES_NUM;
                    l_bbox_raw += REG * 4;
                    continue;
                }

                float x1 = (w + 0.5 - ltrb[0]) * 32.0;
                float y1 = (h + 0.5 - ltrb[1]) * 32.0;
                float x2 = (w + 0.5 + ltrb[2]) * 32.0;
                float y2 = (h + 0.5 + ltrb[3]) * 32.0;

                bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
                scores[cls_id].push_back(score);

                l_cls_raw += CLASSES_NUM;
                l_bbox_raw += REG * 4;
            }
        }

        // 7.4 对每一个类别进行NMS
        std::vector<std::vector<int>> indices(CLASSES_NUM);
        for (int i = 0; i < CLASSES_NUM; i++)
        {
            cv::dnn::NMSBoxes(bboxes[i], scores[i], SCORE_THRESHOLD, NMS_THRESHOLD, indices[i], 1.f, NMS_TOP_K);
        }
        
        std::cout << "\033[31m Post Process time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;

        // 8. 渲染
        begin_time = std::chrono::system_clock::now();
        for (int cls_id = 0; cls_id < CLASSES_NUM; cls_id++)
        {
            for (std::vector<int>::iterator it = indices[cls_id].begin(); it != indices[cls_id].end(); ++it)
            {
                float x1 = (bboxes[cls_id][*it].x - x_shift) / x_scale;
                float y1 = (bboxes[cls_id][*it].y - y_shift) / y_scale;
                float x2 = x1 + (bboxes[cls_id][*it].width) / x_scale;
                float y2 = y1 + (bboxes[cls_id][*it].height) / y_scale;
                float score = scores[cls_id][*it];
                std::string name = object_names[cls_id % CLASSES_NUM];
                
                // 8.3 绘制矩形
                cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), LINE_SIZE);

                // 8.4 绘制字体
                std::string text = name + ": " + std::to_string(static_cast<int>(score * 100)) + "%";
                cv::putText(img, text, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, FONT_SIZE, cv::Scalar(0, 0, 255), FONT_THICKNESS, cv::LINE_AA);

                // 8.5 打印检测信息
                std::cout << "(" << x1 << " " << y1 << " " << x2 << " " << y2 << "): \t" << text << std::endl;
            }
        }
        std::cout << "\033[31m Draw Result time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;
        
        // 9. 将处理后的帧写入视频文件
        //video_writer.write(img);
        // 可选：实时显示
        cv::imshow("Result", img);
        if (cv::waitKey(1) == 27) break;  // ESC退出

        auto frame_end_time = std::chrono::system_clock::now();
        // 计算帧时间（单位：秒）
        std::chrono::duration<double> frame_time = frame_end_time - frame_begin_time;
        double fps = 1.0 / frame_time.count();
        std::cout << "FPS: " << fps << std::endl;

        // 释放任务句柄
        if (task_handle) {
            hbDNNReleaseTask(task_handle);
        }
    }

    // 释放资源
    cap.release();
    video_writer.release();
    cv::destroyAllWindows();
    
    // 释放内存
    hbSysFreeMem(&(input.sysMem[0]));
    for (int i = 0; i < 6; i++)
        hbSysFreeMem(&(output[i].sysMem[0]));

    // 释放模型
    hbDNNRelease(packed_dnn_handle);
    
    return 0;
}
