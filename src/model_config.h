#pragma once
// Cau hinh model v9b cua doi (CenterNet stride 4 + TextCNN hash). Dung cung main/models/*.cc
#include "det_model.h"
#include "textcnn_model.h"
#include "vocab.h"   /* giu du bo 3 cap file nhu baseline; v9b KHONG doc vocab (ma hoa bang ham bam trong logic.c) */

#define DET_MODEL_DATA   det_model        /* mang trong det_model.cc (detector_J_int8.tflite) */
#define TXT_MODEL_DATA   textcnn_model    /* mang trong textcnn_model.cc (intent_int8_v5.tflite) */
#define VOCAB_DATA       vocab            /* vocab.cc cua baseline - khong dung boi v9b */
#define VOCAB_LEN        vocab_len

#define DET_ARENA_KB     2048   /* PSRAM; xem log "arena dung" luc khoi dong, tang neu AllocateTensors loi */
#define TXT_ARENA_KB     256
#define CONF_THRESH      0.3f   /* nguong diem hien thi/tra loi tren thiet bi (nop Kaggle dung 0.01); chinh lai theo thuc nghiem */
#define NMS_IOU          0.55f  /* NMS theo lop, nhu decode v9b */

/* Op cua 2 model v9b (doc tu 2 file .tflite). Thieu op -> AllocateTensors loi hoac bao loi bien dich */
#define RESOLVER_SIZE    16
#define REGISTER_OPS(r) do { \
    (r).AddConv2D(); (r).AddDepthwiseConv2D(); (r).AddAdd(); (r).AddMul();   /* detection */           \
    (r).AddResizeNearestNeighbor();                                                                 \
    (r).AddConcatenation(); (r).AddDequantize(); (r).AddExpandDims();        /* textcnn */            \
    (r).AddFullyConnected(); (r).AddGather(); (r).AddLess(); (r).AddReduceMax(); \
    (r).AddReshape(); (r).AddSelectV2(); (r).AddStridedSlice();                                     \
} while (0)
