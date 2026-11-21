#pragma once
/* Cau hinh model (cung kieu esp32_c_infer_script). Doi ten mang theo file .h trong main/models/ */
#include "det_model.h"
#include "textcnn_model.h"
#include "vocab.h"

#define DET_MODEL_DATA   det_model        /* mang trong det_model.cc */
#define TXT_MODEL_DATA   textcnn_model    /* mang trong textcnn_model.cc */
#define VOCAB_DATA       vocab            /* vocab.cc: vocab[k] co token id k+2 */
#define VOCAB_LEN        vocab_len

#define DET_ARENA_KB     2048   /* PSRAM; peak ~1.5 MB (conv1 240x320x16 int8) */
#define TXT_ARENA_KB     128
#define CONF_THRESH      0.3f   /* nhu baseline.ipynb */
#define NMS_IOU          0.5f

/* Dung cac op 2 model dang dung (doc tu file .tflite). Doi model -> AllocateTensors loi se bao op thieu */
#define RESOLVER_SIZE    12
#define REGISTER_OPS(r) do { \
    (r).AddConv2D(); (r).AddMaxPool2D();                          /* detection */          \
    (r).AddLess(); (r).AddCast(); (r).AddMul(); (r).AddAdd();     /* textcnn ban onnx2tf */ \
    (r).AddGather(); (r).AddTranspose(); (r).AddPad(); (r).AddReshape();                   \
    (r).AddReduceMax(); (r).AddFullyConnected();                                           \
} while (0)
