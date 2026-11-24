#include "infer.h"
#include <math.h>
#include <string.h>
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"
#include "model_config.h"

/* Phien ban model v9b cua doi (edgeai-v9b-train.ipynb):
 *  - Detection: input int8 1x240x320x3, pixel 0..255 luong tu hoa (scale 1, zp -128);
 *    2 output int8: heatmap [1,60,80,7] va reg [1,60,80,4] (CenterNet stride 4) -> decode_v9b() trong logic.c
 *  - TextCNN: input int32 [1,120] (96 ma ky tu + 24 ma bam tu) -> tokenize_v9b(); output 5 intent (float32) */

static const char *TAG = "infer";

static tflite::MicroInterpreter *s_det, *s_txt;
static int8_t s_lut[256];             /* pixel 0..255 -> int8 da quantize */
static int s_hm_idx = 0, s_rg_idx = 1;   /* chi so output: heatmap (7 kenh) / reg (4 kenh) */

static tflite::MicroInterpreter *load(const unsigned char *data, int arena_kb, const char *name)
{
    static tflite::MicroMutableOpResolver<RESOLVER_SIZE> ops;
    static bool ops_ready;
    if (!ops_ready) { REGISTER_OPS(ops); ops_ready = true; }
    const tflite::Model *model = tflite::GetModel(data);
    if (model->version() != TFLITE_SCHEMA_VERSION) { ESP_LOGE(TAG, "%s: sai schema version", name); return nullptr; }
    uint8_t *arena = (uint8_t *)heap_caps_aligned_alloc(16, arena_kb * 1024, MALLOC_CAP_SPIRAM);
    if (!arena) { ESP_LOGE(TAG, "%s: khong cap phat duoc %d KB PSRAM", name, arena_kb); return nullptr; }
    auto *it = new tflite::MicroInterpreter(model, ops, arena, arena_kb * 1024);
    if (it->AllocateTensors() != kTfLiteOk) { ESP_LOGE(TAG, "%s: AllocateTensors loi (xem log op thieu / tang arena)", name); return nullptr; }
    ESP_LOGI(TAG, "%s OK | arena dung %u / %d KB", name, (unsigned)it->arena_used_bytes() / 1024, arena_kb);
    return it;
}

static bool det_shape_ok(const TfLiteTensor *t, int ch)
{
    return t->type == kTfLiteInt8 && t->dims->size == 4 && t->dims->data[1] == V9B_GH &&
           t->dims->data[2] == V9B_GW && t->dims->data[3] == ch;
}

int infer_init(void)
{
    s_det = load(DET_MODEL_DATA, DET_ARENA_KB, "detection");
    s_txt = load(TXT_MODEL_DATA, TXT_ARENA_KB, "textcnn");
    if (!s_det || !s_txt) { s_det = s_txt = nullptr; return -1; }

    TfLiteTensor *in = s_det->input(0);
    if (in->type != kTfLiteInt8 || in->bytes != IMG_BYTES || s_det->outputs_size() != 2) {
        ESP_LOGE(TAG, "detection: can input int8 1x240x320x3 va 2 output (heatmap, reg)");
        s_det = s_txt = nullptr;
        return -1;
    }
    /* nhan dien output nao la heatmap (7 kenh), output nao la reg (4 kenh) - thu tu co the khac nhau */
    if (det_shape_ok(s_det->output(0), NUM_CLASSES) && det_shape_ok(s_det->output(1), 4)) { s_hm_idx = 0; s_rg_idx = 1; }
    else if (det_shape_ok(s_det->output(1), NUM_CLASSES) && det_shape_ok(s_det->output(0), 4)) { s_hm_idx = 1; s_rg_idx = 0; }
    else {
        ESP_LOGE(TAG, "detection: output phai la int8 [1,%d,%d,%d] va [1,%d,%d,4]", V9B_GH, V9B_GW, NUM_CLASSES, V9B_GH, V9B_GW);
        s_det = s_txt = nullptr;
        return -1;
    }
    /* model v9b nhan pixel 0..255 (khong chia 255) */
    for (int v = 0; v < 256; v++)
        s_lut[v] = (int8_t)fminf(127, fmaxf(-128, roundf(v / in->params.scale + in->params.zero_point)));

    TfLiteTensor *ti = s_txt->input(0), *to = s_txt->output(0);
    if (ti->type != kTfLiteInt32 || ti->dims->data[ti->dims->size - 1] != V9B_SEQ_LEN) {
        ESP_LOGE(TAG, "textcnn: can input int32 [1,%d]", V9B_SEQ_LEN);
        s_det = s_txt = nullptr;
        return -1;
    }
    if (to->dims->data[to->dims->size - 1] != NUM_INTENTS) {
        ESP_LOGE(TAG, "textcnn: output can %d intent", NUM_INTENTS);
        s_det = s_txt = nullptr;
        return -1;
    }
    return 0;
}

static float deq(const TfLiteTensor *t, int i)
{
    return t->type == kTfLiteInt8 ? (t->data.int8[i] - t->params.zero_point) * t->params.scale : t->data.f[i];
}

int infer_detect(const uint8_t *rgb, det_t *out, int max)
{
    if (!s_det) return -1;
    TfLiteTensor *in = s_det->input(0);
    for (int i = 0; i < IMG_BYTES; i++) in->data.int8[i] = s_lut[rgb[i]];
    if (s_det->Invoke() != kTfLiteOk) return -1;
    const TfLiteTensor *hm = s_det->output(s_hm_idx), *rg = s_det->output(s_rg_idx);
    return decode_v9b(hm->data.int8, hm->params.scale, hm->params.zero_point,
                      rg->data.int8, rg->params.scale, rg->params.zero_point,
                      CONF_THRESH, NMS_IOU, out, max);
}

int infer_intent(const char *norm_q)
{
    if (!s_txt) return -1;
    int32_t ids[V9B_SEQ_LEN];
    tokenize_v9b(norm_q, ids);
    TfLiteTensor *in = s_txt->input(0);
    for (int i = 0; i < V9B_SEQ_LEN; i++) in->data.i32[i] = ids[i];
    if (s_txt->Invoke() != kTfLiteOk) return -1;
    TfLiteTensor *o = s_txt->output(0);
    int best = 0;
    for (int k = 1; k < NUM_INTENTS; k++) if (deq(o, k) > deq(o, best)) best = k;
    return best;
}
