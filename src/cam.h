#pragma once
#include "esp_err.h"

/* Khoi tao OV7670 (QVGA 320x240 RGB565 = dung kich thuoc input detection) va mo server
 * MJPEG http://<ip>:81/stream. Loi -> tra ve ma loi, phan con lai cua firmware van chay.
 * Chua noi vao detect/intent: sau nay esp_camera_fb_get() -> RGB565 -> RGB888 -> infer_detect(). */
esp_err_t camera_web_start(void);
esp_err_t cam_capture_and_send_uart(void);
