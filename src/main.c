/*
 * Web server inference tren ESP32-S3:
 *   GET  /     trang web (tai anh, nhap cau hoi, Run)
 *   POST /run  body = 320x240 RGB888 (230400 byte, trinh duyet da resize) + cau hoi (ASCII, da bo dau)
 *              -> JSON {ms, dets:[{c,s,b:[x1,y1,x2,y2]}], intent, answer}
 *   GET  :81/stream  MJPEG camera OV7670 (neu co) - chua noi vao detect/intent
 */
#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_http_server.h"
#include "nvs_flash.h"
#include "infer.h"
#include "cam.h"

#define MAX_Q 512
#define MAX_DETS 64

static const char *TAG = "main";
static EventGroupHandle_t s_ev;
#define BIT_GOT_IP BIT0
#define BIT_FAIL   BIT1
static int s_retry;
static bool s_ap_mode, s_connected_once;

static void on_wifi(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        if (s_ap_mode) return;
        if (s_connected_once || s_retry < CONFIG_WIFI_MAX_RETRY) {
            s_retry++;
            ESP_LOGW(TAG, "Mat ket noi WiFi, thu lai (%d)", s_retry);
            esp_wifi_connect();
        } else {
            xEventGroupSetBits(s_ev, BIT_FAIL);
        }
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *e = data;
        s_retry = 0;
        s_connected_once = true;
        ESP_LOGI(TAG, "==> Mo trinh duyet: http://" IPSTR "/", IP2STR(&e->ip_info.ip));
        xEventGroupSetBits(s_ev, BIT_GOT_IP);
    }
}

static void start_softap(void)
{
    s_ap_mode = true;
    esp_wifi_stop();
    wifi_config_t ap = {0};
    strlcpy((char *)ap.ap.ssid, CONFIG_AP_SSID, sizeof(ap.ap.ssid));
    strlcpy((char *)ap.ap.password, CONFIG_AP_PASSWORD, sizeof(ap.ap.password));
    ap.ap.ssid_len = strlen(CONFIG_AP_SSID);
    ap.ap.channel = 6;
    ap.ap.max_connection = 4;
    ap.ap.authmode = strlen(CONFIG_AP_PASSWORD) >= 8 ? WIFI_AUTH_WPA2_PSK : WIFI_AUTH_OPEN;
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_LOGW(TAG, "==> Che do AP: ket noi WiFi \"%s\" (pass \"%s\") roi mo http://192.168.4.1/",
             CONFIG_AP_SSID, CONFIG_AP_PASSWORD);
}

static void wifi_start(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();
    esp_netif_create_default_wifi_ap();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    s_ev = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, on_wifi, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, on_wifi, NULL));

    if (strlen(CONFIG_WIFI_SSID) == 0) {
        start_softap();
    } else {
        wifi_config_t sta = {0};
        strlcpy((char *)sta.sta.ssid, CONFIG_WIFI_SSID, sizeof(sta.sta.ssid));
        strlcpy((char *)sta.sta.password, CONFIG_WIFI_PASSWORD, sizeof(sta.sta.password));
        ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
        ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta));
        ESP_ERROR_CHECK(esp_wifi_start());
        ESP_LOGI(TAG, "Dang ket noi WiFi \"%s\"...", CONFIG_WIFI_SSID);
        EventBits_t b = xEventGroupWaitBits(s_ev, BIT_GOT_IP | BIT_FAIL, pdFALSE, pdFALSE,
                                            pdMS_TO_TICKS(20000));
        if (!(b & BIT_GOT_IP)) {
            ESP_LOGW(TAG, "Khong ket noi duoc WiFi -> chuyen sang AP du phong");
            start_softap();
        }
    }
    esp_wifi_set_ps(WIFI_PS_NONE);
}

extern const char index_html[] asm("_binary_index_html_start");
extern const char index_html_end[] asm("_binary_index_html_end");
static uint8_t *s_body;   /* PSRAM: anh + cau hoi */

static esp_err_t index_get(httpd_req_t *req)
{
    httpd_resp_set_type(req, "text/html; charset=utf-8");
    return httpd_resp_send(req, index_html, index_html_end - index_html);
}

static esp_err_t run_post(httpd_req_t *req)
{
    size_t len = req->content_len;
    if (len < IMG_BYTES || len > IMG_BYTES + MAX_Q)
        return httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "body = 230400 byte RGB + cau hoi");
    for (size_t got = 0; got < len;) {
        int r = httpd_req_recv(req, (char *)s_body + got, len - got);
        if (r == HTTPD_SOCK_ERR_TIMEOUT) continue;
        if (r <= 0) return ESP_FAIL;
        got += r;
    }
    char *q = (char *)s_body + IMG_BYTES;
    q[len - IMG_BYTES] = 0;
    normalize_text(q);

    static det_t d[MAX_DETS];
    int64_t t0 = esp_timer_get_time();
    int n = infer_detect(s_body, d, MAX_DETS);
    int64_t t1 = esp_timer_get_time();
    int intent = -1;
    char answer[256] = "";
    if (n >= 0 && strspn(q, " ") < strlen(q)) {
        intent = infer_intent(q);
        if (intent >= 0) make_answer(intent, q, d, n, answer, sizeof answer);
    }
    int64_t t2 = esp_timer_get_time();
    if (n < 0) return httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "detection loi");

    static char js[MAX_DETS * 80 + 512];
    size_t p = snprintf(js, sizeof js, "{\"ms_det\":%d,\"ms_intent\":%d,\"dets\":[",
                        (int)((t1 - t0) / 1000), (int)((t2 - t1) / 1000));
    for (int i = 0; i < n; i++)
        p += snprintf(js + p, sizeof js - p, "%s{\"c\":\"%s\",\"s\":%.3f,\"b\":[%.4f,%.4f,%.4f,%.4f]}",
                      i ? "," : "", CLASS_NAMES[d[i].cls], d[i].score, d[i].x1, d[i].y1, d[i].x2, d[i].y2);
    snprintf(js + p, sizeof js - p, "],\"intent\":\"%s\",\"answer\":\"%s\"}",
             intent >= 0 ? INTENT_NAMES[intent] : "", answer);   /* answer khong chua dau " */
    ESP_LOGI(TAG, "%d box, det %d ms, intent %s", n, (int)((t1 - t0) / 1000), intent >= 0 ? INTENT_NAMES[intent] : "-");
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_sendstr(req, js);
}

void app_main(void)
{
    s_body = heap_caps_malloc(IMG_BYTES + MAX_Q + 1, MALLOC_CAP_SPIRAM);
    ESP_ERROR_CHECK(s_body ? ESP_OK : ESP_ERR_NO_MEM);
    //
    if (infer_init() != 0) ESP_LOGE(TAG, "Nap model loi -> /run se khong dung duoc");
    wifi_start();

    httpd_handle_t h;
    httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
    cfg.stack_size = 16 * 1024;   /* Invoke() chay trong task httpd */
    ESP_ERROR_CHECK(httpd_start(&h, &cfg));
    httpd_register_uri_handler(h, &(httpd_uri_t){.uri = "/", .method = HTTP_GET, .handler = index_get});
    httpd_register_uri_handler(h, &(httpd_uri_t){.uri = "/run", .method = HTTP_POST, .handler = run_post});

    if (camera_web_start() != ESP_OK) {
        ESP_LOGW(TAG, "Khong co camera -> web chi dung anh tai len");
    }
    //comment lại block else khi ko can chup  
    else {
        ESP_LOGI(TAG, "San sang chup UART: nhan 'c' hoac 'C' de chup anh");
        while (1) {
            int ch = getchar();
            if (ch == 'c' || ch == 'C') cam_capture_and_send_uart();
            else if (ch == EOF) vTaskDelay(pdMS_TO_TICKS(20));
        }
    }
}
