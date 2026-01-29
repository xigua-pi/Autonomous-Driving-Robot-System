#include "esp_camera.h"
#include <WiFi.h>
#include <WiFiUdp.h>
#include <esp_wifi.h>

// 1. 硬件引脚定义 (幻尔 S3-Cam 专用)
#define CAMERA_MODEL_ESP32S3_EYE 
#include "camera_pins.h"

// 2. 网络配置
const char* ap_ssid = "HW_ESP32S3CAM";    //注意，这里一定需要命名wifi为这个
const char* pc_ip = "192.168.5.2"; 
const int udp_port = 1234;

WiFiUDP udp;

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("\n--- GC2145 CIF Balanced Streamer ---");

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  
  // GC2145 平衡参数
  config.xclk_freq_hz = 24000000;         // 提升时钟，加速数据读取
  config.frame_size = FRAMESIZE_CIF;      // 400x296 分辨率，视角比QVGA大，负载比VGA轻，可以自行根据latency选择不同的分辨率
  config.pixel_format = PIXFORMAT_RGB565; 
  config.grab_mode = CAMERA_GRAB_LATEST;  // 丢弃旧帧，只要最新
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.fb_count = 2;

  // 初始化摄像头
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed! 0x%x", err);
    return;
  }

  // 启动热点
  WiFi.mode(WIFI_AP);
  IPAddress local_IP(192, 168, 5, 1);
  IPAddress gateway(192, 168, 5, 1);
  IPAddress subnet(255, 255, 255, 0);
  WiFi.softAPConfig(local_IP, gateway, subnet);
  WiFi.softAP(ap_ssid, NULL);
  
  // 极致低延迟WiFi设置
  esp_wifi_set_ps(WIFI_PS_NONE); 
  WiFi.setTxPower(WIFI_POWER_19_5dBm);

  udp.begin(udp_port);
  Serial.println("System Ready. Resolution: CIF (400x296)");
}

void loop() {
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) return;

  uint8_t * jpg_buf = NULL;
  size_t jpg_len = 0;
  
  // --- 转码逻辑 ---
  // 质量设为 22。在 CIF 分辨率下，12-15 是速度与清晰度的黄金分割点。经过不断调参发现22的图形质量能够保证延迟较低，同时图形质量较好
  if (frame2jpg(fb, 22, &jpg_buf, &jpg_len)) {
    if (jpg_len < 60000) { 
      udp.beginPacket(pc_ip, udp_port);
      udp.write(jpg_buf, jpg_len);
      udp.endPacket();
    }
    free(jpg_buf); 
  }

  esp_camera_fb_return(fb);
  
  // 保持高速循环，不加额外延迟
}
