// ==============================
// display_control.h
// ==============================
#ifndef DISPLAY_CONTROL_H
#define DISPLAY_CONTROL_H

#include <Arduino.h>
#include <Wire.h>

#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#include "config.h"
#include "protocol.h"


Adafruit_SSD1306 display(
  SCREEN_WIDTH,
  SCREEN_HEIGHT,
  &Wire,
  -1
);


// 가운데 정렬
void drawCenteredText(
  String text,
  int y,
  int textSize
) {

  display.setTextSize(textSize);

  int16_t x1;
  int16_t y1;

  uint16_t w;
  uint16_t h;


  display.getTextBounds(
    text,
    0,
    y,
    &x1,
    &y1,
    &w,
    &h
  );


  int x =
    (SCREEN_WIDTH - w) / 2;


  display.setCursor(
    x,
    y
  );

  display.print(text);
}


// OLED 초기화
void setupDisplay() {

  Wire.begin(
    OLED_SDA,
    OLED_SCL
  );


  if (
    !display.begin(
      SSD1306_SWITCHCAPVCC,
      OLED_ADDRESS
    )
  ) {

    Serial.println(
      "[OLED] FAIL"
    );

    while (true) {
      delay(100);
    }
  }


  display.clearDisplay();

  display.setTextColor(
    SSD1306_WHITE
  );


  drawCenteredText(
    "SenseOn",
    10,
    2
  );


  drawCenteredText(
    "READY",
    38,
    2
  );


  display.display();


  Serial.println(
    "[OLED] Ready"
  );
}


// 위험 정보 표시
void showHazard(
  const HazardData& hazard
) {

  display.clearDisplay();

  display.setTextColor(
    SSD1306_WHITE
  );


  // 객체
  drawCenteredText(
    hazard.object,
    0,
    1
  );


  // 방향
  drawCenteredText(
    hazard.direction,
    14,
    2
  );


  // 위험도
  drawCenteredText(
    hazard.risk,
    34,
    2
  );


  String ttcText;


  if (hazard.ttcValid) {

    ttcText =
      "TTC: " +
      String(hazard.ttc, 1) +
      "s";

  } else {

    ttcText =
      "TTC: None";
  }


  drawCenteredText(
    ttcText,
    55,
    1
  );


  display.display();
}

#endif
