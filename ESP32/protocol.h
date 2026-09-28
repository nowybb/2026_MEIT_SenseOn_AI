#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <Arduino.h>


// =====================================================
// 위험 정보 구조체
// =====================================================

struct HazardData {

  String object;

  String direction;

  String risk;

  float ttc;

  bool ttcValid;

  bool valid;
};


// =====================================================
// 숫자 문자열인지 확인
// 예:
// "1.23"  -> true
// "0.8"   -> true
// "abc"   -> false
// =====================================================

bool isValidNumberString(String text) {

  if (text.length() == 0) {
    return false;
  }


  bool dotFound = false;


  for (
    int i = 0;
    i < text.length();
    i++
  ) {

    char c = text.charAt(i);


    // 소수점
    if (c == '.') {

      if (dotFound) {
        return false;
      }

      dotFound = true;

      continue;
    }


    // 숫자가 아니면 실패
    if (!isDigit(c)) {
      return false;
    }
  }


  return true;
}


// =====================================================
// BLE 패킷 파싱
//
// 형식:
// object,direction,risk,ttc
//
// 예:
// car,LEFT,DANGER,1.82
// person,RIGHT,CAUTION,3.14
// none,CENTER,SAFE,None
// =====================================================

HazardData parsePacket(
  String packet
) {

  HazardData data;


  // 기본값
  data.object = "";

  data.direction = "";

  data.risk = "";

  data.ttc = 0.0;

  data.ttcValid = false;

  data.valid = false;


  packet.trim();


  // ===================================================
  // 쉼표 위치 찾기
  // ===================================================

  int comma1 =
    packet.indexOf(',');


  if (comma1 == -1) {

    Serial.println(
      "[PROTOCOL] ERROR: comma1 missing"
    );

    return data;
  }


  int comma2 =
    packet.indexOf(
      ',',
      comma1 + 1
    );


  if (comma2 == -1) {

    Serial.println(
      "[PROTOCOL] ERROR: comma2 missing"
    );

    return data;
  }


  int comma3 =
    packet.indexOf(
      ',',
      comma2 + 1
    );


  if (comma3 == -1) {

    Serial.println(
      "[PROTOCOL] ERROR: comma3 missing"
    );

    return data;
  }


  // ===================================================
  // 쉼표가 3개보다 더 있는지 확인
  // ===================================================

  int comma4 =
    packet.indexOf(
      ',',
      comma3 + 1
    );


  if (comma4 != -1) {

    Serial.println(
      "[PROTOCOL] ERROR: too many fields"
    );

    return data;
  }


  // ===================================================
  // 각 필드 분리
  // ===================================================

  data.object =
    packet.substring(
      0,
      comma1
    );


  data.direction =
    packet.substring(
      comma1 + 1,
      comma2
    );


  data.risk =
    packet.substring(
      comma2 + 1,
      comma3
    );


  String ttcString =
    packet.substring(
      comma3 + 1
    );


  data.object.trim();
  data.direction.trim();
  data.risk.trim();
  ttcString.trim();


  // ===================================================
  // object 검사
  // ===================================================

  if (
    data.object.length() == 0
  ) {

    Serial.println(
      "[PROTOCOL] ERROR: empty object"
    );

    return data;
  }


  // ===================================================
  // direction 검사
  //
  // Pi/AI 규격:
  // LEFT / CENTER / RIGHT
  // ===================================================

  if (
    data.direction != "LEFT" &&
    data.direction != "CENTER" &&
    data.direction != "RIGHT"
  ) {

    Serial.print(
      "[PROTOCOL] ERROR: invalid direction = "
    );

    Serial.println(
      data.direction
    );

    return data;
  }


  // ===================================================
  // risk 검사
  //
  // Pi/AI 규격:
  // SAFE / CAUTION / DANGER
  // ===================================================

  if (
    data.risk != "SAFE" &&
    data.risk != "CAUTION" &&
    data.risk != "DANGER"
  ) {

    Serial.print(
      "[PROTOCOL] ERROR: invalid risk = "
    );

    Serial.println(
      data.risk
    );

    return data;
  }


  // ===================================================
  // TTC 검사
  //
  // None 또는 양수 숫자
  // ===================================================

  if (
    ttcString == "None"
  ) {

    data.ttcValid = false;

  } else {

    // 숫자 문자열인지 먼저 확인
    if (
      !isValidNumberString(
        ttcString
      )
    ) {

      Serial.print(
        "[PROTOCOL] ERROR: invalid TTC text = "
      );

      Serial.println(
        ttcString
      );

      return data;
    }


    data.ttc =
      ttcString.toFloat();


    // AI팀 규격상 TTC는 양수
    if (
      data.ttc <= 0.0
    ) {

      Serial.print(
        "[PROTOCOL] ERROR: TTC must be positive = "
      );

      Serial.println(
        data.ttc
      );

      return data;
    }


    data.ttcValid = true;
  }


  // ===================================================
  // SAFE 해제 패킷 확인
  //
  // none,CENTER,SAFE,None
  // 정상적으로 통과
  // ===================================================

  if (
    data.risk == "SAFE" &&
    data.object == "none"
  ) {

    Serial.println(
      "[PROTOCOL] SAFE release packet"
    );
  }


  // ===================================================
  // 정상 패킷
  // ===================================================

  data.valid = true;


  Serial.println(
    "[PROTOCOL] Packet valid"
  );


  return data;
}


#endif
