// ==============================
// protocol.h
// ==============================
#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <Arduino.h>

struct HazardData {
  String object;
  String direction;
  String risk;

  float ttc;
  bool ttcValid;

  bool valid;
};


HazardData parsePacket(String packet) {

  HazardData data;

  data.object = "";
  data.direction = "";
  data.risk = "";

  data.ttc = 0.0;
  data.ttcValid = false;

  data.valid = false;

  packet.trim();


  int comma1 = packet.indexOf(',');

  if (comma1 == -1) {
    return data;
  }


  int comma2 =
    packet.indexOf(',', comma1 + 1);

  if (comma2 == -1) {
    return data;
  }


  int comma3 =
    packet.indexOf(',', comma2 + 1);

  if (comma3 == -1) {
    return data;
  }


  data.object =
    packet.substring(0, comma1);

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
    packet.substring(comma3 + 1);


  data.object.trim();
  data.direction.trim();
  data.risk.trim();
  ttcString.trim();


  if (
    data.direction != "LEFT" &&
    data.direction != "CENTER" &&
    data.direction != "RIGHT"
  ) {
    return data;
  }


  if (
    data.risk != "SAFE" &&
    data.risk != "CAUTION" &&
    data.risk != "DANGER"
  ) {
    return data;
  }


  if (
    ttcString == "None" ||
    ttcString == "NONE" ||
    ttcString == "none"
  ) {

    data.ttcValid = false;

  } else {

    data.ttc =
      ttcString.toFloat();

    data.ttcValid = true;
  }


  data.valid = true;

  return data;
}

#endif
