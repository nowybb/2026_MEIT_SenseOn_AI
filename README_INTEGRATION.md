# SenseOn Raspberry Pi ↔ ESP32 통합 (실험용)

## 추가 파일 및 위치
프로젝트 **최상위** (`main_video.py`와 같은 위치)에 다음 파일을 복사합니다.
- `senseon_protocol.py`: 4필드 출력/해제 패킷 직렬화
- `senseon_pipeline.py`: 기존 AI2 계산 함수를 사용하는 실시간 프레임 분석
- `pi_ble.py`: BLE 장치 검색/연결, GATT Write, Notify ACK
- `main_realtime.py`: USB/CSI 카메라 → 추론 → BLE + 옵션 시연 영상 녹화
- `tests/test_integration.py`: 프로토콜 단위 테스트 (pytest로 직접 실행)

기존 `main_video.py`, `ai2/approach.py`, `ai2/risk.py`, `ai2/ttc.py`, `ESP32/` 파일은 변경하지 않습니다. 기존 MP4 평가 결과를 유지하기 위한 별도 실시간 실행 파일입니다. 기존의 위험도 오판 문제를 해결한 버전이 **아닙니다**.

## 사전 합의 / 실기기 확인 필수
1. ESP32 `protocol.h`는 현재 `none,CENTER,SAFE,None`을 구조상 파싱하고 `motor_control.h`는 SAFE 수신 시 모터를 끕니다. 이 21바이트 해제 패킷을 팀 간 **확정**해야 합니다.
2. 일반 BLE 기본 ATT MTU=23이면 1회 Write의 통상 payload 20바이트를 넘습니다. `none,CENTER,SAFE,None`은 21바이트, `motorcycle,CENTER,CAUTION,None`은 30바이트입니다. 전자과는 MTU 협상을 구성·확인하고 30바이트 포함 긴 패킷 단일 수신 시험을 수행해야 합니다. Pi의 BlueZ `mtu_size`가 23을 보고해도 실제 MTU를 항상 정확히 반영하는 것은 아닙니다. 이 코드에서는 무분별하게 잘라 보내지 않고 Write with response + ACK로 실패를 탐지합니다. 이 검증 전 현장 시연하지 마세요.
3. ESP32 `ble_handler.h`의 `lastPacketTime`은 파싱 전 업데이트됩니다. **유효한 패킷 수신/적용 후에만** 타임아웃 갱신하도록 전자과가 수정해야 합니다. `protocol.h`의 TTC 숫자 검증도 강화해야 합니다.
4. `direction`은 객체의 화면상 좌/중/우 위치이며 실제 주행 방향이 아닙니다. 카메라 장착 방향 및 좌우 반전을 실물로 확인하세요 (`--mirror-direction`은 통신 방향만 뒤집습니다).
5. ESP32 타임아웃 1초: 실측 프레임 처리+송신 간격이 1초를 초과하면 진동이 꺼집니다. 처리 지연/전원 부족/BLE 실패는 **안전 상태가 아니며 시스템 오류**입니다.
6. 현재 ESP32에는 진동 코드만 있고 LED 제어는 별도 구현해야 합니다.

## 환경/실행
Raspberry Pi OS에서 장치에 맞는 Ultralytics/PyTorch/OpenCV, BlueZ가 준비되어야 합니다. 예: `python -m pip install bleak` (필요하면 가상환경 사용). `picamera2`는 Raspberry Pi OS 시스템 패키지 설치를 우선 고려하세요. BLE는 Pi에서 중앙 장치, ESP32에서 주변 장치입니다.

프로젝트 최상위에서:
```bash
python -m pytest -q
python -m pytest -q tests/test_integration.py
python main_realtime.py --ble-test                       # 카메라/AI 없이 BLE·진동·ACK 먼저
python main_realtime.py --no-ble --show --frames 60       # USB 카메라에서 AI만 점검
python main_realtime.py --backend picamera2 --no-ble --frames 60  # CSI Pi 카메라
python main_realtime.py --backend picamera2 --record demo.mp4    # BLE까지 연결한 실시간 시연 녹화
```

기본 카메라는 USB OpenCV 0번이며 파일 입력을 허용하지 않습니다. Headless Pi는 `--show` 생략. USB로 실제 시연할 경우 `python main_realtime.py --record demo.mp4`.

## 실행 논리
- 현재 프레임에 위험 객체가 있으면 `object,direction,risk,ttc` 전송
- 객체가 없거나 **분석 완료된** 객체들이 SAFE면 `none,CENTER,SAFE,None` 전송
- 검출됐지만 ID/이력이 부족하면 UNKNOWN(미확정): SAFE라고 간주하지 않고 송신 보류. 수신 없음 1초 시 ESP32는 모터 OFF하므로 통신/추론 상태를 별도로 보여주는 결함 표시가 필요합니다.
- object/direction/risk 상태 변경 시 즉시 송신, TTC 수치 변화만 있을 때는 기본 0.35초 간격으로 최신값 전송. 너무 느린 Pi 추론 시 자동으로 실시간성이 보장되는 것이 아닙니다.
- ACK는 상수 `ACK`라 명령 번호와 직접 연결되지 않으므로 반드시 1개씩 순차 전송, 실기기 검증 필수.

## 코드 취급
- 기존 오프라인 평가 코드와 실시간 코드는 같은 `ai2/*` 함수(접근, 궤적, TTC, 위험)를 호출하지만 실행 제어는 분리되어 있습니다. 알고리즘을 바꿀 때 두 경로가 같은 결과를 내는지 회귀 테스트를 추가하세요.
- `BLE TX/ACK`는 통신 코드가 ACK 문자를 받았다는 의미일 뿐, 실물 모터/LED 동작 성공을 증명하지 않습니다.
- 안전 검증 이전에는 실제 도로에서 사용하지 말고 통제된 공간에서 저속 모의 접근으로 시연하세요.
