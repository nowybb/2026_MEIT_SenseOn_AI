import time


def now_ms():
    # 현재 시간을 밀리초 단위로 반환
    return time.perf_counter() * 1000


def calc_latency_ms(start_ms, end_ms):
    # 두 시점 사이의 시간 차이를 ms로 계산
    return end_ms - start_ms

# End-to-End Latency = ESP32의 모터 ON 직후 전송한 ACK 수신 시점 - AI 판단 완료 시점