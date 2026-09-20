import pytest

from senseon_protocol import (
    RELEASE_PACKET,
    build_final_result,
    get_direction,
    make_ble_packet,
)


# ============================================================
# 1. 위험 객체가 없을 때 경고 해제 패킷 생성
# ============================================================

def test_release_packet():
    assert make_ble_packet(None) == RELEASE_PACKET


# ============================================================
# 2. 위험 객체의 최종 결과 및 BLE 패킷 생성
# ============================================================

def test_hazard_packet():
    primary = {
        "class_name": "car",
        "center_x": 50,
        "risk_level": "DANGER",
        "visual_ttc": 1.8,
    }

    result = build_final_result(primary, 640)

    assert result == {
        "object": "car",
        "direction": "LEFT",
        "risk": "DANGER",
        "ttc": 1.8,
    }

    assert make_ble_packet(result) == "car,LEFT,DANGER,1.80"


# ============================================================
# 3. 카메라 좌우 반전 시 방향 변환
# ============================================================

def test_mirror_direction():
    assert get_direction(50, 640, mirror=True) == "RIGHT"
    assert get_direction(320, 640, mirror=True) == "CENTER"


# ============================================================
# 4. TTC가 None인 상황 및 bus 클래스 지원
# ============================================================

def test_none_ttc_and_bus():
    assert make_ble_packet({
        "object": "bus",
        "direction": "RIGHT",
        "risk": "CAUTION",
        "ttc": None,
    }) == "bus,RIGHT,CAUTION,None"


# ============================================================
# 5. 잘못된 TTC 값 차단
# ============================================================

def test_invalid_ttc_rejected():
    with pytest.raises(ValueError):
        make_ble_packet({
            "object": "car",
            "direction": "LEFT",
            "risk": "DANGER",
            "ttc": float("nan"),
        })