"""AI 최종 결과 -> 전자과 ESP32/protocol.h의 4필드 텍스트 프로토콜.

주의: 해제 패킷은 제안안. 전자과와 합의하기 전 실제 시연에 적용하지 말 것.
"""

import math

RELEASE_PACKET = "none,CENTER,SAFE,None"  # 21 bytes (UTF-8)
DIRECTIONS = frozenset({"LEFT", "CENTER", "RIGHT"})
RISKS = frozenset({"SAFE", "CAUTION", "DANGER"})


def get_direction(center_x, frame_width, mirror=False):
    """후방 화면의 객체 위치. 차량의 주행 방향이 아님."""
    if frame_width <= 0 or not 0 <= center_x <= frame_width:
        raise ValueError("잘못된 화면 너비 또는 중심점")
    if center_x < frame_width / 3:
        direction = "LEFT"
    elif center_x < frame_width * 2 / 3:
        direction = "CENTER"
    else:
        direction = "RIGHT"
    if mirror:
        direction = {"LEFT": "RIGHT", "RIGHT": "LEFT"}.get(direction, direction)
    return direction


def build_final_result(primary_hazard, frame_width, mirror=False):
    """Primary Hazard -> object/direction/risk/ttc. 없으면 None."""
    if primary_hazard is None:
        return None
    risk = primary_hazard["risk_level"]
    if risk not in ("CAUTION", "DANGER"):
        raise ValueError("Primary Hazard는 CAUTION/DANGER여야 합니다")
    return {
        "object": primary_hazard["class_name"],
        "direction": get_direction(primary_hazard["center_x"], frame_width, mirror),
        "risk": risk,
        "ttc": primary_hazard["visual_ttc"],
    }


def make_ble_packet(final_result):
    """None=위험 객체 없음 -> SAFE 명령. 카메라/AI 장애는 None으로 넣지 말 것."""
    if final_result is None:
        return RELEASE_PACKET

    if set(final_result) != {"object", "direction", "risk", "ttc"}:
        raise ValueError("최종 결과의 키는 object/direction/risk/ttc 4개여야 합니다")
    obj = final_result["object"]
    direction = final_result["direction"]
    risk = final_result["risk"]
    ttc = final_result["ttc"]

    if not isinstance(obj, str) or not obj or "," in obj or not obj.isascii():
        raise ValueError("object는 빈 문자열/쉼표/비ASCII를 포함할 수 없습니다")
    if direction not in DIRECTIONS or risk not in RISKS:
        raise ValueError("direction 또는 risk가 규격과 다릅니다")
    if ttc is None:
        text = "None"
    else:
        if isinstance(ttc, bool) or not isinstance(ttc, (int, float)):
            raise ValueError("ttc는 float 또는 None이어야 합니다")
        if not math.isfinite(float(ttc)) or ttc <= 0:
            raise ValueError("ttc는 양수이자 유한한 값이어야 합니다")
        text = f"{float(ttc):.2f}"
    return f"{obj},{direction},{risk},{text}"
