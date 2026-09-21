"""실시간 SenseOn AI1/AI2 프레임 처리기.

main_video.py와 동일한 기존 접근/경로/TTC/위험도 함수를 사용한다.
기존 위험도는 baseline 경로로 결정; stable 경로는 디버깅 그림 전용.
실험용 프로토타입이며 물리적 충돌 판정을 보장하지 않는다.
"""

import cv2

from ai2.track_history import TrackHistory
from ai2.approach import calculate_approach_rate, is_approaching
from ai2.trajectory import predict_future_position, predict_stable_position
from ai2.collision import create_collision_zone, check_path_collision
from ai2.ttc import calculate_visual_ttc
from ai2.risk import determine_risk_level, select_primary_hazard
from senseon_protocol import build_final_result

TARGET_CLASSES = [1, 2, 3, 5, 7]  # bicycle/car/motorcycle/bus/truck
COLORS = {
    "SAFE": (0, 255, 0),
    "CAUTION": (0, 255, 255),
    "DANGER": (0, 0, 255),
    "UNKNOWN": (200, 200, 200),
}


class FrameAnalyzer:
    def __init__(self, model_path, mirror_direction=False):
        from ultralytics import YOLO
        self.model = YOLO(str(model_path))
        self.history = TrackHistory(max_history=10)
        self.mirror_direction = mirror_direction
        self.last_time_by_id = {}

    def process(self, frame, timestamp, annotate=False):
        """반환: (final_result, state, annotated_frame).

        state='READY': 결과가 확인됨. final_result=None은 경고 객체가 없음을 뜻함.
        state='UNKNOWN': 검출은 있는데 추적/이력 부족. SAFE라고 단정해 보내지 않음.
        """
        height, width = frame.shape[:2]
        zone = create_collision_zone(width, height)

        # 연속 프레임에서 persist=True로 동일 tracker 상태 유지.
        result = self.model.track(
            frame, persist=True, tracker="botsort.yaml", classes=TARGET_CLASSES,
            conf=0.3, verbose=False,
        )[0]
        boxes = result.boxes
        hazards = []
        drawings = []
        unknown_present = False

        if boxes is not None and len(boxes) > 0 and boxes.id is None:
            unknown_present = True

        if boxes is not None and boxes.id is not None:
            for box in boxes:
                tid = int(box.id[0])
                class_name = self.model.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                bbox = [x1, y1, x2, y2]
                detection = {
                    "track_id": tid, "class_name": class_name,
                    "confidence": float(box.conf[0]),
                    "timestamp": timestamp, "bbox": bbox,
                    "center_x": cx, "center_y": cy,
                }
                # 긴 추적 공백 후 동일 ID 재등장: 오래된 중심점 연결 방지.
                last_seen = self.last_time_by_id.get(tid)
                if last_seen is not None and timestamp - last_seen > 0.75:
                    self.history.history[tid].clear()
                self.last_time_by_id[tid] = timestamp
                self.history.update(detection)
                history = self.history.get_history(tid)
                enough = (len(history) >= 3 and
                          history[-1]["timestamp"] - history[0]["timestamp"] >= 0.1)
                risk, old, stable = "UNKNOWN", None, None
                if enough:
                    approaching = is_approaching(history)
                    old = predict_future_position(history, future_time=1.0)
                    stable = predict_stable_position(history, future_time=0.3,
                                                     window_size=10)
                    path = check_path_collision((cx, cy), old, zone)
                    ttc = calculate_visual_ttc(history)
                    risk = determine_risk_level(approaching, path, ttc)
                    # select_primary_hazard는 SAFE를 제외하므로 추가 정보 전달.
                    hazards.append({
                        "track_id": tid, "class_name": class_name,
                        "center_x": cx, "risk_level": risk, "visual_ttc": ttc,
                    })
                else:
                    unknown_present = True
                drawings.append((tid, class_name, bbox, (cx, cy), old, stable, risk))

        primary = select_primary_hazard(hazards)
        if primary is not None:
            final_result = build_final_result(primary, width,
                                              mirror=self.mirror_direction)
            state = "READY"
        elif unknown_present:
            final_result, state = None, "UNKNOWN"
        else:
            final_result, state = None, "READY"

        if not annotate:
            return final_result, state, frame

        out = frame.copy()
        zx1, zy1, zx2, zy2 = zone
        cv2.rectangle(out, (int(zx1), int(zy1)), (int(zx2), int(zy2)),
                      (255, 255, 0), 2)
        for tid, name, bbox, center, old, stable, risk in drawings:
            x1, y1, x2, y2 = bbox
            for target, color in ((old, (255, 0, 0)), (stable, (255, 0, 255))):
                if target is not None:
                    cv2.arrowedLine(out, tuple(map(int, center)),
                                    tuple(map(int, target)), color, 2, tipLength=0.15)
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                          COLORS[risk], 3 if primary and tid == primary["track_id"] else 2)
            cv2.putText(out, f"ID:{tid} {name} {risk}",
                        (max(0, int(x1)), max(95, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[risk], 2)
        text = (f"HAZARD: {final_result['object']} {final_result['direction']} "
                f"{final_result['risk']}" if final_result is not None
                else ("ANALYSIS PENDING" if state == "UNKNOWN" else "HAZARD: NONE"))
        cv2.rectangle(out, (0, 0), (width, 70), (30, 30, 30), -1)
        cv2.putText(out, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        cv2.putText(out, "BLUE: baseline / MAGENTA: candidate (debug only)",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return final_result, state, out
