# tests/test_scenarios.py

from ai2.track_history import TrackHistory
from ai2.approach import is_approaching
from ai2.trajectory import predict_future_position
from ai2.collision import (
    create_collision_zone,
    check_path_collision,
)
from ai2.ttc import calculate_visual_ttc
from ai2.risk import (
    determine_risk_level,
    select_primary_hazard,
)

from data.mock_data import SCENARIOS


FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def analyze_scenario(frames):
    track_history = TrackHistory(max_history=10)

    # Mock frame 입력
    for detections in frames:
        for detection in detections:
            track_history.update(detection)

    collision_zone = create_collision_zone(
        FRAME_WIDTH,
        FRAME_HEIGHT,
    )

    hazards = []

    for track_id in track_history.get_active_track_ids():
        history = track_history.get_history(track_id)
        current = history[-1]

        approaching = is_approaching(history)

        current_position = (
            current["center_x"],
            current["center_y"],
        )

        predicted_position = predict_future_position(
            history,
            future_time=1.0,
        )

        path_collision = check_path_collision(
            current_position,
            predicted_position,
            collision_zone,
        )

        visual_ttc = calculate_visual_ttc(history)

        risk_level = determine_risk_level(
            approaching,
            path_collision,
            visual_ttc,
        )

        hazards.append(
            {
                "track_id": track_id,
                "class_name": current["class_name"],
                "approaching": approaching,
                "path_collision": path_collision,
                "visual_ttc": visual_ttc,
                "risk_level": risk_level,
            }
        )

    primary_hazard = select_primary_hazard(hazards)

    return hazards, primary_hazard


def test_danger_approach():
    hazards, primary = analyze_scenario(
        SCENARIOS["danger_approach"]
    )

    assert hazards[0]["risk_level"] == "DANGER"
    assert primary is not None


def test_stationary_object():
    hazards, primary = analyze_scenario(
        SCENARIOS["stationary_object"]
    )

    assert hazards[0]["risk_level"] == "SAFE"
    assert primary is None


def test_moving_away():
    hazards, primary = analyze_scenario(
        SCENARIOS["moving_away"]
    )

    assert hazards[0]["risk_level"] == "SAFE"
    assert primary is None


def test_side_pass():
    hazards, primary = analyze_scenario(
        SCENARIOS["side_pass"]
    )

    assert hazards[0]["risk_level"] == "SAFE"
    assert primary is None


def test_bbox_jitter():
    hazards, primary = analyze_scenario(
        SCENARIOS["bbox_jitter"]
    )

    assert hazards[0]["risk_level"] == "SAFE"
    assert primary is None


def test_multi_object():
    hazards, primary = analyze_scenario(
        SCENARIOS["multi_object"]
    )

    risk_by_id = {
        hazard["track_id"]: hazard["risk_level"]
        for hazard in hazards
    }

    assert risk_by_id[10] == "DANGER"
    assert risk_by_id[20] == "SAFE"

    assert primary is not None
    assert primary["track_id"] == 10