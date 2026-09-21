from ai2.risk import (
    determine_risk_level,
    select_primary_hazard,
    RiskStabilizer,
    RiskStabilizerV2,
)


def test_safe_when_not_approaching():

    result = determine_risk_level(
        approaching=False,
        path_collision=True,
        visual_ttc=1.0,
    )

    assert result == "SAFE"


def test_safe_when_no_path_collision():

    result = determine_risk_level(
        approaching=True,
        path_collision=False,
        visual_ttc=1.0,
    )

    assert result == "SAFE"


def test_caution_when_ttc_none():

    result = determine_risk_level(
        approaching=True,
        path_collision=True,
        visual_ttc=None,
    )

    assert result == "CAUTION"


def test_danger_when_ttc_short():

    result = determine_risk_level(
        approaching=True,
        path_collision=True,
        visual_ttc=1.5,
    )

    assert result == "DANGER"


def test_caution_when_ttc_medium():

    result = determine_risk_level(
        approaching=True,
        path_collision=True,
        visual_ttc=3.0,
    )

    assert result == "CAUTION"


def test_safe_when_ttc_long():

    result = determine_risk_level(
        approaching=True,
        path_collision=True,
        visual_ttc=6.0,
    )

    assert result == "SAFE"


def test_primary_hazard_prefers_higher_risk():

    hazards = [
        {
            "track_id": 1,
            "risk_level": "CAUTION",
            "visual_ttc": 1.0,
        },
        {
            "track_id": 2,
            "risk_level": "DANGER",
            "visual_ttc": 2.0,
        },
    ]

    result = select_primary_hazard(
        hazards
    )

    assert result["track_id"] == 2


def test_primary_hazard_prefers_shorter_ttc():

    hazards = [
        {
            "track_id": 1,
            "risk_level": "DANGER",
            "visual_ttc": 1.8,
        },
        {
            "track_id": 2,
            "risk_level": "DANGER",
            "visual_ttc": 1.2,
        },
    ]

    result = select_primary_hazard(
        hazards
    )

    assert result["track_id"] == 2


def test_risk_increase_is_immediate():

    stabilizer = RiskStabilizer(
        release_frames=3
    )

    assert (
        stabilizer.update(
            1,
            "SAFE",
        )
        == "SAFE"
    )

    assert (
        stabilizer.update(
            1,
            "DANGER",
        )
        == "DANGER"
    )


def test_risk_decrease_requires_persistence():

    stabilizer = RiskStabilizer(
        release_frames=3
    )

    assert (
        stabilizer.update(
            1,
            "DANGER",
        )
        == "DANGER"
    )

    assert (
        stabilizer.update(
            1,
            "SAFE",
        )
        == "DANGER"
    )

    assert (
        stabilizer.update(
            1,
            "SAFE",
        )
        == "DANGER"
    )

    assert (
        stabilizer.update(
            1,
            "SAFE",
        )
        == "SAFE"
    )


def test_decrease_counter_resets():

    stabilizer = RiskStabilizer(
        release_frames=3
    )

    stabilizer.update(
        1,
        "DANGER",
    )

    assert (
        stabilizer.update(
            1,
            "SAFE",
        )
        == "DANGER"
    )

    # 다시 위험해지면 하락 카운터 초기화
    assert (
        stabilizer.update(
            1,
            "DANGER",
        )
        == "DANGER"
    )

    assert (
        stabilizer.update(
            1,
            "SAFE",
        )
        == "DANGER"
    )

    assert (
        stabilizer.update(
            1,
            "SAFE",
        )
        == "DANGER"
    )


def test_tracks_are_independent():

    stabilizer = RiskStabilizer(
        release_frames=2
    )

    stabilizer.update(
        1,
        "DANGER",
    )

    stabilizer.update(
        2,
        "SAFE",
    )

    assert (
        stabilizer.update(
            1,
            "SAFE",
        )
        == "DANGER"
    )

    assert (
        stabilizer.update(
            2,
            "SAFE",
        )
        == "SAFE"
    )


def test_remove_track():
    
    stabilizer = RiskStabilizer(
        release_frames=3
    )

    stabilizer.update(
        1,
        "DANGER",
    )

    stabilizer.remove_track(
        1
    )

    # 삭제 후 다시 등장하면 새로운 track처럼 처리
    assert (
        stabilizer.update(
            1,
            "SAFE",
        )
        == "SAFE"
    )


def test_v2_mixed_lower_levels_allow_release():

    stabilizer = RiskStabilizerV2(
        release_frames=3
    )

    assert (
        stabilizer.update(
            1,
            "DANGER",
        )
        == "DANGER"
    )

    assert (
        stabilizer.update(
            1,
            "SAFE",
        )
        == "DANGER"
    )

    assert (
        stabilizer.update(
            1,
            "CAUTION",
        )
        == "DANGER"
    )

    # SAFE / CAUTION / SAFE 모두
    # DANGER보다 낮으므로 3회 연속 하락으로 인정
    assert (
        stabilizer.update(
            1,
            "SAFE",
        )
        == "SAFE"
    )


def test_v2_same_level_resets_release_counter():

    stabilizer = RiskStabilizerV2(
        release_frames=3
    )

    stabilizer.update(
        1,
        "DANGER",
    )

    assert (
        stabilizer.update(
            1,
            "SAFE",
        )
        == "DANGER"
    )

    assert (
        stabilizer.update(
            1,
            "CAUTION",
        )
        == "DANGER"
    )

    # 다시 DANGER가 나오면 하락 카운터 초기화
    assert (
        stabilizer.update(
            1,
            "DANGER",
        )
        == "DANGER"
    )

    assert (
        stabilizer.update(
            1,
            "SAFE",
        )
        == "DANGER"
    )


def test_v2_risk_increase_is_immediate():

    stabilizer = RiskStabilizerV2(
        release_frames=3
    )

    assert (
        stabilizer.update(
            1,
            "SAFE",
        )
        == "SAFE"
    )

    assert (
        stabilizer.update(
            1,
            "CAUTION",
        )
        == "CAUTION"
    )

    assert (
        stabilizer.update(
            1,
            "DANGER",
        )
        == "DANGER"
    )


def test_v2_tracks_are_independent():

    stabilizer = RiskStabilizerV2(
        release_frames=3
    )

    stabilizer.update(
        1,
        "DANGER",
    )

    stabilizer.update(
        2,
        "SAFE",
    )

    stabilizer.update(
        1,
        "SAFE",
    )

    stabilizer.update(
        1,
        "CAUTION",
    )

    assert (
        stabilizer.update(
            1,
            "SAFE",
        )
        == "SAFE"
    )

    assert (
        stabilizer.update(
            2,
            "SAFE",
        )
        == "SAFE"
    )