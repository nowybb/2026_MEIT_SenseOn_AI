VALID_DIRECTIONS = {"LEFT", "CENTER", "RIGHT"}
VALID_RISKS = {"SAFE", "CAUTION", "DANGER"}


def encode_hazard(hazard):

    # AI 결과가 None이면 위험 객체 없음으로 처리
    if hazard is None:
        return "none,CENTER,SAFE,None"

    direction = hazard["direction"]
    risk = hazard["risk"]

    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"잘못된 direction 값: {direction}")

    if risk not in VALID_RISKS:
        raise ValueError(f"잘못된 risk 값: {risk}")

    return (
        f"{hazard['object']},"
        f"{direction},"
        f"{risk},"
        f"{hazard['ttc']}"
    )