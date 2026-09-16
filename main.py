from data.mock_data import MOCK_FRAMES
from ai2.track_history import TrackHistory
from ai2.approach import calculate_approach_rate, is_approaching
from ai2.trajectory import (
    calculate_velocity,
    predict_future_position,
)
from ai2.collision import (
    create_collision_zone,
    check_path_collision,
)
from ai2.ttc import calculate_visual_ttc

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def main():

    track_history = TrackHistory(max_history=10)
    
    # Collision Zone 생성
    collision_zone = create_collision_zone(
        FRAME_WIDTH,
        FRAME_HEIGHT
    )

    print(f"\nCollision Zone: {collision_zone}")

    for frame_number, detections in enumerate(MOCK_FRAMES):

        print(f"\n===== Frame {frame_number + 1} =====")

        for detection in detections:

            track_history.update(detection)

            print(
                f"Track ID: {detection['track_id']} | "
                f"Class: {detection['class_name']} | "
                f"BBox: {detection['bbox']}"
            )

    print("\n===== Final Track History =====")

    for track_id in track_history.get_active_track_ids():

        history = track_history.get_history(track_id)

        print(f"\nTrack ID {track_id}")

        for item in history:
            print(
                f"time={item['timestamp']:.1f} "
                f"bbox={item['bbox']} "
                f"center=({item['center_x']}, {item['center_y']})"
            )

        print("\n===== Approach Analysis =====")

    for track_id in track_history.get_active_track_ids():

        history = track_history.get_history(track_id)

        approach_rate = calculate_approach_rate(history)
        approaching = is_approaching(history)

        class_name = history[-1]["class_name"]

        print(
            f"Track ID: {track_id} | "
            f"Class: {class_name} | "
            f"ApproachRate: {approach_rate:.3f} /s | "
            f"Approaching: {approaching}"
        )
        
        print("\n===== Trajectory Analysis =====")

    for track_id in track_history.get_active_track_ids():

        history = track_history.get_history(track_id)

        vx, vy = calculate_velocity(history)

        predicted_position = predict_future_position(
            history,
            future_time=1.0
        )

        class_name = history[-1]["class_name"]

        print(
            f"Track ID: {track_id} | "
            f"Class: {class_name} | "
            f"Velocity: ({vx:.2f}, {vy:.2f}) px/s | "
            f"Predicted Position: "
            f"({predicted_position[0]:.2f}, "
            f"{predicted_position[1]:.2f})"
        )

        print("\n===== Collision Analysis =====")

    for track_id in track_history.get_active_track_ids():

        history = track_history.get_history(track_id)

        current = history[-1]

        current_position = (
            current["center_x"],
            current["center_y"],
        )

        predicted_position = predict_future_position(
            history,
            future_time=1.0
        )

        path_collision = check_path_collision(
            current_position,
            predicted_position,
            collision_zone,
        )

        print(
            f"Track ID: {track_id} | "
            f"Class: {current['class_name']} | "
            f"Current: {current_position} | "
            f"Predicted: "
            f"({predicted_position[0]:.2f}, "
            f"{predicted_position[1]:.2f}) | "
            f"PathCollision: {path_collision}"
        )
    
        print("\n===== Visual TTC Analysis =====")

    for track_id in track_history.get_active_track_ids():

        history = track_history.get_history(track_id)

        visual_ttc = calculate_visual_ttc(history)

        class_name = history[-1]["class_name"]

        if visual_ttc is None:
            ttc_text = "N/A"
        else:
            ttc_text = f"{visual_ttc:.2f} s"

        print(
            f"Track ID: {track_id} | "
            f"Class: {class_name} | "
            f"Visual TTC: {ttc_text}"
        )
    
if __name__ == "__main__":
    main()