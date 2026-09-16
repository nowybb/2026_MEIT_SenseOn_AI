from data.mock_data import MOCK_FRAMES
from ai2.track_history import TrackHistory


def main():

    track_history = TrackHistory(max_history=10)

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


if __name__ == "__main__":
    main()