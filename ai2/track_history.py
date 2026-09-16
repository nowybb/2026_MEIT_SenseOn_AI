# ai2/track_history.py

from collections import defaultdict, deque


class TrackHistory:
    def __init__(self, max_history=10):
        self.max_history = max_history

        self.history = defaultdict(
            lambda: deque(maxlen=self.max_history)
        )

    def update(self, detection):
        """
        하나의 객체 탐지 결과를 history에 저장한다.
        """

        track_id = detection["track_id"]

        self.history[track_id].append(detection)

    def get_history(self, track_id):
        """
        특정 객체의 history를 반환한다.
        """

        return list(self.history.get(track_id, []))

    def get_active_track_ids(self):
        """
        현재 저장되어 있는 모든 track_id를 반환한다.
        """

        return list(self.history.keys())