# video_recorder.py
# 성능 평가 시 사용할 영상 로그 저장

from collections import deque
from pathlib import Path
from datetime import datetime

import cv2


class EventVideoRecorder:
    def __init__(
        self,
        output_dir="outputs/event_videos",
        fps=15,
        width=640,
        height=480,
        pre_seconds=3,
        post_seconds=3,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.fps = fps
        self.width = width
        self.height = height

        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds

        # DANGER 발생 전 프레임 저장용
        self.buffer = deque(
            maxlen=int(fps * pre_seconds)
        )

        self.writer = None
        self.recording = False

        # 마지막 DANGER 이후 몇 프레임 더 저장할지
        self.post_frames_left = 0

    def _start_recording(self):
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )[:-3]

        output_path = (
            self.output_dir
            / f"danger_{timestamp}.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        self.writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.fps,
            (self.width, self.height),
        )

        if not self.writer.isOpened():
            self.writer = None
            raise RuntimeError(
                f"영상 파일 생성 실패: {output_path}"
            )

        # DANGER 이전 3초 저장
        for buffered_frame in self.buffer:
            self.writer.write(
                buffered_frame
            )

        self.recording = True

        print(
            f"[VIDEO] 녹화 시작: {output_path}"
        )

    def _stop_recording(self):
        if self.writer is not None:
            self.writer.release()

        self.writer = None
        self.recording = False
        self.post_frames_left = 0

        print("[VIDEO] 녹화 종료")

    def update(self, frame, hazard):
        """
        매 프레임마다 호출.

        hazard:
        {
            "object": ...,
            "direction": ...,
            "risk": ...,
            "ttc": ...
        }

        또는 None
        """

        # OpenCV VideoWriter용 크기 보정
        if (
            frame.shape[1] != self.width
            or frame.shape[0] != self.height
        ):
            frame = cv2.resize(
                frame,
                (self.width, self.height),
            )

        is_danger = (
            hazard is not None
            and hazard.get("risk") == "DANGER"
        )

        # 아직 녹화 중이 아닐 때
        if not self.recording:

            if is_danger:
                self._start_recording()

                # 현재 DANGER 프레임 저장
                self.writer.write(frame)

                self.post_frames_left = int(
                    self.fps
                    * self.post_seconds
                )

            else:
                # 최근 3초치 유지
                self.buffer.append(
                    frame.copy()
                )

            return

        # 이미 녹화 중
        self.writer.write(frame)

        if is_danger:
            # DANGER가 계속 발생하면
            # 종료 타이머를 계속 초기화
            self.post_frames_left = int(
                self.fps
                * self.post_seconds
            )

        else:
            self.post_frames_left -= 1

            if self.post_frames_left <= 0:
                self._stop_recording()

                # 다음 이벤트용 버퍼 초기화
                self.buffer.clear()

    def close(self):
        if self.recording:
            self._stop_recording()