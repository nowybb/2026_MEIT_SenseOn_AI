from picamera2 import Picamera2
import time


class Camera:
    def __init__(self, width=640, height=480):
        self.picam2 = Picamera2()

        # 카메라 출력 설정
        config = self.picam2.create_preview_configuration(
            main={
                "size": (width, height),
                "format": "RGB888"
            }
        )

        self.picam2.configure(config)

    def start(self):
        # 카메라 시작
        self.picam2.start()

        # 카메라가 안정화될 시간
        time.sleep(1)

    def get_frame(self):
        # 현재 프레임을 NumPy 배열 형태로 반환
        frame = self.picam2.capture_array()
        return frame

    def stop(self):
        # 카메라 종료
        self.picam2.stop()