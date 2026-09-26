import time
import threading

import cv2
from flask import Flask, Response


# =========================================================
# Flask
# =========================================================

app = Flask(__name__)


# =========================================================
# 최신 프레임 저장
# =========================================================

latest_frame = None

frame_lock = threading.Lock()


# =========================================================
# 외부에서 프레임 업데이트
#
# test_video_ble.py에서:
#
# update_frame(annotated_frame)
#
# 형태로 호출
# =========================================================

def update_frame(frame):

    global latest_frame

    if frame is None:
        return

    with frame_lock:

        # 다른 스레드에서 읽기 때문에 copy
        latest_frame = frame.copy()


# =========================================================
# MJPEG Generator
# =========================================================

def generate_frames():

    while True:

        # ---------------------------------------------
        # 현재 최신 프레임 가져오기
        # ---------------------------------------------

        with frame_lock:

            if latest_frame is None:

                frame = None

            else:

                frame = latest_frame.copy()


        # 아직 프레임이 없으면 잠깐 대기
        if frame is None:

            time.sleep(0.05)

            continue


        # ---------------------------------------------
        # JPEG 인코딩
        #
        # quality 70:
        # 네트워크 사용량 감소
        # ---------------------------------------------

        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                70
            ]
        )


        if not success:

            time.sleep(0.05)

            continue


        frame_bytes = buffer.tobytes()


        # ---------------------------------------------
        # MJPEG 전송
        # ---------------------------------------------

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


        # ---------------------------------------------
        # 스트리밍 최대 약 10 FPS
        #
        # AI 자체가 현재 3~4 FPS 정도라
        # 이 정도면 충분
        # ---------------------------------------------

        time.sleep(0.1)


# =========================================================
# 메인 페이지
# =========================================================

@app.route("/")
def index():

    return """
    <!DOCTYPE html>

    <html>

    <head>

        <meta charset="UTF-8">

        <title>SenseOn AI Live Feed</title>

        <style>

            body {
                background-color: #111;
                color: white;

                font-family: Arial, sans-serif;

                text-align: center;

                margin: 0;
                padding: 20px;
            }

            h1 {
                margin-bottom: 20px;
            }

            img {
                width: 90%;
                max-width: 1000px;

                height: auto;

                border: 2px solid #444;
            }

        </style>

    </head>


    <body>

        <h1>
            SenseOn AI Live Feed
        </h1>


        <img
            src="/video"
            alt="SenseOn Stream"
        >

    </body>

    </html>
    """


# =========================================================
# 영상 스트림
# =========================================================

@app.route("/video")
def video():

    return Response(
        generate_frames(),

        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# =========================================================
# 서버 실행
# =========================================================

def run_stream_server():

    print(
        "[STREAM] Starting server..."
    )

    print(
        "[STREAM] http://0.0.0.0:5000"
    )


    app.run(
        host="0.0.0.0",
        port=5000,

        debug=False,

        use_reloader=False,

        threaded=True
    )


# =========================================================
# 단독 실행 테스트
# =========================================================

if __name__ == "__main__":

    run_stream_server()