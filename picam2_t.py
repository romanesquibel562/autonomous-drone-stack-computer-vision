from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput
import cv2
import time
from collections import deque

SIZE = (640, 480)
WARMUP_S = 0.5
SAVE_ENERGY_S = 1.0

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": SIZE, "format": "RGB888"})
picam2.configure(config)

frame_q = deque(maxlen=1)

def on_frame(request):
    frame = request.make_array("main")
    frame_q.append(frame)

picam2.post_callback = on_frame

picam2.start()
time.sleep(WARMUP_S)  # Allow camera to warm up

print("Starting camera stream. Press 'CTRL+C' to exit.")

frame_count = 0
t0 = time.time()
last_save = 0.0

try:
    while True:
        if not frame_q:
            continue

        frame = frame_q[-1]
        frame_count += 1

        now = time.time()
        fps = frame_count / (now - t0) if (now - t0) > 0 else 0

        cv2.putText(
            frame, 
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        if now - last_save >= SAVE_ENERGY_S:
            cv2.imwrite("latest_frame.jpg", frame)
            last_save = now 

except KeyboardInterrupt:
    pass
finally:
    picam2.stop()
    print("Stopping camera stream...")

# Excecute with:
# python3 picam2_t.py