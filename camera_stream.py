from picamera2 import Picamera2
import cv2
import time

picam2 = Picamera2()

config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})

picam2.configure(config)

picam2.start()
time.sleep(2)  # Allow camera to warm up

print("Starting camera stream. Press 'q' to exit.")

frame_count = 0
start_time = time.time()

try:
    while True:
        frame = picam2.capture_array()
        frame_count += 1

        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )
        # Headless-safe: save one frame per second
        if frame_count % int(fps + 1) == 0:
            cv2.imwrite("latest_frame.jpg", frame)

except KeyboardInterrupt:
    print("\nStopping camera stream...")

finally:
    picam2.stop()

# execute code:
# python3 camera_stream.py