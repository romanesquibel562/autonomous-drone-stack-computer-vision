# src/perception/hailo_detector.py

from __future__ import annotations

from pathlib import Path
import numpy as np

from typing import List
from PIL import Image

from hailo_platform.pyhailort import pyhailort as hrt
from src.core.types import Detection

COCO80 = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
    "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

def _load_rgb_640(image_path: Path) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((640, 640))
    img = img.resize((640, 640))
    return np.array(img, dtype=np.uint8)

def _decode(frame0, img_w: int, img_h: int, conf_thresh: float) -> List[Detection]:
    dets: List[Detection] = []


    for cls_id, arr in enumerate(frame0):
        if arr is None or len(arr) == 0:
            continue

        for row in np.asarray(arr):
            x1, y1, x2, y2, score = map(float, row.tolist())
            if score < conf_thresh:
                continue

            bx1 = x1 * img_w
            by1 = y1 * img_h
            bx2 = x2 * img_w
            by2 = y2 * img_h

            dets.append(
                Detection(
                    label=COCO80[cls_id] if cls_id < len(COCO80) else f"class_{cls_id}",
                    confidence=score,
                    bbox=(bx1, by1, bx2, by2),
                    center_px=((bx1 + bx2) / 2, (by1 + by2) / 2),
                )
            )

    dets.sort(key=lambda d: d.confidence, reverse=True)
    return dets

def detect_from_image_path(
    image_path: str | Path,
    hef_path: str | Path = "/usr/share/hailo-models/yolov8s_h8l.hef",
    conf_thresh: float = 0.25,
) -> List[Detection]:

    frame = _load_rgb_640(Path(image_path))
    batch = 640
    inp = np.repeat(frame[None, ...], batch, axis=0)

    hef = hrt.HEF(str(hef_path))
    vdevice = hrt.VDevice()
    cfg = hrt.ConfigureParams.create_from_hef(hef, interface=hrt.HailoStreamInterface.PCIe)
    ng = vdevice.configure(hef, cfg)[0]

    in_info = ng.get_input_vstream_infos()[0]
    out_info = ng.get_output_vstream_infos()[0]

    in_params = hrt.InputVStreamParams.make_from_network_group(ng)
    out_params = hrt.OutputVStreamParams.make_from_network_group(ng)

    with hrt.InferVStreams(ng, in_params, out_params) as pipe:
        with ng.activate():
            outputs = pipe.infer({in_info.name: inp})

    frame0 = outputs[out_info.name][0]
    return _decode(frame0, img_w=640, img_h=640, conf_thresh=conf_thresh)

if __name__ == "__main__":
    dets = detect_from_image_path("/tmp/frame.jpg", conf_thresh=0.25)
    print("detections:", len(dets))
    for d in dets:
        print(d)

# run code:
# python -m src.perception.hailo_detector
