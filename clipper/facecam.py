"""Component 7: OpenCV YuNet facecam detector + layout classifier + preview frame saver."""

from __future__ import annotations

from pathlib import Path

from .models import FacecamBox

_YUNET_SEARCH_PATHS = [
    Path("models/face_detection_yunet_2023mar.onnx"),
    Path("face_detection_yunet_2023mar.onnx"),
]


def _find_yunet_model(explicit_path: str | Path | None = None) -> Path | None:
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p
    for p in _YUNET_SEARCH_PATHS:
        if p.exists():
            return p
    return None


def _detect_faces_in_frame(frame, yunet_detector, haar_cascade) -> list[tuple[int, int, int, int, float]]:
    """Return list of (x, y, w, h, confidence) face detections."""
    h, w = frame.shape[:2]
    results: list[tuple[int, int, int, int, float]] = []

    if yunet_detector is not None:
        yunet_detector.setInputSize((w, h))
        _ok, faces = yunet_detector.detect(frame)
        if faces is not None:
            for f in faces:
                fx, fy, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                score = float(f[-1])
                if fw > 20 and fh > 20 and score >= 0.65:
                    results.append((max(0, fx), max(0, fy), fw, fh, score))
        return results

    if haar_cascade is not None:
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(36, 36))
        for fx, fy, fw, fh in faces:
            results.append((int(fx), int(fy), int(fw), int(fh), 0.75))
    return results


def detect_facecam(
    video_path: str | Path,
    sample_count: int = 8,
    yunet_model_path: str | Path | None = None,
    layout_override: str | None = None,
    preview_out: str | Path | None = None,
) -> FacecamBox:
    """Sample frames from video_path and classify layout as 'facecam', 'cam', or 'gameplay'."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return FacecamBox(0, 0, 0, 0, 1920, 1080, layout=layout_override or "gameplay", confidence=0.0)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    yunet_detector = None
    haar_cascade = None
    model_file = _find_yunet_model(yunet_model_path)
    if model_file is not None and hasattr(cv2, "FaceDetectorYN"):
        yunet_detector = cv2.FaceDetectorYN.create(str(model_file), "", (frame_w, frame_h), 0.65, 0.3, 50)
    else:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if cascade_path.exists():
            haar_cascade = cv2.CascadeClassifier(str(cascade_path))

    detections: list[tuple[int, int, int, int, float]] = []
    preview_frame = None

    indices = (
        [int(total_frames * (i + 1) / (sample_count + 1)) for i in range(sample_count)]
        if total_frames > sample_count
        else list(range(max(1, min(total_frames, sample_count))))
    )

    for idx_i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if idx_i == len(indices) // 2:
            preview_frame = frame.copy()
        faces = _detect_faces_in_frame(frame, yunet_detector, haar_cascade)
        detections.extend(faces)

    cap.release()

    if preview_out is not None and preview_frame is not None:
        preview_path = Path(preview_out)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview_path), preview_frame)

    if layout_override in ("gameplay", "cam"):
        return FacecamBox(0, 0, frame_w, frame_h, frame_w, frame_h, layout=layout_override, confidence=1.0)

    if not detections:
        return FacecamBox(0, 0, 0, 0, frame_w, frame_h, layout="gameplay", confidence=0.0)

    # Cluster around the most consistent face region (median center & size)
    xs = sorted(d[0] + d[2] / 2 for d in detections)
    ys = sorted(d[1] + d[3] / 2 for d in detections)
    ws = sorted(d[2] for d in detections)
    hs = sorted(d[3] for d in detections)
    med_cx = xs[len(xs) // 2]
    med_cy = ys[len(ys) // 2]
    med_w = ws[len(ws) // 2]
    med_h = hs[len(hs) // 2]
    avg_conf = sum(d[4] for d in detections) / len(detections)

    # Check consistency: at least 35% of sampled frames should show a face near the median
    consistent = [
        d
        for d in detections
        if abs((d[0] + d[2] / 2) - med_cx) < frame_w * 0.15
        and abs((d[1] + d[3] / 2) - med_cy) < frame_h * 0.15
    ]
    if len(consistent) < max(2, int(sample_count * 0.35)) and layout_override != "facecam":
        return FacecamBox(0, 0, 0, 0, frame_w, frame_h, layout="gameplay", confidence=avg_conf)

    # Full-screen webcam (just chatting / face fills center) vs corner facecam overlay
    face_area_ratio = (med_w * med_h) / max(1, frame_w * frame_h)
    is_centered = (0.30 * frame_w <= med_cx <= 0.70 * frame_w) and (0.20 * frame_h <= med_cy <= 0.75 * frame_h)
    if face_area_ratio > 0.06 and is_centered and layout_override != "facecam":
        return FacecamBox(0, 0, frame_w, frame_h, frame_w, frame_h, layout="cam", confidence=avg_conf)

    # Expand face bounding box into a 16:9-ish facecam crop region
    cam_w = int(min(frame_w * 0.38, max(med_w * 3.4, frame_w * 0.22)))
    cam_h = int(min(frame_h * 0.42, max(med_h * 2.8, cam_w * 9 / 16)))
    cam_x = int(max(0, min(frame_w - cam_w, med_cx - cam_w / 2)))
    cam_y = int(max(0, min(frame_h - cam_h, med_cy - cam_h * 0.45)))

    return FacecamBox(
        x=cam_x,
        y=cam_y,
        w=cam_w,
        h=cam_h,
        frame_w=frame_w,
        frame_h=frame_h,
        layout="facecam",
        confidence=round(avg_conf, 3),
    )
