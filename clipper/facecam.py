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


def _snap_cam_to_edges(
    frames: list,
    med_cx: float,
    med_cy: float,
    med_w: float,
    med_h: float,
) -> tuple[int, int, int, int]:
    """Detect boxed (16:9/overlay) vs transparent streamer cam and compute clean crop bounds."""
    import cv2
    import numpy as np

    ref = frames[len(frames) // 2]
    fh, fw = ref.shape[:2]

    # Use 25th-percentile gradient across sampled frames so stationary webcam box borders
    # stand out while moving game edges fade out.
    gx_stack = []
    gy_stack = []
    for fr in frames:
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        gx_stack.append(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)))
        gy_stack.append(np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)))
    gx = np.percentile(gx_stack, 25, axis=0) if len(gx_stack) > 1 else gx_stack[0]
    gy = np.percentile(gy_stack, 25, axis=0) if len(gy_stack) > 1 else gy_stack[0]

    # Search bands around the face for crisp vertical/horizontal webcam borders
    def_w = int(min(fw * 0.34, max(med_w * 3.0, fw * 0.20)))
    def_h = int(min(fh * 0.36, max(med_h * 2.6, def_w * 9 / 16)))
    y_band_top = max(0, int(med_cy - def_h * 0.40))
    y_band_bot = min(fh, int(med_cy + def_h * 0.40))
    x_band_l = max(0, int(med_cx - def_w * 0.40))
    x_band_r = min(fw, int(med_cx + def_w * 0.40))

    def _strongest_x(x_min: int, x_max: int) -> int | None:
        if x_max <= x_min + 4 or y_band_bot <= y_band_top + 4:
            return None
        band = gx[y_band_top:y_band_bot, x_min:x_max]
        col_energy = band.mean(axis=0)
        best_rel = int(col_energy.argmax())
        peak = float(col_energy[best_rel])
        mean_val = float(col_energy.mean())
        # Require both strong peak and straight-line continuity along the vertical segment
        line_cov = float((band[:, best_rel] > max(12.0, mean_val * 1.5)).mean())
        if peak > max(14.0, mean_val * 2.3) and line_cov >= 0.50:
            return x_min + best_rel
        return None

    def _strongest_y(y_min: int, y_max: int) -> int | None:
        if y_max <= y_min + 4 or x_band_r <= x_band_l + 4:
            return None
        band = gy[y_min:y_max, x_band_l:x_band_r]
        row_energy = band.mean(axis=1)
        best_rel = int(row_energy.argmax())
        peak = float(row_energy[best_rel])
        mean_val = float(row_energy.mean())
        line_cov = float((band[best_rel, :] > max(12.0, mean_val * 1.5)).mean())
        if peak > max(14.0, mean_val * 2.3) and line_cov >= 0.50:
            return y_min + best_rel
        return None

    left_edge = 0 if med_cx < fw * 0.16 else _strongest_x(int(max(0, med_cx - fw * 0.24)), int(max(4, med_cx - med_w * 0.75)))
    right_edge = fw if med_cx > fw * 0.84 else _strongest_x(int(min(fw - 4, med_cx + med_w * 0.75)), int(min(fw, med_cx + fw * 0.24)))
    top_edge = 0 if med_cy < fh * 0.14 else _strongest_y(int(max(0, med_cy - fh * 0.22)), int(max(4, med_cy - med_h * 0.65)))
    bot_edge = fh if med_cy > fh * 0.86 else _strongest_y(int(min(fh - 4, med_cy + med_h * 0.65)), int(min(fh, med_cy + fh * 0.24)))

    detected_count = sum(e is not None for e in (left_edge, right_edge, top_edge, bot_edge))
    has_horiz = (top_edge is not None) or (bot_edge is not None)
    has_vert = (left_edge is not None) or (right_edge is not None)

    if detected_count >= 2 and has_horiz and has_vert:
        # Boxed / background facecam: respect detected borders and infer any missing side via 16:9
        if top_edge is not None and bot_edge is not None:
            cam_y = top_edge
            cam_h = max(int(med_h * 1.6), min(int(fh * 0.38), bot_edge - top_edge))
        elif top_edge is not None:
            cam_y = top_edge
            cam_h = def_h
        else:
            cam_h = def_h
            cam_y = max(0, (bot_edge or fh) - cam_h)

        if left_edge is not None and right_edge is not None:
            cam_x = left_edge
            cam_w = max(int(med_w * 2.0), min(int(fw * 0.36), right_edge - left_edge))
        elif left_edge is not None:
            cam_x = left_edge
            cam_w = min(int(fw * 0.36), max(int(med_w * 2.2), int(cam_h * 16 / 9)))
        else:
            cam_w = min(int(fw * 0.36), max(int(med_w * 2.2), int(cam_h * 16 / 9)))
            cam_x = max(0, (right_edge or fw) - cam_w)
    else:
        # Transparent / chroma-keyed streamer cam: keep head, hat/headphones, mic, and shoulders visible
        cam_w = int(max(med_w * 3.2, fw * 0.22))
        cam_w = max(int(fw * 0.16), min(int(fw * 0.34), cam_w))
        cam_h = int(max(med_h * 3.0, cam_w * 0.78))
        cam_h = max(int(fh * 0.20), min(int(fh * 0.40), cam_h))
        if med_cx < fw * 0.16:
            cam_x = 0
        elif med_cx > fw * 0.84:
            cam_x = fw - cam_w
        else:
            cam_x = int(med_cx - cam_w * 0.5)
        cam_y = int(med_cy - cam_h * 0.38)

    cam_x = max(0, min(fw - cam_w, cam_x))
    cam_y = max(0, min(fh - cam_h, cam_y))
    return int(cam_x), int(cam_y), int(cam_w), int(cam_h)


def _cluster_face_detections(
    frame_detections: list[tuple[int, list[tuple[int, int, int, int, float]]]],
    frame_w: int,
    frame_h: int,
) -> list[dict]:
    """Group per-frame face detections into spatial clusters (at most 1 hit per frame per cluster)."""
    clusters: list[dict] = []
    tol_x = frame_w * 0.10
    tol_y = frame_h * 0.10

    for frame_idx, faces in frame_detections:
        for det in faces:
            x, y, w, h, conf = det
            cx = x + w / 2.0
            cy = y + h / 2.0
            best_cluster = None
            best_dist = float("inf")
            for cl in clusters:
                dx = abs(cx - cl["sum_cx"] / cl["count"])
                dy = abs(cy - cl["sum_cy"] / cl["count"])
                if dx <= tol_x and dy <= tol_y and (dx + dy) < best_dist:
                    best_dist = dx + dy
                    best_cluster = cl

            if best_cluster is None:
                clusters.append(
                    {
                        "hits": {frame_idx: det},
                        "sum_cx": cx,
                        "sum_cy": cy,
                        "count": 1,
                    }
                )
            else:
                prev = best_cluster["hits"].get(frame_idx)
                if prev is None or conf > prev[4]:
                    best_cluster["hits"][frame_idx] = det
                best_cluster["sum_cx"] += cx
                best_cluster["sum_cy"] += cy
                best_cluster["count"] += 1

    summaries: list[dict] = []
    for cl in clusters:
        dets = list(cl["hits"].values())
        xs = sorted(d[0] + d[2] / 2.0 for d in dets)
        ys = sorted(d[1] + d[3] / 2.0 for d in dets)
        ws = sorted(d[2] for d in dets)
        hs = sorted(d[3] for d in dets)
        med_cx = xs[len(xs) // 2]
        med_cy = ys[len(ys) // 2]
        med_w = ws[len(ws) // 2]
        med_h = hs[len(hs) // 2]
        avg_conf = sum(d[4] for d in dets) / len(dets)
        pos_spread = sum(
            abs((d[0] + d[2] / 2.0) - med_cx) / max(1, frame_w)
            + abs((d[1] + d[3] / 2.0) - med_cy) / max(1, frame_h)
            for d in dets
        ) / len(dets)
        is_side_cam = (med_cx < 0.35 * frame_w) or (med_cx > 0.65 * frame_w)
        is_middle_gameplay = (0.33 * frame_w <= med_cx <= 0.67 * frame_w) and (
            0.20 * frame_h <= med_cy <= 0.80 * frame_h
        )
        summaries.append(
            {
                "frame_hits": len(dets),
                "med_cx": med_cx,
                "med_cy": med_cy,
                "med_w": med_w,
                "med_h": med_h,
                "avg_conf": avg_conf,
                "pos_spread": pos_spread,
                "is_side_cam": is_side_cam,
                "is_middle_gameplay": is_middle_gameplay,
                "face_area_ratio": (med_w * med_h) / max(1, frame_w * frame_h),
            }
        )
    return summaries


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

    frame_detections: list[tuple[int, list[tuple[int, int, int, int, float]]]] = []
    sampled_frames = []
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
        sampled_frames.append(frame)
        if idx_i == len(indices) // 2:
            preview_frame = frame.copy()
        faces = _detect_faces_in_frame(frame, yunet_detector, haar_cascade)
        if faces:
            frame_detections.append((idx_i, faces))

    cap.release()

    if preview_out is not None and preview_frame is not None:
        preview_path = Path(preview_out)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview_path), preview_frame)

    if layout_override in ("gameplay", "cam"):
        return FacecamBox(0, 0, frame_w, frame_h, frame_w, frame_h, layout=layout_override, confidence=1.0)

    if not frame_detections:
        return FacecamBox(0, 0, 0, 0, frame_w, frame_h, layout="gameplay", confidence=0.0)

    clusters = _cluster_face_detections(frame_detections, frame_w, frame_h)
    min_hits = max(2, int(sample_count * 0.35))

    # Full-screen webcam (just chatting / large centered face)
    if layout_override != "facecam":
        cam_clusters = [
            c
            for c in clusters
            if c["frame_hits"] >= min_hits and c["face_area_ratio"] > 0.06 and c["is_middle_gameplay"]
        ]
        if cam_clusters:
            best_cam = max(cam_clusters, key=lambda c: (c["frame_hits"], c["face_area_ratio"], c["avg_conf"]))
            return FacecamBox(0, 0, frame_w, frame_h, frame_w, frame_h, layout="cam", confidence=best_cam["avg_conf"])

    # Streamer facecam overlay: exclude middle-of-screen game/loading-screen faces and prefer left/right side cams
    side_clusters = [c for c in clusters if not c["is_middle_gameplay"] and c["frame_hits"] >= min_hits]
    if not side_clusters:
        return FacecamBox(0, 0, 0, 0, frame_w, frame_h, layout="gameplay", confidence=0.0)

    best = max(
        side_clusters,
        key=lambda c: (c["is_side_cam"], c["frame_hits"], -c["pos_spread"], c["avg_conf"]),
    )
    med_cx = best["med_cx"]
    med_cy = best["med_cy"]
    med_w = best["med_w"]
    med_h = best["med_h"]
    avg_conf = best["avg_conf"]

    # Snap webcam crop to crisp rectangular borders (boxed cam) or frame streamer (transparent cam)
    if sampled_frames:
        cam_x, cam_y, cam_w, cam_h = _snap_cam_to_edges(sampled_frames, med_cx, med_cy, med_w, med_h)
    else:
        cam_w = int(min(frame_w * 0.28, max(med_w * 2.6, frame_w * 0.18)))
        cam_h = int(min(frame_h * 0.28, max(med_h * 2.2, cam_w * 9 / 16)))
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
