import cv2
import numpy as np


def _resize_rgb(frame, max_pixels):
    height, width = frame.shape[:2]
    scale = min(1., (max_pixels / (height * width)) ** .5)
    frame = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def load_video(path, num_frames=8, max_pixels=65536):
    if num_frames < 2 or max_pixels < 784:
        raise ValueError("Require at least 2 frames and 784 pixels")
    cap = cv2.VideoCapture(str(path))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not cap.isOpened() or count < 2 or fps <= 0:
            raise ValueError(f"Cannot decode video {path}")
        indices = np.linspace(0, count - 1, num_frames).round().astype(int)
        frames = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Failed to decode frame {index}: {path}")
            frames.append(_resize_rgb(frame, max_pixels))
        return np.stack(frames), {"source_fps": fps, "source_frames": count,
            "sampled_indices": indices.tolist(), "timestamps": (indices / fps).tolist(),
            "sampled_fps": (num_frames - 1) * fps / (count - 1),
            "height": frames[0].shape[0], "width": frames[0].shape[1]}
    finally:
        cap.release()


def load_frames(paths, num_frames=8, max_pixels=65536, assumed_sampled_fps=2.0):
    """Decode an ordered frame sequence. Original video timing is unknown."""
    if num_frames < 2 or max_pixels < 784 or len(paths) < 2:
        raise ValueError("Require at least 2 source/sample frames and 784 pixels")
    if not np.isfinite(assumed_sampled_fps) or assumed_sampled_fps <= 0:
        raise ValueError("Assumed sampled FPS must be positive and finite")
    indices = np.linspace(0, len(paths) - 1, num_frames).round().astype(int)
    frames = []
    for index in indices:
        image = cv2.imread(str(paths[index]))
        if image is None:
            raise ValueError(f"Cannot decode frame {index}: {paths[index]}")
        frames.append(_resize_rgb(image, max_pixels))
    if len({frame.shape for frame in frames}) != 1:
        raise ValueError("Frame dimensions must be consistent within a sequence")
    return np.stack(frames), {
        'input_type': 'frame_sequence', 'source_fps': None, 'source_frames': len(paths),
        'sampled_indices': indices.tolist(), 'timestamps': None,
        'sampled_fps': float(assumed_sampled_fps), 'timing_known': False,
        'timing_policy': 'assumed_sampled_fps_for_model_only',
        'height': frames[0].shape[0], 'width': frames[0].shape[1],
    }
