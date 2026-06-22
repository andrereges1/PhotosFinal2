"""Detecção local de rostos com OpenCV e cálculo de cortes seguros."""

from __future__ import annotations

import logging
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

FaceBox = tuple[int, int, int, int]
CropBox = tuple[int, int, int, int]
FaceDetectionBox = dict[str, object]

logger = logging.getLogger(__name__)
_FACE_CASCADE: cv2.CascadeClassifier | None = None


def _get_face_cascade() -> cv2.CascadeClassifier | None:
    global _FACE_CASCADE
    if _FACE_CASCADE is not None:
        return _FACE_CASCADE

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        logger.warning("Nao foi possivel carregar o Haar Cascade em %s", cascade_path)
        return None

    _FACE_CASCADE = cascade
    return _FACE_CASCADE


def _resize_image_for_detection(image: Image.Image, max_side: int = 800) -> tuple[Image.Image, float]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    largest = max(width, height)
    if largest <= max_side:
        return rgb, 1.0

    scale = max_side / largest
    resized_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return rgb.resize(resized_size, Image.Resampling.LANCZOS), 1 / scale


def detect_faces_mediapipe(image: Image.Image) -> list[FaceDetectionBox]:
    """Detecta rostos com MediaPipe e retorna caixas em pixels da imagem original."""
    try:
        import mediapipe as mp
    except Exception:
        logger.info("MediaPipe nao disponivel para deteccao de rostos")
        return []

    try:
        face_detection_module = getattr(getattr(mp, "solutions", None), "face_detection", None)
        if face_detection_module is None:
            logger.info("MediaPipe sem modulo solutions.face_detection; usando fallback OpenCV")
            return []

        resized, scale_back = _resize_image_for_detection(image)
        rgb_array = np.array(resized.convert("RGB"))
        with face_detection_module.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5,
        ) as detector:
            results = detector.process(rgb_array)
    except Exception:
        logger.info("MediaPipe falhou na deteccao de rostos; usando fallback OpenCV")
        return []

    detections = getattr(results, "detections", None)
    if not detections:
        return []

    image_width, image_height = image.size
    boxes: list[FaceDetectionBox] = []
    for detection in detections:
        try:
            relative_box = detection.location_data.relative_bounding_box
            confidence = float(detection.score[0]) if detection.score else 0.0
            x = int(round(relative_box.xmin * resized.width * scale_back))
            y = int(round(relative_box.ymin * resized.height * scale_back))
            width = int(round(relative_box.width * resized.width * scale_back))
            height = int(round(relative_box.height * resized.height * scale_back))
            x = max(0, min(x, image_width - 1))
            y = max(0, min(y, image_height - 1))
            width = max(1, min(width, image_width - x))
            height = max(1, min(height, image_height - y))
            boxes.append(
                {
                    "x": x,
                    "y": y,
                    "w": width,
                    "h": height,
                    "confidence": confidence,
                    "source": "mediapipe",
                }
            )
        except Exception:
            logger.exception("Falha ao converter caixa do MediaPipe")
            continue

    return sorted(boxes, key=lambda box: (int(box["y"]), int(box["x"])))


def detect_faces_opencv(image: Image.Image) -> list[FaceDetectionBox]:
    """Detecta rostos com Haar Cascade do OpenCV."""
    cascade = _get_face_cascade()
    if cascade is None:
        return []

    try:
        rgb, scale_back = _resize_image_for_detection(image)
        image_array = np.array(rgb)
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
    except Exception:
        logger.info("OpenCV falhou na deteccao de rostos; seguindo sem rostos")
        return []

    boxes: list[FaceDetectionBox] = []
    image_width, image_height = image.size
    for x, y, width, height in faces:
        scaled_x = max(0, int(round(int(x) * scale_back)))
        scaled_y = max(0, int(round(int(y) * scale_back)))
        scaled_width = max(1, int(round(int(width) * scale_back)))
        scaled_height = max(1, int(round(int(height) * scale_back)))
        scaled_width = min(scaled_width, image_width - scaled_x)
        scaled_height = min(scaled_height, image_height - scaled_y)
        boxes.append(
            {
                "x": scaled_x,
                "y": scaled_y,
                "w": scaled_width,
                "h": scaled_height,
                "confidence": 0.0,
                "source": "opencv",
            }
        )
    return sorted(boxes, key=lambda box: (int(box["y"]), int(box["x"])))


def detect_faces_local(image: Image.Image) -> list[FaceDetectionBox]:
    """Tenta MediaPipe, depois OpenCV, e nunca deixa a app quebrar."""
    try:
        mediapipe_faces = detect_faces_mediapipe(image)
        if mediapipe_faces:
            return mediapipe_faces
    except Exception:
        logger.info("MediaPipe indisponivel; usando fallback OpenCV")

    try:
        return detect_faces_opencv(image)
    except Exception:
        logger.info("Fallback OpenCV indisponivel; seguindo sem deteccao de rostos")
        return []


def detect_faces(image: Image.Image) -> list[FaceBox]:
    """Retorna caixas de rostos no formato (x, y, largura, altura)."""
    boxes: list[FaceBox] = []
    for box in detect_faces_local(image):
        boxes.append((int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])))
    return sorted(boxes, key=lambda box: (box[1], box[0]))


def get_faces_union_box(faces: Iterable[FaceBox]) -> FaceBox | None:
    faces_list = list(faces)
    if not faces_list:
        return None

    left = min(x for x, _, _, _ in faces_list)
    top = min(y for _, y, _, _ in faces_list)
    right = max(x + width for x, _, width, _ in faces_list)
    bottom = max(y + height for _, y, _, height in faces_list)
    return (left, top, right - left, bottom - top)


def expand_box(
    box: FaceBox,
    margin_percent: float,
    image_width: int,
    image_height: int,
) -> FaceBox:
    x, y, width, height = box
    margin_x = width * margin_percent
    margin_y = height * margin_percent

    left = max(0, int(round(x - margin_x)))
    top = max(0, int(round(y - margin_y)))
    right = min(image_width, int(round(x + width + margin_x)))
    bottom = min(image_height, int(round(y + height + margin_y)))
    return (left, top, max(1, right - left), max(1, bottom - top))


def _contains_with_margin(crop_box: CropBox, face: FaceBox) -> bool:
    crop_x, crop_y, crop_width, crop_height = crop_box
    face_x, face_y, face_width, face_height = face

    crop_left = crop_x
    crop_top = crop_y
    crop_right = crop_x + crop_width
    crop_bottom = crop_y + crop_height

    face_left = face_x
    face_top = face_y
    face_right = face_x + face_width
    face_bottom = face_y + face_height

    margin_left = max(2, int(round(face_width * 0.10)))
    margin_right = max(2, int(round(face_width * 0.10)))
    margin_top = max(2, int(round(face_height * 0.15)))
    margin_bottom = max(2, int(round(face_height * 0.10)))

    return (
        face_left >= crop_left + margin_left
        and face_right <= crop_right - margin_right
        and face_top >= crop_top + margin_top
        and face_bottom <= crop_bottom - margin_bottom
    )


def is_box_inside_crop(
    box: FaceBox,
    crop_box: CropBox,
    safety_margin_px: int = 0,
) -> bool:
    """Confere se uma caixa está dentro do corte com margem opcional."""
    box_x, box_y, box_width, box_height = box
    crop_x, crop_y, crop_width, crop_height = crop_box
    return (
        box_x >= crop_x + safety_margin_px
        and box_y >= crop_y + safety_margin_px
        and box_x + box_width <= crop_x + crop_width - safety_margin_px
        and box_y + box_height <= crop_y + crop_height - safety_margin_px
    )


def is_crop_safe_for_faces(
    crop_box: CropBox,
    faces: Iterable[FaceBox],
    image_width: int,
    image_height: int,
) -> bool:
    """Valida se o corte preserva todos os rostos com folga."""
    faces_list = list(faces)
    if not faces_list:
        return True

    crop_x, crop_y, crop_width, crop_height = crop_box
    if crop_width <= 0 or crop_height <= 0:
        return False
    if crop_x < 0 or crop_y < 0:
        return False
    if crop_x + crop_width > image_width or crop_y + crop_height > image_height:
        return False

    for face in faces_list:
        if not _contains_with_margin(crop_box, face):
            return False

    union = get_faces_union_box(faces_list)
    if union is None:
        return True

    margin = 0.35 if len(faces_list) > 1 else 0.25
    _, _, raw_union_width, raw_union_height = union
    desired_union_width = raw_union_width * (1 + margin * 2)
    desired_union_height = raw_union_height * (1 + margin * 2)
    if desired_union_width > crop_width or desired_union_height > crop_height:
        return False

    expanded_union = expand_box(union, margin, image_width, image_height)
    union_x, union_y, union_width, union_height = expanded_union

    crop_right = crop_x + crop_width
    crop_bottom = crop_y + crop_height
    union_right = union_x + union_width
    union_bottom = union_y + union_height

    if union_x < crop_x or union_y < crop_y:
        return False
    if union_right > crop_right or union_bottom > crop_bottom:
        return False

    if union_width > crop_width * 0.95 or union_height > crop_height * 0.95:
        return False

    return True


def calculate_smart_crop_box(
    image_size: tuple[int, int],
    target_aspect_ratio: float,
    faces: Iterable[FaceBox],
) -> CropBox:
    """Calcula um crop com a proporção final, priorizando o grupo de rostos."""
    image_width, image_height = image_size
    image_aspect = image_width / image_height

    if image_aspect > target_aspect_ratio:
        crop_height = image_height
        crop_width = int(round(crop_height * target_aspect_ratio))
    else:
        crop_width = image_width
        crop_height = int(round(crop_width / target_aspect_ratio))

    crop_width = max(1, min(crop_width, image_width))
    crop_height = max(1, min(crop_height, image_height))

    faces_list = list(faces)
    if not faces_list:
        return (
            int(round((image_width - crop_width) / 2)),
            int(round((image_height - crop_height) / 2)),
            crop_width,
            crop_height,
        )

    union = get_faces_union_box(faces_list)
    if union is None:
        return (0, 0, crop_width, crop_height)

    union_x, union_y, union_width, union_height = union
    if len(faces_list) == 1:
        center_x = union_x + union_width / 2
        face_center_y = union_y + union_height / 2
        crop_x = center_x - crop_width / 2
        crop_y = face_center_y - crop_height * 0.42
        expanded = expand_box(union, 0.60, image_width, image_height)
    else:
        center_x = union_x + union_width / 2
        center_y = union_y + union_height / 2
        crop_x = center_x - crop_width / 2
        crop_y = center_y - crop_height / 2
        expanded = expand_box(union, 0.40, image_width, image_height)

    exp_x, exp_y, exp_width, exp_height = expanded
    exp_right = exp_x + exp_width
    exp_bottom = exp_y + exp_height

    if exp_width <= crop_width:
        if exp_x < crop_x:
            crop_x = exp_x
        if exp_right > crop_x + crop_width:
            crop_x = exp_right - crop_width

    if exp_height <= crop_height:
        if exp_y < crop_y:
            crop_y = exp_y
        if exp_bottom > crop_y + crop_height:
            crop_y = exp_bottom - crop_height

    crop_x = max(0, min(int(round(crop_x)), image_width - crop_width))
    crop_y = max(0, min(int(round(crop_y)), image_height - crop_height))

    return (crop_x, crop_y, crop_width, crop_height)
