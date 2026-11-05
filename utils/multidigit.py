"""
Multi-digit segmentation utilities for InkVision AI.

This module detects separate handwritten digit regions from one image,
orders them from left to right, and prepares each region using the
existing single-digit preprocessing pipeline.

This first version works best when digits have visible gaps between them.
Touching/overlapping handwriting may need more advanced segmentation.
"""

import cv2
import numpy as np

from utils.preprocessing import preprocess_image


def _to_binary_foreground(image):
    """Convert a PIL/NumPy image to a binary image with white foreground."""
    array = np.asarray(image)

    if array.ndim == 3:
        if array.shape[2] == 4:
            gray = cv2.cvtColor(array, cv2.COLOR_RGBA2GRAY)
        elif array.shape[2] == 3:
            gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError("Unsupported image channel format.")
    elif array.ndim == 2:
        gray = array.copy()
    else:
        raise ValueError("Upload a grayscale, RGB, or RGBA image.")

    if gray.dtype != np.uint8:
        if np.issubdtype(gray.dtype, np.floating) and gray.max(initial=0) <= 1:
            gray = gray * 255
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    if gray.size == 0:
        raise ValueError("The uploaded image is empty.")

    # Use the image border to determine whether the background is light.
    border = np.concatenate([
        gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
    ])
    if np.median(border) > 127:
        gray = 255 - gray

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Remove tiny isolated noise components.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    cleaned = np.zeros_like(binary)
    image_area = binary.shape[0] * binary.shape[1]
    min_area = max(3, int(image_area * 0.00005))

    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area >= min_area:
            cleaned[labels == label] = 255

    if cv2.countNonZero(cleaned) == 0:
        raise ValueError("No handwritten digits were detected.")

    return gray, cleaned


def _merge_digit_fragments(boxes, image_height):
    """
    Merge nearby components that are likely separate strokes of one digit.

    Components are merged when their horizontal gap is small and their
    vertical ranges overlap substantially. This helps with disconnected
    strokes, but does not guarantee separation of touching digits.
    """
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda box: (box[0], box[1]))
    merged = []
    gap_limit = max(3, int(image_height * 0.10))

    for box in boxes:
        x, y, w, h = box
        current = [x, y, x + w, y + h]
        changed = True

        while changed:
            changed = False
            remaining = []

            for old in merged:
                ox, oy, oright, obottom = old
                horizontal_gap = max(0, max(ox - current[2], current[0] - oright))
                overlap_y = max(0, min(current[3], obottom) - max(current[1], oy))
                min_height = max(1, min(current[3] - current[1], obottom - oy))
                vertical_overlap = overlap_y / min_height

                if horizontal_gap <= gap_limit and vertical_overlap >= 0.35:
                    current = [
                        min(current[0], ox),
                        min(current[1], oy),
                        max(current[2], oright),
                        max(current[3], obottom)
                    ]
                    changed = True
                else:
                    remaining.append(old)

            merged = remaining

        merged.append(current)

    return [
        (x1, y1, x2 - x1, y2 - y1)
        for x1, y1, x2, y2 in sorted(merged, key=lambda item: item[0])
    ]


def segment_digits(image):
    """
    Detect handwritten digit regions and return them left-to-right.

    Returns a list of dictionaries:
      {
        "index": 1-based position,
        "box": (x, y, width, height),
        "crop": cropped grayscale source image
      }
    """
    gray, binary = _to_binary_foreground(image)
    image_height, image_width = binary.shape

    count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    boxes = []
    min_height = max(3, int(image_height * 0.08))
    min_area = max(5, int(image_height * image_width * 0.00008))

    for label in range(1, count):
        x, y, width, height, area = stats[label]

        if height < min_height or area < min_area:
            continue

        # Ignore extremely wide, thin artefacts such as scan lines.
        if width > image_width * 0.95 and height < image_height * 0.12:
            continue

        boxes.append((int(x), int(y), int(width), int(height)))

    boxes = _merge_digit_fragments(boxes, image_height)

    if not boxes:
        raise ValueError(
            "No individual digit regions were found. Try a clearer image."
        )

    # Add a small margin around each region without crossing image borders.
    segments = []
    margin_x = max(2, int(image_height * 0.035))
    margin_y = max(2, int(image_height * 0.035))

    for index, (x, y, width, height) in enumerate(boxes, start=1):
        left = max(0, x - margin_x)
        top = max(0, y - margin_y)
        right = min(image_width, x + width + margin_x)
        bottom = min(image_height, y + height + margin_y)

        crop = gray[top:bottom, left:right].copy()
        segments.append({
            "index": index,
            "box": (left, top, right - left, bottom - top),
            "crop": crop
        })

    return segments


def predict_multiple_digits(model, image, predict_digit_function):
    """
    Segment an image, preprocess each region, and predict each digit.

    predict_digit_function is the existing predict_digit function from
    utils.prediction. It is passed in to avoid coupling this module to
    model-loading code.
    """
    segments = segment_digits(image)
    digit_results = []

    for segment in segments:
        processed = preprocess_image(segment["crop"])
        possibilities = predict_digit_function(model, processed)

        digit_results.append({
            "index": segment["index"],
            "box": segment["box"],
            "digit": possibilities[0]["digit"],
            "confidence": possibilities[0]["confidence"],
            "top_3": possibilities,
            "processed_image": processed[0, :, :, 0],
            "crop": segment["crop"]
        })

    recognised_number = "".join(str(item["digit"]) for item in digit_results)

    return {
        "number": recognised_number,
        "digits": digit_results,
        "count": len(digit_results)
    }
