import cv2
import numpy as np


def preprocess_image(image):

    # 1. Convert input into NumPy array
    image = np.array(image)

    # 2. Convert RGB/RGBA to grayscale
    if len(image.shape) == 3:

        if image.shape[2] == 4:
            image = cv2.cvtColor(
                image, cv2.COLOR_RGBA2GRAY
            )
        else:
            image = cv2.cvtColor(
                image, cv2.COLOR_RGB2GRAY
            )

    if len(image.shape) != 2:
        raise ValueError("Invalid image format.")

    # 3. Ensure uint8 format
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # 4. Detect background using image borders
    border = np.concatenate([
        image[0, :],
        image[-1, :],
        image[:, 0],
        image[:, -1]
    ])

    # Make digit foreground white
    if np.median(border) > 127:
        image = 255 - image

    # 5. Reduce noise
    image = cv2.GaussianBlur(
        image, (3, 3), 0
    )

    # 6. Apply Otsu thresholding
    _, binary = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 7. Find digit boundaries
    points = cv2.findNonZero(binary)

    if points is None:
        raise ValueError("No handwritten digit detected.")

    x, y, w, h = cv2.boundingRect(points)

    digit = binary[y:y+h, x:x+w]

    # 8. Resize while preserving aspect ratio
    scale = 20.0 / max(w, h)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interpolation = (
        cv2.INTER_AREA if scale < 1
        else cv2.INTER_CUBIC
    )

    digit = cv2.resize(
        digit,
        (new_w, new_h),
        interpolation=interpolation
    )

    # 9. Place digit on a 28x28 canvas
    canvas = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = digit

    # 10. Centre digit using its centre of mass
    moments = cv2.moments(canvas)

    if moments["m00"] > 0:

        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]

        target_x = 13.5
        target_y = 13.5

        shift_x = target_x - center_x
        shift_y = target_y - center_y

        transformation = np.float32([
            [1, 0, shift_x],
            [0, 1, shift_y]
        ])

        canvas = cv2.warpAffine(
            canvas,
            transformation,
            (28, 28),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

    # 11. Normalize pixel values
    canvas = canvas.astype("float32") / 255.0

    # 12. Prepare CNN input
    canvas = canvas.reshape(1, 28, 28, 1)

    return canvas