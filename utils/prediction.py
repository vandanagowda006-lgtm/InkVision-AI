import os
import numpy as np
import tensorflow as tf

MODEL_PATH = "model/digit_cnn.keras"


def load_model():
    """Load the trained InkVision CNN."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Trained model not found. Run train_model.py first."
        )

    return tf.keras.models.load_model(MODEL_PATH)


def predict_digit(model, processed_image):
    """Return the three most likely digit classes."""
    predictions = model.predict(processed_image, verbose=0)[0]
    top_3_indices = np.argsort(predictions)[-3:][::-1]

    return [
        {
            "digit": int(index),
            "confidence": round(float(predictions[index]) * 100, 2)
        }
        for index in top_3_indices
    ]


def _validate_single_image(model, processed_image, purpose="prediction"):
    """Validate one preprocessed image against the loaded model's input shape."""
    image = np.asarray(processed_image, dtype=np.float32)

    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(
            f"{purpose} expects one image in batch format, "
            f"for example (1, 28, 28, 1). Received: {image.shape}"
        )

    if not np.isfinite(image).all():
        raise ValueError("The processed image contains invalid values.")

    expected_shape = tuple(model.input_shape[1:])
    if tuple(image.shape[1:]) != expected_shape:
        raise ValueError(
            f"Model expects an image shaped like {expected_shape}, "
            f"but received {tuple(image.shape[1:])}."
        )

    return image


def _apply_inference_layer(layer, tensor):
    """Apply an existing trained layer in inference mode."""
    try:
        return layer(tensor, training=False)
    except TypeError:
        return layer(tensor)


def _build_gradcam_model(model):
    """
    Rebuild a differentiable view of the existing Sequential CNN.

    Existing layer objects are reused, so their trained weights are not
    changed and the model is not retrained.
    """
    if not model.layers:
        raise ValueError("The loaded CNN has no layers.")

    input_shape = tuple(model.input_shape[1:])
    if not input_shape or any(dimension is None for dimension in input_shape):
        raise ValueError(
            f"Unsupported model input shape for Grad-CAM: {model.input_shape}"
        )

    inputs = tf.keras.Input(shape=input_shape, name="gradcam_input")
    x = inputs
    last_conv_output = None

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        x = _apply_inference_layer(layer, x)

        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_output = x

    if last_conv_output is None:
        raise ValueError("No Conv2D layer was found in the loaded model.")

    return tf.keras.Model(
        inputs=inputs,
        outputs=[last_conv_output, x],
        name="inkvision_gradcam_model"
    )


def generate_gradcam(model, processed_image, class_index=None):
    """
    Generate a Grad-CAM heatmap from the final Conv2D layer.

    Returns a 2D float32 heatmap normalized to 0..1. Grad-CAM is an
    approximate visual explanation, not a pixel-perfect explanation.
    """
    image = _validate_single_image(
        model, processed_image, purpose="Grad-CAM"
    )

    grad_model = _build_gradcam_model(model)
    input_tensor = tf.convert_to_tensor(image)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(
            input_tensor,
            training=False
        )

        if class_index is None:
            class_index = int(tf.argmax(predictions[0]).numpy())

        class_index = int(class_index)
        if class_index < 0 or class_index >= int(predictions.shape[-1]):
            raise ValueError(
                "The selected class index is outside the model output range."
            )

        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, conv_outputs)

    if gradients is None:
        raise RuntimeError(
            "Gradients are unavailable. The selected class output is not "
            "connected to the final convolutional feature maps."
        )

    channel_weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    feature_maps = conv_outputs[0]

    heatmap = tf.reduce_sum(
        feature_maps * channel_weights,
        axis=-1
    )
    heatmap = tf.maximum(heatmap, 0.0)

    maximum = tf.reduce_max(heatmap)
    heatmap = tf.math.divide_no_nan(heatmap, maximum)

    result = heatmap.numpy().astype(np.float32)

    if result.ndim != 2 or not np.isfinite(result).all():
        raise RuntimeError("Grad-CAM produced an invalid heatmap.")

    return result


def get_feature_map_layers(model):
    """
    Return the CNN layers that are useful for visual feature exploration.

    The current InkVision CNN contains two Conv2D and two MaxPooling2D
    layers. The function detects them from the actual loaded model rather
    than relying on hard-coded layer numbers.
    """
    supported_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.MaxPooling2D
    )

    selected_layers = []
    for index, layer in enumerate(model.layers):
        if isinstance(layer, supported_types):
            layer_type = (
                "Convolution"
                if isinstance(layer, tf.keras.layers.Conv2D)
                else "Max Pooling"
            )

            selected_layers.append({
                "index": index,
                "name": layer.name,
                "type": layer_type,
                "description": (
                    "Learns visual patterns such as edges, curves and strokes."
                    if layer_type == "Convolution"
                    else "Downsamples feature maps while retaining strong responses."
                )
            })

    if not selected_layers:
        raise ValueError(
            "No Conv2D or MaxPooling2D layers were found in this model."
        )

    return selected_layers


def _normalize_feature_map(feature_map):
    """Normalize one activation map to 0..1 for display."""
    feature_map = np.asarray(feature_map, dtype=np.float32)

    if not np.isfinite(feature_map).all():
        feature_map = np.nan_to_num(
            feature_map, nan=0.0, posinf=0.0, neginf=0.0
        )

    minimum = float(np.min(feature_map))
    maximum = float(np.max(feature_map))

    if maximum - minimum < 1e-8:
        return np.zeros_like(feature_map, dtype=np.float32)

    return ((feature_map - minimum) / (maximum - minimum)).astype(np.float32)


def generate_feature_maps(model, processed_image, layer_name):
    """
    Extract real activations from a selected Conv2D or MaxPooling2D layer.

    Parameters
    ----------
    model:
        The loaded, trained InkVision CNN.
    processed_image:
        One CNN-ready image with shape (1, height, width, channels).
    layer_name:
        Exact layer name returned by get_feature_map_layers(model).

    Returns
    -------
    dict:
        layer_name, layer_type, output_shape, filter_count,
        feature_maps (normalized 2D arrays), and raw_feature_maps.

    This performs inference using existing weights. It does not retrain
    or modify the CNN.
    """
    image = _validate_single_image(
        model, processed_image, purpose="Feature-map extraction"
    )

    target_layer = None
    for layer in model.layers:
        if layer.name == layer_name:
            target_layer = layer
            break

    if target_layer is None:
        raise ValueError(f"Layer '{layer_name}' was not found in the model.")

    if not isinstance(
        target_layer,
        (tf.keras.layers.Conv2D, tf.keras.layers.MaxPooling2D)
    ):
        raise ValueError(
            f"Layer '{layer_name}' is not a supported feature-map layer."
        )

    inputs = tf.keras.Input(
        shape=tuple(model.input_shape[1:]),
        name="feature_explorer_input"
    )
    x = inputs
    target_output = None

    # Replay the original model's layers and capture the selected layer.
    # The original layer instances (and their trained weights) are reused.
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        x = _apply_inference_layer(layer, x)

        if layer.name == layer_name:
            target_output = x
            break

    if target_output is None:
        raise RuntimeError(
            f"Could not create an activation output for '{layer_name}'."
        )

    feature_model = tf.keras.Model(
        inputs=inputs,
        outputs=target_output,
        name=f"inkvision_{layer_name}_feature_model"
    )

    activations = feature_model(
        tf.convert_to_tensor(image),
        training=False
    ).numpy()

    if activations.ndim != 4 or activations.shape[0] != 1:
        raise RuntimeError(
            "The selected layer did not return a single batch of 2D feature maps."
        )

    raw_maps = activations[0].astype(np.float32)
    normalized_maps = [
        _normalize_feature_map(raw_maps[:, :, channel])
        for channel in range(raw_maps.shape[-1])
    ]

    return {
        "layer_name": layer_name,
        "layer_type": (
            "Convolution"
            if isinstance(target_layer, tf.keras.layers.Conv2D)
            else "Max Pooling"
        ),
        "output_shape": tuple(int(value) for value in activations.shape[1:]),
        "filter_count": int(raw_maps.shape[-1]),
        "feature_maps": normalized_maps,
        "raw_feature_maps": raw_maps
    }
