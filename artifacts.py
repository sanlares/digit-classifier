"""Names of the model artifacts, in one place.

Imported by the export script, the inference layer and the upload helper, so
renaming an artifact is a one-line change and `upload_weights.py` does not have
to pull in onnxruntime just to learn a filename.
"""

MLP_ONNX = "mlp.onnx"
CNN_ONNX = "cnn.onnx"

# Display name -> artifact filename. Order sets the order of the UI panels.
MODEL_FILES = {
    "MLP": MLP_ONNX,
    "CNN": CNN_ONNX,
}
