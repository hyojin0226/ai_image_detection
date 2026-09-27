"""Convert the final Keras model to a TensorFlow.js graph model for web/.

Usage (Python 3.10+, tensorflow==2.21.0, tensorflowjs installed with --no-deps):
    python scripts/convert_to_tfjs.py <path/to/efficientnetv2s_fake_face_final.keras> web/model

What it does:
1. Removes config keys written by newer Keras (renorm*, quantization_config) so older Keras can load it.
2. Rebuilds the inference graph without the data_augmentation block (identity at inference,
   but its random seed state blocks SavedModel export).
3. Checks the rebuilt model gives the same output as the original.
4. Exports a SavedModel and converts it to a float16-quantized TF.js graph model.
"""
import json
import shutil
import sys
import tempfile
import types
import zipfile
from pathlib import Path

import numpy as np

UNSUPPORTED_KEYS = {"renorm", "renorm_clipping", "renorm_momentum", "quantization_config"}


def strip_keys(obj):
    if isinstance(obj, dict):
        return {k: strip_keys(v) for k, v in obj.items() if k not in UNSUPPORTED_KEYS}
    if isinstance(obj, list):
        return [strip_keys(v) for v in obj]
    return obj


def patch_keras_file(src, dst):
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "config.json":
                data = json.dumps(strip_keys(json.loads(data))).encode()
            zout.writestr(item, data)


def build_inference_model(model):
    import keras

    layer = model.get_layer
    x = keras.Input((256, 256, 3), name="image")
    features = layer("efficientnetv2-s")(x, training=False)
    h = layer("pooled")([layer("gap")(features), layer("gmp")(features)])
    h = layer("dense_256")(h)
    h = layer("bn")(h, training=False)
    h = layer("relu")(h)
    h = layer("dropout_1")(h, training=False)
    return keras.Model(x, layer("probability_real")(h))


def main(keras_path, out_dir):
    import keras
    import tensorflow as tf

    work = Path(tempfile.mkdtemp())
    patched = work / "patched.keras"
    patch_keras_file(keras_path, patched)

    original = keras.models.load_model(patched, compile=False)
    inference = build_inference_model(original)

    sample = np.random.default_rng(0).uniform(0, 255, (4, 256, 256, 3)).astype("float32")
    diff = np.abs(original.predict(sample, verbose=0) - inference.predict(sample, verbose=0)).max()
    print(f"rebuilt model max diff: {diff}")
    assert diff < 1e-5, "rebuilt model does not match the original"

    class Serving(tf.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        @tf.function(input_signature=[tf.TensorSpec([None, 256, 256, 3], tf.float32, name="image")])
        def serve(self, image):
            return {"probability_real": self.model(image, training=False)}

    saved_dir = work / "saved"
    serving = Serving(inference)
    tf.saved_model.save(serving, str(saved_dir), signatures={"serving_default": serving.serve})

    # The converter imports these optional packages at module load but doesn't need them here.
    for name in ["tensorflow_decision_forests", "jax", "jax.experimental", "jax.experimental.jax2tf", "flax"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    from tensorflowjs.converters import converter

    out_dir = Path(out_dir)
    for old in out_dir.glob("group1-shard*.bin"):
        old.unlink()
    converter.convert([
        "--input_format=tf_saved_model", "--output_format=tfjs_graph_model",
        "--signature_name=serving_default", "--saved_model_tags=serve",
        "--quantize_float16=*", str(saved_dir), str(out_dir),
    ])
    shutil.rmtree(work, ignore_errors=True)
    print(f"TF.js model written to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
