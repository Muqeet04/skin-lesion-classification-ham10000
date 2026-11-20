"""Run the trained HAM10000 models on one or more images.

Usage:
    python predict.py path/to/image.jpg [more.jpg ...] [--models models]

Research prototype only - the output is not a diagnosis.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def preprocess(img_path, size, arch):
    x = np.asarray(Image.open(img_path).convert("RGB").resize((size, size), Image.BILINEAR), dtype="float32")[None]
    if arch == "MobileNetV2":
        return x / 127.5 - 1.0
    if arch == "ResNet50":
        return x[..., ::-1] - np.array([103.939, 116.779, 123.68], dtype="float32")
    return x  # EfficientNet models normalise internally


def predict(model, x, tta=False):
    views = [x]
    if tta:  # same views the notebook used: original, h-flip, v-flip, 90 degree rotation
        views += [x[:, :, ::-1], x[:, ::-1], np.rot90(x, k=1, axes=(1, 2))]
    return np.mean([model.predict(np.ascontiguousarray(v), verbose=0)[0] for v in views], axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--models", default="models", help="folder containing model_config.json and the .keras files")
    args = ap.parse_args()

    import keras  # imported late so --help is fast

    mdir = Path(args.models)
    cfg = json.loads((mdir / "model_config.json").read_text())
    dc, lc = cfg["disease"], cfg["location"]
    disease_model = keras.models.load_model(mdir / dc["file"])
    location_model = keras.models.load_model(mdir / lc["file"])

    for path in args.images:
        try:
            p_dx = predict(disease_model, preprocess(path, dc["img_size"], dc["arch"]), dc.get("use_tta", False))
            p_loc = predict(location_model, preprocess(path, lc["img_size"], lc["arch"]))
        except Exception as e:  # unreadable / corrupted image
            print(f"\n{path}: could not process ({e})")
            continue
        top3 = np.argsort(p_dx)[::-1][:3]
        print(f"\n{path}")
        for rank, i in enumerate(top3, 1):
            code = dc["classes"][i]
            print(f"  {rank}. {dc['class_names'][code]:<46} ({code:<5}) model confidence {p_dx[i]:6.1%}")
        li = int(p_loc.argmax())
        print(f"  Body site: {lc['classes'][li]} (model confidence {p_loc[li]:.1%})")
        if p_dx[top3[0]] < 0.5:
            print("  ! Low confidence - image may be ambiguous or unlike the training data.")
    print("\nModel confidence is a softmax score, not the probability of disease. Not a diagnosis.")


if __name__ == "__main__":
    main()
