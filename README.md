# Skin Lesion Classification & Body-Site Identification (HAM10000)

Deep-learning prototype that classifies dermatoscopic images into the **7 HAM10000 lesion categories** and, as a separate task, predicts the **anatomical site** of the lesion. Built in Google Colab with TensorFlow/Keras, using transfer learning on EfficientNetB3.

**Test accuracy: 82.8%** (7 classes, lesion-level held-out test set, test-time augmentation). The project target was ≥ 80%.

> ⚠️ **Educational and research prototype — not a medical device.** It has not been clinically validated. It only knows the 7 HAM10000 categories and will always answer with one of them, even for unrelated images. Never use it to diagnose a lesion or rule one out.

---

## Highlights

- **Leak-free evaluation.** Data is split 70 / 15 / 15 by `lesion_id`, stratified by diagnosis, so every image of a lesion lands in the same split. Assertions check that no lesion, image or byte-identical file appears in two splits.
- **Model selection on validation only.** The notebook can compare EfficientNetB0, EfficientNetB3, MobileNetV2 and ResNet50 on the same schedule and retrain the best one by validation accuracy. Test-time augmentation is enabled only when it improves validation accuracy, which it did for the released model. The test set is used once.
- **Gradual fine-tuning.** Training runs in stages: head only, then the top 30% of the backbone, then the full network at a low learning rate. BatchNorm stays frozen. The pipeline also uses augmentation, dropout + L2, early stopping, LR-on-plateau, √ class weighting and label smoothing.
- **Explainability.** Grad-CAM heatmaps over the last convolutional layer.
- **Interface.** Upload an image in Colab and get the predicted disease with its model confidence, the top-3 predictions, the predicted body site with its confidence, and a Grad-CAM overlay.

## Results

| Task | Model | Input | Classes | Test accuracy |
|---|---|---|---|---|
| Lesion classification | EfficientNetB3 (ImageNet → fine-tuned) + TTA | 300×300 | 7 | **82.8%** |
| Body-site classification | EfficientNetB3 (ImageNet → fine-tuned) | 300×300 | 13 | see notebook report |

For context, always predicting the majority class (`nv`) scores about 67%. Accuracy should therefore be read together with macro-F1, balanced accuracy and per-class recall, especially melanoma recall. The notebook produces all of these, along with confusion matrices, ROC-AUC curves, training curves, an architecture comparison table and `performance_report.md`.

## Classes

| Code | Lesion | Share of dataset |
|---|---|---|
| `nv` | Melanocytic nevi | ~67% |
| `mel` | Melanoma | ~11% |
| `bkl` | Benign keratosis-like lesions | ~11% |
| `bcc` | Basal cell carcinoma | ~5% |
| `akiec` | Actinic keratoses / intraepithelial carcinoma | ~3% |
| `vasc` | Vascular lesions | ~1.4% |
| `df` | Dermatofibroma | ~1.1% |

Body-site classes: abdomen, back, chest, ear, face, foot, genital, hand, lower extremity, neck, scalp, trunk, upper extremity. `unknown` and `acral` were excluded (`acral` has fewer than 40 images).

## Repository structure

```
├── notebooks/
│   └── Skin_Lesion_Classification_HAM10000.ipynb   # full pipeline: data → training → evaluation → Grad-CAM → interface
├── models/
│   ├── disease_EfficientNetB3.keras                # 7-class lesion classifier (~45 MB)
│   ├── location_EfficientNetB3.keras               # 13-class body-site classifier (~45 MB)
│   └── model_config.json                           # classes, input size, preprocessing, TTA flag
├── predict.py                                      # command-line inference with the saved models
├── requirements.txt
└── README.md
```

## Quick start: run the trained models

```bash
git clone https://github.com/Muqeet04/skin-lesion-classification-ham10000.git
cd skin-lesion-classification-ham10000
pip install -r requirements.txt
python predict.py path/to/dermoscopy_image.jpg
```

Example output:

```
path/to/dermoscopy_image.jpg
  1. Melanocytic nevi                               (nv   ) model confidence  91.3%
  2. Benign keratosis-like lesions                  (bkl  ) model confidence   4.2%
  3. Melanoma                                       (mel  ) model confidence   3.1%
  Body site: back (model confidence 41.0%)
```

Inside Python:

```python
import json, numpy as np, keras
from PIL import Image

cfg = json.load(open("models/model_config.json"))["disease"]
model = keras.models.load_model(f"models/{cfg['file']}")
x = np.asarray(Image.open("image.jpg").convert("RGB").resize((cfg["img_size"],) * 2), dtype="float32")[None]
probs = model.predict(x)[0]          # EfficientNet normalises internally; feed raw 0-255 RGB
print(cfg["classes"][probs.argmax()], probs.max())
```

## Reproduce training (Google Colab)

1. Open `notebooks/Skin_Lesion_Classification_HAM10000.ipynb` in Colab and select a **GPU runtime** (a T4 is enough).
2. Create a Kaggle API token (Kaggle → Settings → API). In Colab's 🔑 Secrets panel, add it as `KAGGLE_API_TOKEN` and turn on notebook access. The legacy `KAGGLE_USERNAME` + `KAGGLE_KEY` pair, or uploading `kaggle.json`, also works.
3. **Runtime → Run all.** The dataset (`kmader/skin-cancer-mnist-ham10000`) downloads automatically. Checkpoints, models, figures and reports are saved to Google Drive, and with `RESUME = True` an interrupted run picks up finished models instead of retraining them.

A full run takes about 1.5–3 hours on a T4, most of it spent on the 4-architecture comparison. Set `RUN_ARCH_COMPARISON = False` to train only EfficientNetB3.

### Notebook sections

0 Execution strategy & risks · 1 Setup & config · 2 Google Drive · 3 Kaggle download · 4 Integrity checks (missing / corrupted / duplicate files) · 5 EDA · 6 Lesion-level split · 7 Class weighting · 8 `tf.data` + augmentation · 9 Model builder & trainer · 10 Architecture comparison · 11 Final training · 12 TTA decision · 13 Test evaluation · 14 Troubleshooting if below target · 15 Body-site model · 16 Grad-CAM · 17 Save / reload models · 18 Upload & predict interface · 19 Auto-generated report · 20 Limitations

## Limitations

- **Dataset scope.** HAM10000 is dermoscopy from a few centres, mainly Vienna and Queensland. About half the labels are histopathology-confirmed; the rest come from follow-up or expert consensus. Performance on phone photos, other devices or other populations is unknown.
- **Imbalance.** `nv` dominates, while `df` and `vasc` are about 1% each, so minority-class metrics are noisy.
- **Bias.** Patients are predominantly fair-skinned, and skin type is not recorded. Artefacts such as rulers, ink, hair and vignetting can act as shortcuts.
- **Lesion vs. body site.** The diagnosis is visible in the lesion itself. The body site is metadata about where the photo was taken and is barely visible in a close-up, so the site model is much weaker and secondary.
- **Confidence ≠ diagnostic certainty.** Softmax scores are uncalibrated relative scores over forced choices, and they can be high for out-of-distribution images.
- **Grad-CAM** shows which regions influenced a prediction. It is not proof the prediction is medically correct.

## Dataset & citation

Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Sci. Data 5, 180161 (2018). Kaggle mirror: [kmader/skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000). The dataset is released under CC BY-NC 4.0 and is **not** included in this repository.

## Tech stack

Python · TensorFlow / Keras 3 · scikit-learn · pandas · NumPy · Matplotlib / Seaborn · Pillow · Google Colab (T4 GPU) · Kaggle API
