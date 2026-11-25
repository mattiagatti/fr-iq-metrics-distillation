# 🧪 Distilling Full-Reference Image Quality Metrics into Neural No-Reference Surrogates

Train deep-learning models to **predict full-reference (FR) image-quality metrics from a single image**, using a range of popular vision backbones. Given a (possibly degraded) image, the network regresses the value of an FR metric such as SSIM — without ever seeing the reference image at inference time.

---

## ⚠️ Motivation & Limitations — a naive experiment

This repository is a deliberately **naive, exploratory experiment**, and it is worth being upfront about why.

Full-reference metrics such as SSIM, FSIM, or VSI measure the similarity between a **distorted image and its pristine reference** — by definition they are a function of *two* images. Here we instead train a no-reference (NR) model to reproduce that FR value from the **distorted image alone**, discarding the reference.

That makes the task fundamentally ill-posed: the same distorted image could have originated from many different references with different FR scores, so the target is not a deterministic function of the input. The best such a model can do is regress the *expected* metric value given the visible degradation — in effect it learns a **degradation-severity estimator**, not the FR metric itself.

The consequences are visible throughout the results:

- Predictions hold up only while the test-time degradations resemble the (synthetic) ones seen during training. The `our` vs `their` degradation splits and the many cross-dataset folders (CSIQ, LIVE, TID2013, KonIQ-10k) exist precisely to probe this, and accuracy drops off outside the training distribution.
- Reproducing a synthetic FR target is **not** the same as predicting human-perceived quality (`mos`). The resulting model is neither a genuine FR metric (no reference at inference) nor a proper perceptual NR model (not trained on human opinion scores) — it sits awkwardly between the two.

In short, treat this as a feasibility probe — *how far can a single-image network mimic an FR metric?* — rather than a usable quality metric. It is an interesting thing to try, but naive by construction.

---

## ⚙️ Requirements

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 📂 Dataset Generation

```bash
python misc/generate_dataset.py --min-degradations <value> --max-degradations <value> --input_path /path/to/imagenet
```

Generates degraded datasets from the ImageNet dataset located at `/path/to/imagenet`. The output is written to `/path/to/imagenet_degraded/` with `train`, `val`, and `test` subdirectories. `--min-degradations` and `--max-degradations` set the range of degradations applied to each image; the FR metric computed between each original and its degraded version becomes the regression target.

---

## 🚀 Training

Each backbone outputs a single scalar (the predicted metric), trained against the precomputed FR ground-truth values.

### Command-line arguments

| Argument        | Description                                                                                              |
| --------------- | -------------------------------------------------------------------------------------------------------- |
| `--model`       | Backbone to use: `swinv2`, `mobilevitv2`, `resnet50v2`, `vit`, `efficientnet`, `mobilenetv3`, `tinyvit`  |
| `--metric`      | Metric to predict: `ssim`, `fsim`, `ms_ssim`, `iw_ssim`, `sr_sim`, `vsi`, `dss`, `haarpsi`, `mdsi`, `mos` |
| `--config_path` | Path to the experiment configuration file (see `experiments/`)                                           |

### Examples (all backbones)

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model mobilevitv2  --metric SSIM --config_path train_IMAGENET_our.yaml
CUDA_VISIBLE_DEVICES=1 python train.py --model resnet50v2   --metric SSIM --config_path train_IMAGENET_our.yaml
CUDA_VISIBLE_DEVICES=2 python train.py --model vit          --metric SSIM --config_path train_IMAGENET_our.yaml
CUDA_VISIBLE_DEVICES=3 python train.py --model efficientnet --metric SSIM --config_path train_IMAGENET_our.yaml
CUDA_VISIBLE_DEVICES=4 python train.py --model mobilenetv3  --metric SSIM --config_path train_IMAGENET_our.yaml
CUDA_VISIBLE_DEVICES=5 python train.py --model tinyvit      --metric SSIM --config_path train_IMAGENET_our.yaml
```

---

## 🧪 Testing

There are two evaluation entry points, depending on what you compare the predictions against.

### Against the FR-metric target — `test.py`

Scores the predicted value against the **true full-reference metric** (the same quantity used as the training target). This is the in-distribution / cross-degradation evaluation on the synthetic ImageNet datasets.

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model <model_name> --metric SSIM --config_path test_IMAGENETour_IMAGENETour.yaml
```

Experiment configs follow the naming convention `<TrainDegradation>_<TestDegradation>` (see
`test_results/directory_derscription.txt`), where `our` denotes a random number of degradation
techniques and `their` denotes a single degradation applied at one of five levels.

### Against human MOS — `test_nr.py`

Evaluates the trained surrogate on **real no-reference IQA datasets** (KonIQ-10k, LIVE in-the-wild) and correlates its output with **human Mean Opinion Scores (MOS)** rather than an FR target. In other words, it asks whether a network trained to mimic an FR metric actually tracks perceived quality — the central question raised in [Motivation & Limitations](#️-motivation--limitations--a-naive-experiment).

```bash
# Correlate predictions with human MOS on KonIQ-10k
CUDA_VISIBLE_DEVICES=0 python test_nr.py --model tinyvit --metric mos --config_path test_KonIQ10k_KonIQ10k_MOS.yaml

# Correlate an FR-metric surrogate against MOS on LIVE in-the-wild
CUDA_VISIBLE_DEVICES=0 python test_nr.py --model tinyvit --metric ssim --config_path test_Live_Combined_IQA.yaml
```

The dataset is selected from the config (`Live_NR`, `KonIQ-10k`, or a generic `test` set), and the
output is a hexbin plot of predicted score vs MOS plus the correlation metrics below.

---

## 📊 Metrics

Evaluation reports, for each model and target metric:

- R² score
- MAE (Mean Absolute Error)
- Spearman rank correlation (SROCC)
- Pearson linear correlation (PLCC)

---

## 📜 License

The code is released under the [MIT License](LICENSE).
