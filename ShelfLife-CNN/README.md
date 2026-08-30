# ShelfLife-CNN

A CNN-based image classifier that determines whether common fruits are **fresh** or **rotten**.

## Classes

10 classes total (freshness x produce type):

| Fresh          | Rotten          |
|----------------|-----------------|
| fresh_apple    | rotten_apple    |
| fresh_banana   | rotten_banana   |
| fresh_mango    | rotten_mango    |
| fresh_orange   | rotten_orange   |
| fresh_tomato   | rotten_tomato   |

## Project Structure

```
ShelfLife-CNN/
├── dataset/
│   ├── train/<class_name>/*.jpg
│   ├── val/<class_name>/*.jpg
│   └── test/<class_name>/*.jpg
├── train.py            # Train the CNN (transfer learning on ResNet18)
├── evaluate.py          # Evaluate a trained model on the test set
├── predict_image.py     # Run inference on a single image
├── predict_webcam.py     # Real-time inference from a webcam
├── class_names.json     # Index -> class name mapping
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Place your images into the appropriate `dataset/train`, `dataset/val`, and
`dataset/test` subfolders, one folder per class (folder name = label).

## Usage

### Train

```bash
python train.py --epochs 15 --batch-size 32 --lr 1e-4
```

This saves the best-performing checkpoint to `shelflife_cnn.pth` and updates
`class_names.json` to match the classes discovered in your dataset.

### Evaluate

```bash
python evaluate.py --model shelflife_cnn.pth --data-dir dataset --plot
```

Prints test accuracy, a per-class classification report, and a confusion
matrix (optionally saved as `confusion_matrix.png`).

### Predict a single image

```bash
python predict_image.py --image path/to/photo.jpg --model shelflife_cnn.pth
```

### Predict from webcam (real-time)

```bash
python predict_webcam.py --model shelflife_cnn.pth --camera 0
```

Press `q` to quit the webcam window.

## Notes

- The model uses a ResNet18 backbone pretrained on ImageNet, with the
  final layer replaced for 10-class classification. The backbone is frozen
  by default for faster training; pass `--unfreeze-backbone` to fine-tune
  the whole network.
- Adjust `--img-size`, `--batch-size`, and `--lr` in `train.py` to tune
  training for your hardware and dataset size.
