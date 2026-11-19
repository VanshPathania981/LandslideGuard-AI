
# LandslideGuard AI - Testing Results

## Functional Testing

| Test Case | Expected Result | Actual Result | Status |
|---|---|---|---|
| Valid HDF5 upload | Successful prediction | | |
| Multiple image uploads | Successful predictions | | |
| Threshold adjustment | Mask changes | | |
| Invalid file upload | Error message | | |
| Mask download | PNG downloaded | | |

## Model Configuration

- Architecture: U-Net
- Input Channels: 14
- Input Resolution: 128 × 128
- Prediction Threshold: 0.70
- Best Training Epoch: 7

## Validation Metrics

- Dice Score: 72.63%
- Landslide IoU: 57.03%
- Precision: 70.80%
- Recall: 74.56%
