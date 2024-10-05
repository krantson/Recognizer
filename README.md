# CAPTCHA Recognizer

## Project Overview

This repository contains a CAPTCHA recognizer that allows users to choose different recognition models for training and testing on a CAPTCHA dataset. Users need to modify the configuration file `config.cfg` initially to set their preferences.

## Configuration

Before training the model, you need to adjust the settings in the `config.cfg` file. Below is the content of the configuration file along with explanations for each section:

```
[TRAIN]
MODEL_PATH: models                # Path where the trained models will be saved
CLOUD_ROOT: /content              # Cloud root directory
CLOUD_MODEL_PATH: /content/drive/MyDrive/models  # Cloud model path
BATCH_SIZE: 4                     # Number of samples processed before the model is updated
NUM_OF_LETTERS: 4                 # Number of letters in the CAPTCHA
EPOCHS: 2                         # Number of epochs for training
EPOCHS_SAVE: 10                   # Save the model every few epochs
IMG_ROWS: 64                      # Image height
IMG_COLS: 192                     # Image width
DATASET_SPLIT: 0.8                # Proportion of dataset to use for training
IMG_FORMAT:                       # Image format (e.g., jpg, png)
CHARSETS: ABCDEFGHIJKLMNOPQRSTUVWXYZ  # Characters used in the CAPTCHA
[DATASET]
DATA_PATH: mcaptcha_samples        # Path to the dataset containing CAPTCHA images
[MODELS]
DeepCAPTCHA: 0                    # Set to 1 to use DeepCAPTCHA model
ConvNet: 1                         # Set to 1 to use ConvNet model
CAPSULE: 0                         # Set to 1 to use CAPSULE model
VeriBypasser: 0                    # Set to 1 to use VeriBypasser model
MULTIVIEW: 0                       # Set to 1 to use MULTIVIEW model
[PARAS]
DeepCAPTCHA: CNN                  # Specify CNN for DeepCAPTCHA;  LSTM for AdaptiveCAPTCHA
```

### Model Selection
- To select a specific model, set the corresponding model flag to `1`. For example:
  - Setting `ConvNet: 1` will choose ConvNet as the recognition model.
  - Setting `CAPSULE: 1` will select the CAPSULE model.
  - Using `DeepCAPTCHA: 1` will choose the DeepCAPTCHA model, but additionally, you need to specify the architecture under `[PARAS]`, such as `DeepCAPTCHA: CNN` or `DeepCAPTCHA: LSTM`.

### Dataset Path
The dataset path should be set in the `DATA_PATH` field.

## Training the Model

To train the chosen model, run the following command in the terminal:

```bash
python train.py
```



## Contact

If you have any questions, feel free to contact me (wanxing123321@gmai.com)