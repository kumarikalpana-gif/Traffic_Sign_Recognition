# Traffic Sign Recognition Using Deep Learning

## 1. Project Overview

This project develops and evaluates deep learning models for recognizing German traffic signs using the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

The objective is to classify traffic sign images into one of 43 categories and evaluate model performance under both normal and corrupted image conditions.

Three CNN-based approaches are implemented and compared:

1. **Baseline CNN** — a convolutional neural network trained from scratch.
2. **EfficientNetB0** — an ImageNet-pretrained model using transfer learning and fine-tuning.
3. **CNN with Heavy Augmentation** — the baseline CNN architecture trained using stronger data augmentation.

The Baseline CNN achieved the best overall results and is recommended for the next deployment prototype stage.

## 2. Dataset

**Dataset:** GTSRB — German Traffic Sign Recognition Benchmark

**Kaggle dataset:** `meowmeowmeowmeow/gtsrb-german-traffic-sign`

The dataset contains images belonging to 43 traffic sign classes.

### Dataset split

| Dataset | Number of images |
|---|---:|
| Training | 31,367 |
| Validation | 7,842 |
| Testing | 12,630 |
| **Total** | **51,839** |

The training data was split into training and validation subsets using stratified sampling with a fixed random seed of 42. The official test dataset was retained for final evaluation.

## 3. Project Structure

```text
Traffic_Sign_Recognition/
│
├── .vscode/
│   └── settings.json
│
├── data/
│   ├── raw/
│   │   └── GTSRB/
│   │
│   └── processed/
│       ├── train_metadata.csv
│       ├── validation_metadata.csv
│       ├── test_metadata.csv
│       ├── class_metadata.csv
│       ├── dataset_summary.json
│       ├── phase2_preprocessing_summary.json
│       └── robustness_test/
│           ├── gaussian_blur/
│           ├── brightness/
│           ├── occlusion/
│           └── combined/
│
├── notebooks/
│   ├── 01_Data_Loading_and_EDA.ipynb
│   ├── 02_Preprocessing_and_Augmentation.ipynb
│   ├── 03_Model_Training.ipynb
│   ├── 04_Evaluation_and_Robustness.ipynb
│   └── 05_Deployment_Recommendation.ipynb
│
├── src/
│
├── models/
│   ├── cnn_baseline.keras
│   ├── efficientnet_b0.keras
│   ├── efficientnet_b0_transfer.keras
│   ├── cnn_heavy_augmentation.keras
│   └── baseline_cnn_architecture.png
│
├── outputs/
│   ├── eda/
│   ├── training_history/
│   ├── confusion_matrices/
│   └── robustness/
│
├── reports/
│   └── deployment_recommendation.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 4. Technologies Used

- Python 3.11
- TensorFlow and Keras
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow
- KaggleHub
- Jupyter Notebook
- Visual Studio Code

## 5. Project Workflow

The project was completed in five phases.

### Phase 1: Data Loading and Exploratory Data Analysis

- Downloaded and extracted the GTSRB dataset.
- Loaded training, testing, and metadata CSV files.
- Checked missing values, duplicates, and image paths.
- Performed a stratified training-validation split.
- Analyzed class distributions and image dimensions.
- Generated six exploratory data analysis plots.

EDA outputs are stored in `outputs/eda/`.

### Phase 2: Preprocessing and Data Augmentation

All images were converted to RGB, resized to 64 × 64 pixels, and normalized to the range [0, 1].

Two augmentation configurations were implemented:

- **Standard augmentation:** rotation, translation, zoom, contrast, and brightness adjustments.
- **Heavy augmentation:** stronger transformations using the same augmentation categories.

Four separate corrupted versions of the official test set were also generated:

1. Gaussian blur
2. Brightness changes
3. Occlusion
4. Combined corruption

These datasets were used to assess model robustness.

### Phase 3: Model Development and Training

Three models were trained.

**Model 1 — Baseline CNN**

- Custom CNN architecture trained from scratch.
- Convolutional layers with 32, 64, and 128 filters.
- Dense classification layer with 43 output neurons.
- Softmax activation.
- Categorical cross-entropy loss.

**Model 2 — EfficientNetB0**

- ImageNet-pretrained EfficientNetB0 backbone.
- Transfer learning with a frozen feature extractor.
- Fine-tuning of the upper layers.
- A custom classification head for 43 traffic sign classes.

**Model 3 — CNN with Heavy Augmentation**

- Same CNN architecture as the baseline.
- Trained using the stronger augmentation pipeline.
- Designed to investigate whether stronger augmentation improves robustness.

Early stopping and learning-rate reduction callbacks were used during training.

### Phase 4: Model Evaluation and Robustness Testing

The models were evaluated using:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Per-class classification reports
- Confusion matrices
- Accuracy under corrupted image conditions
- Accuracy drop relative to clean test performance

### Phase 5: Deployment Recommendation

The models were compared using clean-test accuracy, average corrupted-test accuracy, and overall average accuracy.

The Baseline CNN was selected as the recommended model.

## 6. Model Performance

### Clean Test Results

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Baseline CNN | 99.18% | 98.74% | 98.63% | 98.65% |
| EfficientNetB0 | 75.39% | 68.79% | 67.00% | 67.24% |
| CNN with Heavy Augmentation | 99.12% | 98.71% | 98.62% | 98.64% |

### Corrupted Test Accuracy

| Model | Gaussian Blur | Brightness | Occlusion | Combined |
|---|---:|---:|---:|---:|
| Baseline CNN | 99.21% | 98.30% | 88.23% | 85.46% |
| EfficientNetB0 | 74.33% | 74.22% | 44.24% | 42.16% |
| CNN with Heavy Augmentation | 98.74% | 97.85% | 87.29% | 84.19% |

### Overall Comparison

| Model | Clean Accuracy | Average Corrupted Accuracy | Overall Average Accuracy |
|---|---:|---:|---:|
| Baseline CNN | 99.18% | 92.80% | 94.08% |
| EfficientNetB0 | 75.39% | 58.74% | 62.07% |
| CNN with Heavy Augmentation | 99.12% | 92.02% | 93.44% |

The Baseline CNN achieved the highest average accuracy across the corrupted test conditions.

## 7. Recommended Model

**Selected model:** Baseline CNN

**Model artifact:** `models/cnn_baseline.keras`

Reasons for selection:

- Highest clean-test accuracy among the evaluated models.
- Highest average corrupted-test accuracy.
- Highest overall average accuracy.
- Strong performance under Gaussian blur and brightness changes.

However, performance decreases under occlusion and combined corruption. These limitations should be considered before deployment.

## 8. Loading the Trained Model

The recommended model can be loaded using TensorFlow/Keras:

```python
import tensorflow as tf

model = tf.keras.models.load_model(
    "models/cnn_baseline.keras"
)

model.summary()
```

### Making a Prediction

The model expects RGB images resized to 64 × 64 pixels and normalized to [0, 1].

Example:

```python
import numpy as np
import tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model(
    "models/cnn_baseline.keras"
)

image = Image.open("sample_traffic_sign.png").convert("RGB")
image = image.resize((64, 64))

image_array = np.asarray(image, dtype=np.float32) / 255.0
image_array = np.expand_dims(image_array, axis=0)

predictions = model.predict(image_array)

predicted_class = np.argmax(predictions[0])
confidence = np.max(predictions[0])

print("Predicted class:", predicted_class)
print("Confidence:", f"{confidence * 100:.2f}%")
```

The predicted class is an integer from 0 to 42. Use the class metadata to associate class IDs with their corresponding traffic sign labels.

## 9. How to Run the Project

### Step 1: Clone or download the project

Obtain the project files and open the project folder in Visual Studio Code.

### Step 2: Install dependencies

Open a terminal in the project root and run:

```bash
pip install -r requirements.txt
```

### Step 3: Prepare the dataset

Download the GTSRB dataset from Kaggle and place its extracted contents under:

```text
data/raw/GTSRB/
```

The data-loading notebook also contains the dataset download and preparation workflow.

### Step 4: Run the notebooks in order

Execute the notebooks in this sequence:

1. `01_Data_Loading_and_EDA.ipynb`
2. `02_Preprocessing_and_Augmentation.ipynb`
3. `03_Model_Training.ipynb`
4. `04_Evaluation_and_Robustness.ipynb`
5. `05_Deployment_Recommendation.ipynb`


### Running the Streamlit Application

The project includes an interactive Streamlit application that allows users to upload traffic sign images and receive predictions from the trained Baseline CNN model.

**Start the application:**

```bash
python -m streamlit run app.py
```

Streamlit will launch the application in your browser, usually at:

```text
http://localhost:8501
```

**Application features:**

- Upload traffic sign images in JPG, JPEG, or PNG format.
- Preview the uploaded image.
- Predict the traffic sign class using the trained Baseline CNN.
- Display the predicted class ID and traffic sign name.
- Display model confidence.
- Show the top three predicted classes with their confidence scores.

**Model used:** `models/cnn_baseline.keras`

**Image preprocessing:** RGB conversion, resizing to 64 × 64 pixels, and normalization to the range [0, 1].

**Important:** This application is an educational demonstration and is not validated for real-world driving or safety-critical decisions.


Each phase creates outputs and artifacts used by subsequent phases.

**Note:** Model training and robustness testing can require substantial processing time and memory. A compatible GPU can accelerate training, but CPU execution is also possible.

## 10. Limitations and Future Improvements

Although the Baseline CNN achieved excellent results on the evaluated test data, further work is required before real-world use.

Potential improvements include:

- Testing on real-world driving images and video.
- Evaluating rain, fog, shadows, glare, and motion blur.
- Testing signs at different distances and viewing angles.
- Improving resistance to occlusion and combined distortions.
- Adding confidence-based rejection for uncertain predictions.
- Measuring inference speed and memory usage on deployment hardware.
- Exploring model optimization and quantization.

## 11. Conclusion

This project demonstrates an end-to-end traffic sign recognition workflow involving data preparation, exploratory analysis, image preprocessing, augmentation, deep learning model development, evaluation, and robustness testing.

Among the three evaluated models, the Baseline CNN achieved the strongest overall performance, with 99.18% clean-test accuracy and 92.80% average accuracy across corrupted test conditions.

It is therefore recommended as the primary model for a future traffic sign recognition prototype.

**Safety note:** This project is an academic prototype. It has not been validated for safety-critical vehicle control or autonomous driving applications.
