# InkVision AI
### An Interactive Deep Learning Platform for Handwritten Digit Recognition and Model Interpretability

<p align="center">
  <a href="https://inkvision-ai-vem7rfwvzanp4bkqxcqqhu.streamlit.app/" target="_blank">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open InkVision AI Live Demo">
  </a>
</p>

---

**InkVision AI** is a computer vision application engineered to recognize handwritten numerical digits using a Convolutional Neural Network (CNN). It integrates image preprocessing, deep learning inference, model interpretability, geometric image analysis, and user-specific prediction management into an interactive web platform.

Beyond conventional image classification, InkVision AI makes the recognition pipeline observable, allowing users to explore how handwritten inputs are transformed into machine-readable predictions.

---

## 1. Project Overview

Handwritten digit recognition is a fundamental computer vision problem because handwriting varies in stroke structure, orientation, positioning, thickness, and shape.

InkVision AI addresses this problem using a trained CNN model that classifies handwritten numerical inputs into ten categories, from 0 to 9.

Users can draw digits directly on an interactive canvas or upload handwritten images. The application processes these inputs, performs model inference, and presents predictions alongside visual and analytical insights.

### Core Capabilities

- CNN-based handwritten digit classification.
- Single-digit recognition.
- Segmentation-based multi-digit recognition.
- Prediction confidence and top-three class probabilities.
- Grad-CAM-based visual interpretation.
- Image preprocessing inspection.
- Geometric analysis of handwritten inputs.
- CNN architecture visualization and evaluation reports.
- Authenticated user accounts.
- User-specific prediction history and statistical summaries.

---

## 2. System Modules

| Module | Description |
|---|---|
| Authentication | User registration, login, and secure password hashing. |
| Digit Recognition | CNN-based classification of handwritten numerical digits. |
| Multi-Digit Processing | Segmentation of numerical sequences followed by individual digit classification. |
| Grad-CAM | Visualizes approximate image regions contributing to a model prediction. |
| Digit Autopsy | Displays image preprocessing and transformation stages. |
| Shape DNA | Analyzes geometric characteristics of handwritten digits. |
| Model X-Ray | Explores CNN architecture and training evaluation reports. |
| My Collection | Maintains user-specific prediction records and statistical summaries. |

### 2.1 Recognition Engine

The recognition pipeline accepts handwritten inputs through an interactive drawing canvas or image upload.

The input is converted into a normalized 28 × 28 grayscale representation before being passed to the trained CNN.

The model generates a probability distribution across ten digit classes. The application identifies the highest-probability class and displays the predicted digit, confidence score, and top-three alternatives.

### 2.2 Multi-Digit Recognition

For numerical sequences, the application performs image segmentation to identify individual digit regions.

Each segmented region is independently processed by the CNN, and the resulting predictions are combined into a numerical sequence.

This extends the single-digit classification pipeline without requiring a separate general-purpose OCR engine.

### 2.3 Model Interpretability

InkVision AI integrates Gradient-weighted Class Activation Mapping (Grad-CAM) to generate heatmaps associated with the predicted class.

These visualizations provide an approximate indication of the spatial regions contributing to the model's prediction and support qualitative inspection of model behaviour.

### 2.4 Image Analysis

The platform includes two complementary analytical modules.

**Digit Autopsy**

Examines the transformation of the original handwritten image into the model-ready input representation.

**Shape DNA**

Measures geometric properties such as:

- Bounding width and height.
- Aspect ratio.
- Foreground pixel density.
- Geometric centre.
- Binary shape characteristics.

These measurements provide additional insight into the structural properties of handwritten inputs.

---

## 3. CNN Model Architecture

InkVision AI uses a Sequential Convolutional Neural Network implemented with TensorFlow and Keras.

| Component | Configuration |
|---|---|
| Input Layer | 28 × 28 × 1 |
| Convolutional Layer 1 | 32 filters, 3 × 3, ReLU |
| Max Pooling 1 | 2 × 2 |
| Convolutional Layer 2 | 64 filters, 3 × 3, ReLU |
| Max Pooling 2 | 2 × 2 |
| Flatten | Feature vector |
| Fully Connected Layer | 128 neurons, ReLU |
| Dropout | 0.3 |
| Output Layer | 10 neurons, Softmax |

The architecture progressively extracts spatial features, reduces feature-map dimensions, and transforms learned representations into class probabilities.

The trained model is serialized in Keras format and loaded by the application for inference.

---

## 4. Technical Architecture

The application is organized into four logical layers.

**Application Layer**
- Streamlit interactive interface.
- User authentication and navigation.
- Prediction presentation.
- Visual analytics.

**Processing Layer**
- Image normalization and preprocessing.
- Digit segmentation.
- Geometric feature extraction.
- Grad-CAM generation.

**Machine Learning Layer**
- TensorFlow/Keras CNN.
- Model inference.
- Probability-based classification.

**Data Layer**
- SQLite-based user management.
- Password hash storage.
- User-specific prediction history.

### Recognition Workflow

```text
              User Input
                  |
                  v
            Image Acquisition
             /           \
        Drawing         Upload
             \           /
                  |
                  v
           Image Preprocessing
                  |
                  v
          28 × 28 Grayscale Input
                  |
                  v
             Trained CNN
                  |
                  v
        Probability Distribution
                  |
                  v
            Predicted Digit
                  |
          +-------+-------+
          |       |       |
          v       v       v
        Top-3  Grad-CAM  Prediction
      Results  Analysis  History
```

For multi-digit inputs, an additional segmentation stage identifies individual digit regions before classification.

---

## 5. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Computer Vision | OpenCV, Pillow |
| Application Framework | Streamlit |
| Numerical Computing | NumPy |
| Data Processing | Pandas |
| Model Evaluation | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Database | SQLite |
| Interactive Input | Streamlit Drawable Canvas |

---

## 6. Repository Structure

```text
InkVision-AI/
│
├── app.py
├── train_model.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── assets/
│   └── sample_digits.png
│
├── database/
│   └── db_manager.py
│
├── model/
│   └── digit_cnn.keras
│
├── modules/
│   └── authentication.py
│
├── reports/
│   ├── accuracy_graph.png
│   ├── loss_graph.png
│   └── confusion_matrix.png
│
└── utils/
    ├── __init__.py
    ├── multidigit.py
    ├── prediction.py
    └── preprocessing.py
```

---

## 7. Model Training and Evaluation

The model training implementation is maintained in `train_model.py`.

The training pipeline includes:

- Dataset preparation and preprocessing.
- CNN architecture definition.
- Model training.
- Performance evaluation.
- Accuracy and loss visualization.
- Confusion matrix generation.
- Saving the trained model for subsequent inference.

The trained model is stored at:

```text
model/digit_cnn.keras
```

Evaluation reports are maintained in the `reports/` directory.

Performance claims should be interpreted using the evaluation results generated by the training pipeline and the corresponding test dataset.

---

## 8. Local Development and Execution

### Prerequisites

- Python 3.10 or later.
- pip package manager.
- Git.
- Required dependencies listed in `requirements.txt`.

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/InkVision-AI.git
cd InkVision-AI
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate the Environment

**Windows:**

```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Launch the Application

```bash
streamlit run app.py
```

The application will be accessible locally at:

```text
http://localhost:8501
```

Ensure that the trained model and required report assets are present before launching the application.

---

## 9. Security and Data Management

InkVision AI incorporates basic authentication and user-specific data management.

- Passwords are stored as salted PBKDF2 hashes rather than plaintext.
- SQLite manages user accounts and prediction records.
- Prediction history is retrieved using the authenticated user's identifier.
- Environment-specific secrets and local development files should not be committed to the repository.

For production deployment, authentication, database persistence, and secure hosting configurations should be reviewed independently.

---

## 10. Scope and Technical Limitations

- The classification task is restricted to numerical digits (0–9).
- The system is not designed for unrestricted handwritten text recognition.
- Multi-digit recognition depends on successful image segmentation.
- Touching and overlapping digits may reduce segmentation reliability.
- Softmax confidence is a model output and should not be interpreted as a calibrated probability of correctness.
- Grad-CAM provides approximate visual interpretation rather than a definitive explanation of model reasoning.
- SQLite storage on temporary hosting environments may not provide permanent persistence across restarts or redeployments.

---

## 11. Future Development

Potential engineering improvements include:

- More robust segmentation of connected handwritten digits.
- Evaluation across diverse handwriting distributions.
- Model calibration and uncertainty analysis.
- Improved inference efficiency.
- Expanded model interpretability.
- Production-grade authentication.
- Persistent database infrastructure.

---

## 12. Project Information

| Attribute | Details |
|---|---|
| Project Name | InkVision AI |
| Domain | Artificial Intelligence, Deep Learning, Computer Vision |
| Primary Task | Handwritten Numerical Digit Recognition |
| Model | Convolutional Neural Network |
| Application Framework | Streamlit |
| Implementation Language | Python |
| Model Framework | TensorFlow / Keras |
| Database | SQLite |
| Developed By | Vandana H S |

---

<p align="center">
  <em>InkVision AI — Making handwritten digit recognition observable, interactive and understandable.</em>
</p>
