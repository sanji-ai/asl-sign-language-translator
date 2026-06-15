# Sign Language Translator 🤟

A real-time American Sign Language (ASL) alphabet recognition system built with **TensorFlow**, **OpenCV**, and **Streamlit**. The project uses **MobileNetV2 transfer learning** to recognize hand gestures from webcam input or uploaded images and convert them into text.

---

## 🚀 Project Overview

Communication barriers can make everyday interactions difficult for people who rely on sign language. This project explores how computer vision and deep learning can be used to bridge that gap by recognizing ASL alphabet gestures in real time.

Using a pretrained **MobileNetV2** model, the application can identify hand signs from live camera feeds or uploaded images and instantly translate them into text through an interactive Streamlit interface.

---

## ✨ Features

* Real-time ASL alphabet recognition using a webcam
* Image upload support for gesture prediction
* Transfer learning with MobileNetV2
* OpenCV-based image preprocessing
* Interactive Streamlit web interface
* Model export and deployment-ready workflow
* Performance evaluation with validation metrics and confusion matrix analysis

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* MobileNetV2
* OpenCV
* Streamlit
* NumPy
* Matplotlib
* Scikit-learn

---

## 📁 Project Structure

```text
asl-sign-language-translator/
│
├── notebooks/
│   └── asl_sign_language_translator.ipynb
│
├── src/
│   ├── download_dataset.py
│   └── predict.py
│
├── models/
├── data/
├── streamlit_app.py
└── README.md
```

---

## ⚙️ Setup

TensorFlow works best with Python 3.10 or 3.11.

```powershell
cd C:\Users\patid\OneDrive\Documents\Playground\asl-sign-language-translator

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

---

## 📥 Download the Dataset

1. Create a Kaggle API token from https://www.kaggle.com/settings
2. Place `kaggle.json` in:

```text
C:\Users\<your-user>\.kaggle\kaggle.json
```

3. Download and extract the dataset:

```powershell
python src/download_dataset.py --output data
```

Expected dataset location:

```text
data/asl_alphabet_train/asl_alphabet_train/
```

---

## 🧠 Model Training

Open and run the notebook:

```text
notebooks/asl_sign_language_translator.ipynb
```

After training, the notebook exports:

```text
models/asl_mobilenetv2.keras
models/labels.json
```

---

## ▶️ Run the Application

Start the Streamlit interface:

```powershell
streamlit run streamlit_app.py
```

### Webcam Demo

```powershell
python webcam_demo.py
```

Press **Q** to close the OpenCV window.

---

## 📊 Results

The final MobileNetV2 model achieved **96% validation accuracy** on the ASL Alphabet validation dataset.

Training was performed on **Google Colab using a T4 GPU**. During experimentation, I reduced the image size to **160×160** and lowered the batch size to **8** to manage memory limitations and keep training stable.

---

## 🎯 Challenges & Lessons Learned

This project taught me much more than just training a neural network.

* Learned how to apply transfer learning for image classification tasks.
* Gained hands-on experience with MobileNetV2 and TensorFlow workflows.
* Understood how image resolution and batch size directly impact RAM usage and training performance.
* Spent a lot of time debugging memory issues and training bottlenecks, which improved my problem-solving skills.
* Learned how to export trained models and integrate them into a user-facing Streamlit application.
* Connected OpenCV preprocessing pipelines with real-time model inference.

One of the biggest takeaways from this project was learning that machine learning development involves a lot of experimentation, troubleshooting, and optimization—not just building the model itself.

---
