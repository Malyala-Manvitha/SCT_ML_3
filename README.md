# 🐶🐱 Cats vs Dogs Image Classification using SVM

This project implements a machine learning model to classify images of **cats and dogs** using a **Support Vector Machine (SVM)**.  
It uses **HOG (Histogram of Oriented Gradients)** for feature extraction and is designed to run smoothly in **Visual Studio Code**.

---

## 📌 Project Overview

Image classification is an important problem in computer vision.  
In this project, a classical machine learning approach (SVM) is used instead of deep learning to classify cat and dog images.

The model is trained on labeled images and predicts whether a given image belongs to a **cat** or a **dog**.

---

## 🛠️ Technologies Used

- Python 3
- OpenCV
- Scikit-learn
- Scikit-image (HOG features)
- NumPy
- Matplotlib

---

## 📂 Project Structure

cats_dogs_svm/
│
├── dataset/
│ ├── cats/
│ │ ├── cat.0.jpg
│ │ ├── cat.1.jpg
│ │ └── ...
│ └── dogs/
│ ├── dog.0.jpg
│ ├── dog.1.jpg
│ └── ...
│
├── main.py
├── requirements.txt
└── README.md

---

## 📊 Dataset

- Dataset used: **Cats and Dogs Image Dataset**
- Source: Kaggle  
- Images are separated into two folders:
  - `cats/`
  - `dogs/`

Only a **small subset of images** is used to ensure faster training and easy upload to GitHub.

---

## ⚙️ Installation

1. Clone this repository or download the project folder.
2. Open the project in **Visual Studio Code**.
3. Install the required libraries using:

```bash
python -m pip install -r requirements.txt

▶️ How to Run the Project

1.Ensure the dataset is placed correctly inside the dataset folder.

2.Run the following command in the VS Code terminal:
python main.py

✅ Output

1.The program loads and processes images.

2.Trains an SVM model.

Displays:

a.Model accuracy

b.Classification report (precision, recall, F1-score)

Example output:
Loading and processing images...
Dataset prepared successfully!
Training SVM model...
Model training completed!

Accuracy: 80%+

🎯 Key Features

1.Uses classical machine learning (SVM)

2.Feature extraction with HOG

3.Beginner-friendly and lightweight

4.No GPU required

5.Runs without errors in Visual Studio Code

🚀 Future Improvements

1.Add real-time image prediction

2.Increase dataset size

3.Compare SVM with deep learning models

4.Add GUI or web interface

👩‍💻 Author

Malyala Manvitha
Machine Learning Intern Candidate

📌 Acknowledgement

1.Kaggle for providing the dataset

2.Scikit-learn and OpenCV communities

---

## ✅ HOW TO ADD THIS TO GITHUB

1. Open your project folder in VS Code
2. Open `README.md`
3. Paste the above content
4. Save (`Ctrl + S`)
5. Push to GitHub 🚀

---

## 🧠 SkillCraft Intern Tip (IMPORTANT)

This README shows:
✔️ You understand the project  
✔️ You followed ML workflow  
✔️ You can document your work properly  

This **matters a lot** for internships.

---

If you want, I can also:
- ✅ Shorten this README
- ✅ Make it more **technical**
- ✅ Write a **LinkedIn post** for this project
- ✅ Review your GitHub repo before submission

Just tell me 😊

