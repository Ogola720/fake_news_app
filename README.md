<div align="center">

  <img src="https://github.com/Ogola720/fake_news_app/blob/2f5130a76116798560391b391d7ea32ed84ae009/demo.mp4" width="100" />

  # 🕵️ Veritas AI | Fake News Detector
  
  **Unmasking Misinformation with Natural Language Processing & Explainable AI**

  [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
  [![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
  [![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](./LICENSE)

  <p align="center">
    <a href="#-key-features">Key Features</a> •
    <a href="#-tech-stack">Tech Stack</a> •
    <a href="#-installation">Installation</a> •
    <a href="#-model-performance">Performance</a> •
    <a href="#-live-demo">Live Demo</a>
  </p>
</div>

---

## 🚀 Overview

**Veritas AI** is a cutting-edge machine learning application designed to combat the spread of misinformation. By leveraging **TF-IDF Vectorization** and a **PassiveAggressive Classifier**, this system analyzes news articles in real-time to determine their credibility.

Unlike standard "black box" AI, Veritas includes an **Explainability Engine** that highlights exactly *which words* triggered the decision, offering transparency alongside accuracy.

---

## 🎥 Live Demo

<div align="center">
  <img src="https://github.com/Ogola720/fake_news_app/blob/2f5130a76116798560391b391d7ea32ed84ae009/demo.mp4" alt="Demo Animation" width="700" style="border-radius: 10px; box-shadow: 0px 4px 12px rgba(0,0,0,0.5);">
  <br>
  <i>(Real-time credibility analysis with confidence scoring and keyword highlighting)</i>
</div>

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **🧠 Advanced NLP** | Uses `NLTK` and `TF-IDF` to process complex linguistic patterns. |
| **⚡ Real-Time Analysis** | Instant prediction with a lightweight **PassiveAggressive Classifier**. |
| **📊 Confidence Gauge** | Interactive speedometer showing the model's certainty (0-100%). |
| **🔍 Explainable AI (XAI)** | Breaks down the "Why" by visualizing keyword influence using `Plotly`. |
| **🎨 Modern UI** | Built with `Streamlit` featuring Lottie animations for a seamless experience. |

---

## 🛠 Tech Stack

<div align="center">

| Category | Technologies |
| :---: | :--- |
| **Core** | ![Python](https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=white) |
| **ML & NLP** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat-square&logo=scikit-learn&logoColor=white) ![NLTK](https://img.shields.io/badge/NLTK-306998?style=flat-square&logo=python&logoColor=white) |
| **Data Viz** | ![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=flat-square&logo=plotly&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=flat-square&logo=Matplotlib&logoColor=black) |
| **Frontend** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=Streamlit&logoColor=white) |

</div>

---

## 🏗️ Project Structure

```bash
Veritas-AI/
├── 📂 .venv/                   # Virtual environment
├── 📄 app.py                   # Main Streamlit Dashboard application
├── 📄 fake_news_classifier.pkl # Pre-trained Machine Learning Model
├── 📄 tfidf_vectorizer.pkl     # Pre-fitted TF-IDF Vectorizer
├── 📓 Fake_News_Detection.ipynb # Jupyter Notebook for Model Training
├── 📄 sample_news         # Sample news for test
├── 📄 demo.gif         # Short demonstration video
├── 📄 requirements.txt         # Project dependencies
└── 📄 README.md                # Project documentation

```
---
⚡ Installation & SetupFollow these steps to run Veritas AI locally:<details><summary><b>1. Clone the Repository</b> (Click to expand)</summary>Bashgit clone [https://github.com/your-username/veritas-ai.git](https://github.com/your-username/veritas-ai.git)
cd veritas-ai
</details><details><summary><b>2. Create a Virtual Environment</b></summary>Bash# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
</details><details><summary><b>3. Install Dependencies</b></summary>Bashpip install -r requirements.txt
</details><details><summary><b>4. Run the App</b></summary>Bashstreamlit run app.py
</details>📈 Model PerformanceThe PassiveAggressive Classifier was chosen for its superiority in large-scale text data.<div align="center">MetricScoreAccuracy93.4%Precision92.8%Recall93.1%</div>
Note: The model was trained on the ISOT Fake News Dataset, achieving >90% accuracy on the test set.
🔮 Future Improvements
[ ] LSTM Integration: Incorporating Deep Learning (Long Short-Term Memory) for better context awareness.
[ ] URL Analysis: Adding a feature to scrape and analyze news directly from a URL.
[ ] Multi-language Support: Expanding the dataset to include non-English articles.

🤝 ContributingContributions are what make the open-source community such an amazing place to learn, inspire, and create.
Any contributions you make are greatly appreciated.
Fork the ProjectCreate your Feature Branch (git checkout -b feature/AmazingFeature)Commit your Changes (git commit -m 'Add some AmazingFeature')Push to the Branch (git push origin feature/AmazingFeature)Open a Pull Request
<div align="center">Built with ❤️ by Ogola Peter </div>
