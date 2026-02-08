# Credit Default Prediction with Deep Learning 💳

## 📌 Project Overview
This project implements an end-to-end Deep Learning pipeline to predict the probability of credit card default. Unlike standard accuracy-focused models, this solution is engineered for **Financial Risk Management**, prioritizing the detection of high-risk defaulters.

The model navigates a significant **78:22 class imbalance** and utilizes **Keras Tuner** for automated hyperparameter optimization, ultimately achieving a **Validation AUC of 0.77+** and a **Recall of 61%**.

## 🚀 Key Business Results
| Metric | Result | Business Impact |
| :--- | :--- | :--- |
| **Recall** | **61%** | Maximizes detection of potential defaulters (High Priority). |
| **AUC Score** | **0.77+** | Indicates strong discriminative ability between classes. |
| **Threshold** | **0.60** | Shifted from 0.50 to reduce false positives while maintaining sensitivity. |

## 🛠️ Technical Approach

### 1. Data Integrity & Preprocessing
* **Forensic Data Analysis:** Identified and handled edge cases such as negative bill statements.
* **Universal Workflow:** Implemented strict **Stratified Splitting** and **Standardization** to prevent data leakage and ensure production-grade reliability.

### 2. Handling Class Imbalance (78:22)
Real-world financial data is rarely balanced. I addressed the 78:22 imbalance using **Calculated Class Weights**.
* **Strategy:** The model was penalized **3.5x harder** for False Negatives (missing a defaulter) than for False Positives.
* **Outcome:** This forced the neural network to pay attention to the minority class, significantly boosting Recall.

### 3. Automated Optimization (Keras Tuner)
Instead of manual guesswork, I integrated **Keras Tuner** to programmatically search for the optimal architecture.
* **Search Space:** Tuned number of neurons, dropout rates, and learning rates.
* **Optimal Config:** Converged on a Deep Neural Network (DNN) architecture with **112 neurons** and specific dropout layers to minimize overfitting.

### 4. Strategic Threshold Tuning
Post-training analysis involved tuning the classification decision boundary.
* **Adjustment:** Shifted threshold from default `0.50` to `0.60`.
* **Why?** To align the model's sensitivity with a specific **Financial Risk Appetite**, balancing the cost of false alarms against the risk of default.

## 💻 Tech Stack
* **Core:** Python, Pandas, NumPy
* **Deep Learning:** TensorFlow, Keras
* **Machine Learning:** Scikit-learn (Class Weights, Split)
* **Optimization:** Keras Tuner

## 📂 Project Structure
```text
├── notebooks/          # Jupyter Notebooks for EDA and Modeling
├── src/                # Source code for preprocessing pipelines
├── models/             # Saved Keras model files (.h5)
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
