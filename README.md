
**FinGuard** is a powerful, interactive Streamlit-based web app for detecting anomalies in financial transactions using unsupervised machine learning algorithms — **DBSCAN** and **Isolation Forest**.

It’s designed for detecting potential frauds in datasets like `creditcard.csv`, visualizing anomalies, evaluating performance, and helping analysts explore patterns with ease.

---

## 🧠 Features

- 📦 Upload and process your own `creditcard.csv`-style datasets
- 🧮 Select between **DBSCAN** or **Isolation Forest** for anomaly detection
- 🎚️ Tune model parameters interactively
- 📊 Evaluate results with:
  - Confusion Matrix
  - ROC Curve + AUC
  - Detection Rate & False Alarm Rate
  - Clustering metrics (Silhouette Score, Davies-Bouldin Index)
- 📈 Visualizations:
  - PCA projection with anomaly highlights
  - Anomaly score distribution (for Isolation Forest)
  - Cumulative anomalies vs frauds over time
- ✅ Optional 10% sampling for faster testing on large datasets

---

## 🧪 Algorithms Used

### 🔹 DBSCAN
Density-Based Spatial Clustering of Applications with Noise  
- Groups dense regions into clusters
- Points not in any cluster are considered anomalies (`label = -1`)

### 🔹 Isolation Forest
- Tree-based anomaly detection
- Anomalies are easier to isolate
- Fast and effective for high-dimensional data

---

## 🧮 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Confusion Matrix** | Compares true frauds vs detected anomalies |
| **Detection Rate (TPR)** | % of actual frauds correctly flagged |
| **False Alarm Rate (FPR)** | % of normal transactions incorrectly flagged |
| **Silhouette Score** | Cluster cohesion (for DBSCAN) |
| **Davies-Bouldin Index** | Cluster separation (lower = better) |
| **ROC Curve + AUC** | Overall model discriminative performance |

---

## 📁 File Structure

```bash
project-root/
├── app.py               # Main Streamlit app file
├── requirements.txt     # All required Python packages
├── README.md            # You're reading it!
