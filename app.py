import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Set app title
st.title("Advanced Anomaly Detection in Financial Transactions")

# Sidebar configuration
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload creditcard.csv", type="csv")
algorithm = st.sidebar.selectbox("Select Algorithm", ["DBSCAN", "Isolation Forest"])
sample_data = st.sidebar.checkbox("Sample data (10%)", value=False)

# Algorithm-specific parameters
if algorithm == "DBSCAN":
    eps = st.sidebar.slider("DBSCAN eps", 0.5, 5.0, 3.0, 0.1)
    min_samples = st.sidebar.slider("DBSCAN min_samples", 5, 50, 10, 1)
else:
    contamination = st.sidebar.slider("Isolation Forest contamination", 0.01, 0.5, 0.1, 0.01)

# Feature selection
feature_options = st.sidebar.multiselect("Select Features", [], [])

if uploaded_file is not None:
    # Load and preprocess data
    df = pd.read_csv(uploaded_file)
    if sample_data:
        df = df.sample(frac=0.1, random_state=42)
        st.write("Using 10% sample of the dataset.")
    
    # Update feature options dynamically
    all_features = [col for col in df.columns if col != 'Class']
    if not feature_options:
        feature_options = all_features
    feature_options = st.sidebar.multiselect("Select Features", all_features, default=feature_options)
    
    # Prepare features
    X = df[feature_options]
    y_true = df['Class']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    st.write("Data preprocessed successfully.")

    # Run anomaly detection
    if st.button("Run Anomaly Detection"):
        with st.spinner(f"Running {algorithm}..."):
            if algorithm == "DBSCAN":
                model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
                y_pred = model.fit_predict(X_scaled)
                anomalies = (y_pred == -1).astype(int)
            else:
                model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
                model.fit(X_scaled)
                anomalies = model.predict(X_scaled)
                anomalies = (anomalies == -1).astype(int)
                scores = model.decision_function(X_scaled)  # Compute anomaly scores for Isolation Forest

        st.success(f"{algorithm} completed.")

        # Results
        n_anomalies = np.sum(anomalies)
        st.write(f"**Number of detected anomalies**: {n_anomalies}")

        # Clustering metrics (DBSCAN only)
        if algorithm == "DBSCAN":
            n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
            st.write(f"**Number of clusters**: {n_clusters}")
            core_mask = y_pred != -1
            if len(set(y_pred[core_mask])) > 1:
                st.write(f"**Silhouette Score**: {silhouette_score(X_scaled[core_mask], y_pred[core_mask]):.4f}")
                st.write(f"**Davies-Bouldin Index**: {davies_bouldin_score(X_scaled[core_mask], y_pred[core_mask]):.4f}")

        # Evaluation metrics
        cm = confusion_matrix(y_true, anomalies)
        detection_rate = np.sum((y_true == 1) & (anomalies == 1)) / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0
        false_alarm_rate = np.sum((y_true == 0) & (anomalies == 1)) / np.sum(y_true == 0) if np.sum(y_true == 0) > 0 else 0
        st.write(f"**Detection Rate**: {detection_rate:.4f}")
        st.write(f"**False Alarm Rate**: {false_alarm_rate:.4f}")

        # Visualizations
        st.subheader("Results Visualization")
        
        # 1. Confusion Matrix Heatmap
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, anomalies)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

        # 3. PCA Plot of Anomalies (if >=2 features selected)
        if len(feature_options) >= 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            fig, ax = plt.subplots()
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=anomalies, cmap='coolwarm', alpha=0.5)
            ax.set_title("PCA Plot of Anomalies")
            ax.set_xlabel("PCA Component 1")
            ax.set_ylabel("PCA Component 2")
            plt.colorbar(scatter, label='Anomaly (1) / Normal (0)')
            st.pyplot(fig)

        # 4. Histogram of Anomaly Scores (for Isolation Forest)
        if algorithm == "Isolation Forest":
            fig, ax = plt.subplots()
            ax.hist(scores, bins=50)
            ax.set_title("Histogram of Anomaly Scores")
            ax.set_xlabel("Anomaly Score")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        # 5. Cumulative Anomalies and Frauds Over Time (if 'Time' is present)
        if 'Time' in df.columns:
            df['Anomaly'] = anomalies
            df['True_Fraud'] = y_true
            df_sorted = df.sort_values('Time')
            cum_anomalies = df_sorted['Anomaly'].cumsum()
            cum_frauds = df_sorted['True_Fraud'].cumsum()
            
            fig, ax = plt.subplots()
            ax.plot(df_sorted['Time'], cum_anomalies, label='Detected Anomalies')
            ax.plot(df_sorted['Time'], cum_frauds, label='True Frauds')
            ax.set_title("Cumulative Anomalies and Frauds Over Time")
            ax.set_xlabel("Time")
            ax.set_ylabel("Cumulative Count")
            ax.legend()
            st.pyplot(fig)

else:
    st.write("Please upload the creditcard.csv file to proceed.")