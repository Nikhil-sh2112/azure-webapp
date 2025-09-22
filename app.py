from flask import Flask
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import sys

app = Flask(__name__)

def analyze_logs():

    # Create sample log data if file doesn't exist
    if not os.path.exists("system_logs.txt"):
        sample_logs = [
            "2024-09-10 10:15:23 INFO Application started successfully",
            "2024-09-10 10:16:45 INFO User login: user123",
            "2024-09-10 10:17:12 WARNING Memory usage at 85%", 
            "2024-09-10 10:18:33 INFO Database connection established",
            "2024-09-10 10:19:44 ERROR Failed to connect to external API",
            "2024-09-10 10:20:15 INFO Processing batch job 001",
            "2024-09-10 10:21:22 CRITICAL Database connection timeout",
            "2024-09-10 10:22:11 INFO Batch job 001 completed"
        ]
        with open("system_logs.txt", "w") as f:
            for log in sample_logs:
                f.write(log + "\n")

    # Read log file
    log_file_path = "system_logs.txt" 
    with open(log_file_path, "r") as file:
        logs = file.readlines()

    # Parse logs into structured DataFrame
    data = []
    for log in logs:
        parts = log.strip().split(" ", 3)
        if len(parts) < 4:
            continue
        timestamp = parts[0] + " " + parts[1]
        level = parts[2]
        message = parts[3]
        data.append([timestamp, level, message])

    df = pd.DataFrame(data, columns=["timestamp", "level", "message"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Map log levels to numeric scores
    level_mapping = {"INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
    df["level_score"] = df["level"].map(level_mapping)

    # Add message length as a feature
    df["message_length"] = df["message"].apply(len)

    # Add keyword presence features for important terms
    keywords = ['error', 'fail', 'timeout']
    for kw in keywords:
        df[f'contains_{kw}'] = df['message'].str.contains(kw, case=False).astype(int)

    # Define feature columns for modeling
    feature_cols = ['level_score', 'message_length'] + [f'contains_{kw}' for kw in keywords]

    # Scale features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Initialize and fit Isolation Forest with tuned parameters
    model = IsolationForest(contamination=0.05, n_estimators=150, random_state=42)
    df['anomaly'] = model.fit_predict(df[feature_cols])

    # Calculate anomaly scores for thresholding
    df['score'] = model.decision_function(df[feature_cols])

    # Set dynamic threshold as 5% quantile of scores and classify anomalies
    threshold = df['score'].quantile(0.05)
    df['is_anomaly'] = df['score'].apply(lambda x: "❌ Anomaly" if x < threshold else "✅ Normal")

    # Print detected anomalies
    anomalies = df[df['is_anomaly'] == "❌ Anomaly"]
    print("\n🔍 Improved Detected Anomalies:\n", anomalies, flush=True)

    return df

@app.route('/')
def home():
    # Run your analysis
    df = analyze_logs()
    anomalies = df[df['is_anomaly'] == "❌ Anomaly"]
    normals = df[df['is_anomaly'] == "✅ Normal"]

    return f"""
    <html>
    <head>
        <title>AIOps Analysis</title>
        <style>
            body {{ font-family: Arial; margin: 40px; background: lightblue; }}
            .header {{ background: blue; color: white; padding: 20px; text-align: center; }}
            .box {{ background: white; padding: 20px; margin: 15px 0; border: 2px solid blue; }}
            .problem {{ background: red; color: white; padding: 10px; margin: 5px 0; }}
            .normal {{ background: green; color: white; padding: 10px; margin: 5px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1> AIOps Log Analysis</h1>
            <p>Machine Learning on Microsoft Azure</p>
        </div>

        <div class="box">
            <h2> Results</h2>
            <p><b>Total Logs:</b> {len(df)}</p>
            <p><b>Normal:</b> {len(normals)}</p>
            <p><b>Problems:</b> {len(anomalies)}</p>
        </div>

        <div class="box">
            <h2>Problems Found</h2>
            {''.join([f'<div class="problem"><b>{row["level"]}:</b> {row["message"]}<br><small>Time:{row["timestamp"]} | Score: {row["score"]:.3f}</small></div>' for _, row in anomalies.iterrows()])}
        </div>


        <div class="box">
            <p><b>Algorithm:</b> Isolation Forest | <b>Trees:</b> 150 | <b>Contamination:</b> 5%</p>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    print(" Starting simple AIOps web app...")
    app.run(host='0.0.0.0', port=80)