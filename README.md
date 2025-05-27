Intrusion Detection System (IDS) with Deep Learning
This project implements an Intrusion Detection System (IDS) using deep learning techniques to detect malicious network traffic. It leverages the UNSW-NB15 dataset for training and testing a neural network model and uses Scapy for real-time packet analysis. The system is designed to classify network activities as normal or malicious, with logging capabilities for monitoring network activity.
Table of Contents

Project Overview
Features
Dataset
Installation
Usage
Model Architecture
Real-Time Packet Analysis
Logging
Results
Contributing
License
References

Project Overview
The IDS project aims to enhance network security by detecting intrusions using a deep learning model trained on the UNSW-NB15 dataset. The system preprocesses network traffic data, trains a neural network to classify activities as normal or attack, and performs real-time packet analysis using Scapy. The implementation is executed in a Google Colab environment, with logging to track predictions and system activities.
Features

Data Preprocessing: Encodes categorical features, balances classes, and scales numerical data.
Deep Learning Model: A TensorFlow/Keras-based neural network for binary classification (normal vs. attack).
Real-Time Analysis: Uses Scapy to sniff and analyze network packets in real-time.
Logging: Records packet predictions and system activities to network_activity.txt.
High Accuracy: Achieves ~91.42% test accuracy on the UNSW-NB15 dataset.

Dataset
The UNSW-NB15 dataset is a comprehensive cybersecurity dataset created by the Australian Centre for Cyber Security (ACCS). It contains:

Records: 2,540,044 across four CSV files (UNSW-NB15_1.csv to UNSW-NB15_4.csv).
Features: 49 features, including protocol type, service, state, and packet sizes.
Attacks: Nine types (Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms).
Training/Testing Split: 175,341 training records and 82,332 testing records.
Ground Truth: Provided in UNSW-NB15_GT.csv, with event details in UNSW-NB15_LIST_EVENTS.csv.

The dataset is preprocessed to encode categorical columns (proto, service, state), balance classes (186,000 records, 93,000 per class), and drop non-numeric columns for model compatibility.
Installation
To set up the project, follow these steps:

Clone the Repository:
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


Install Dependencies:Ensure Python 3.8+ is installed. Install required libraries using pip:
pip install pandas numpy scikit-learn tensorflow scapy pyarrow fastparquet adversarial-robustness-toolbox


Install Npcap (Windows):For packet sniffing with Scapy on Windows:

Download and install Npcap.
Add Npcap to your system PATH (e.g., C:\Program Files\Npcap).


Google Colab Setup:If running in Google Colab:

Upload the Jupyter Notebook (IDS.ipynb) to Colab.
Install dependencies in a Colab cell:!pip install pandas numpy scikit-learn tensorflow scapy pyarrow fastparquet adversarial-robustness-toolbox


Note: Real-time packet sniffing may be limited in Colab due to network restrictions.


Download the Dataset:

Download the UNSW-NB15 dataset from UNSW.
Place the CSV files (UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv) in the project directory or upload to Colab.



Usage

Run the Jupyter Notebook:

Open IDS.ipynb in Jupyter Notebook or Google Colab.
Execute the cells sequentially to:
Preprocess the UNSW-NB15 dataset.
Train the neural network model.
Perform real-time packet analysis (if running locally with Npcap).




Local Execution:

Ensure Npcap is installed and configured for Scapy.
Run the notebook locally with administrative privileges to enable packet sniffing:jupyter notebook IDS.ipynb




View Logs:

Check network_activity.txt in the project directory for packet predictions and system logs.
In Colab, download the log file using:from google.colab import files
files.download('network_activity.txt')





Model Architecture
The deep learning model is a Sequential neural network built with TensorFlow/Keras:

Input Layer: Matches the number of preprocessed features.
Hidden Layers: Two dense layers (128 and 64 units, ReLU activation) with 30% dropout.
Output Layer: Single unit with sigmoid activation for binary classification.
Training: 10 epochs, batch size 64, Adam optimizer, binary cross-entropy loss.
Output: Model saved as ids_model.h5.

Real-Time Packet Analysis

Tool: Scapy captures live network packets.
Features Extracted: Duration, packet counts, byte counts, TTL, protocol, flags.
Classification: Packets are scaled and fed into the model. A threshold of 0.6 classifies packets as normal (≤0.6) or attack (>0.6).
Limitations: Feature mismatches between dataset and live packets may affect accuracy. Default values (e.g., 0) are used for missing features.

Logging

File: network_activity.txt
Content: Logs packet details, predictions (normal/attack), and errors.
Troubleshooting:
Ensure write permissions for the log file.
Check the file path in the script.
In Colab, verify the file is generated before downloading.



Results

Test Accuracy: 91.42%
Test Loss: 0.1665
Validation Accuracy: ~91.28% (after 10 epochs).
Observations: High confidence in real-time predictions suggests potential overfitting or feature mismatches, requiring further tuning.

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Make changes and commit (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.

Please ensure code follows PEP 8 standards and includes comments for clarity.
License
This project is licensed under the MIT License. See the LICENSE file for details.
References

Moustafa, Nour, and Jill Slay. "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)." Military Communications and Information Systems Conference (MilCIS), IEEE, 2015.
Moustafa, Nour, and Jill Slay. "The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW-NB15 dataset and the comparison with the KDD99 dataset." Information Security Journal: A Global Perspective, 2016.

