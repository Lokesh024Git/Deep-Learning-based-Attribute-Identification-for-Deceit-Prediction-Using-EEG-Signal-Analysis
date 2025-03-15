import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
from sklearn.metrics import make_scorer
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
import pickle
import json

# Load dataset
df = pd.read_excel('RawEEGData.xlsx')  # Replace with your actual dataset file path

# Prepare data
X = df.drop(columns=['Subject_Number', 'Class Label'])  # Replace with actual columns to drop
y = df['Class Label'].map({1: 0, 2: 1})  # Map '1' to 'Innocent' and '2' to 'Guilty'

# Sample Entropy function
def sample_entropy(data, m=2, r=0.2):
    n = len(data)
    if n <= m:
        return 0
    def _phi(data, m, r):
        count = 0
        for i in range(n - m):
            for j in range(i + 1, n - m):
                if np.max(np.abs(data[i:i + m] - data[j:j + m])) < r:
                    count += 1
        return count / (n - m)
    try:
        phi_m = _phi(data, m, r)
        phi_m1 = _phi(data, m + 1, r)
        if phi_m == 0 or phi_m1 == 0:
            return 0
        return -np.log(phi_m1 / phi_m)
    except ZeroDivisionError:
        return 0

# RQA function
def rqa_features(data, embedding_dim=2, time_delay=1, threshold=0.2, min_diag_line=2, min_vert_line=2):
    def time_delay_embedding(signal, dim, delay):
        n = len(signal)
        embedded = np.array([signal[i:n - dim * delay + i + delay:delay] for i in range(dim)]).T
        return embedded
    def compute_recurrence_matrix(embedded_data, threshold):
        n = len(embedded_data)
        recurrence_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distance = np.linalg.norm(embedded_data[i] - embedded_data[j])
                recurrence_matrix[i, j] = 1 if distance < threshold else 0
        return recurrence_matrix
    def recurrence_rate(matrix):
        return np.sum(matrix) / matrix.size
    def determinism(matrix, min_diag_line):
        diag_counts = []
        n = matrix.shape[0]
        for k in range(1, n):
            diag = np.diag(matrix, k)
            diag_lengths = np.diff(np.where(np.diff(np.concatenate(([0], diag, [0]))) != 0)[0])
            diag_counts.extend(diag_lengths[diag_lengths >= min_diag_line])
        return np.sum(diag_counts) / np.sum(matrix)
    def laminarity(matrix, min_vert_line):
        vert_counts = []
        n = matrix.shape[0]
        for i in range(n):
            column = matrix[:, i]
            vert_lengths = np.diff(np.where(np.diff(np.concatenate(([0], column, [0]))) != 0)[0])
            vert_counts.extend(vert_lengths[vert_lengths >= min_vert_line])
        return np.sum(vert_counts) / np.sum(matrix)
    def trapping_time(matrix):
        vert_lengths = []
        n = matrix.shape[0]
        for i in range(n):
            column = matrix[:, i]
            vert_lengths.extend(np.diff(np.where(np.diff(np.concatenate(([0], column, [0]))) != 0)[0]))
        return np.mean(vert_lengths) if vert_lengths else 0
    embedded_data = time_delay_embedding(data, embedding_dim, time_delay)
    recurrence_matrix = compute_recurrence_matrix(embedded_data, threshold)
    rr = recurrence_rate(recurrence_matrix)
    det = determinism(recurrence_matrix, min_diag_line)
    lam = laminarity(recurrence_matrix, min_vert_line)
    tt = trapping_time(recurrence_matrix)
    return [rr, det, lam, tt]

# SampEn and RQA transformation function
def sampen_and_rqa_transform(data):
    features = []
    for i in range(data.shape[0]):
        channel_data = data[i, :]
        sampen_feature = sample_entropy(channel_data)
        rqa_feature = rqa_features(channel_data)
        combined_features = [sampen_feature] + rqa_feature
        features.append(combined_features)
    return np.array(features)

# Extract SampEn and RQA features
X_features = sampen_and_rqa_transform(X.values)
X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)

if X_features.shape[0] != len(y):
    raise ValueError(f"Mismatch: SampEn and RQA features {X_features.shape[0]} vs labels {len(y)}")

# Apply ICA for dimensionality reduction
ica = FastICA(n_components=12, random_state=42)  # Change the number of components if needed
X_ica = ica.fit_transform(X_features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_ica, y, test_size=0.1, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the data to be 3D for LSTM (batch_size, timesteps, features)
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))  # 1 timestep per sample
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))  # 1 timestep per sample

# Define and train the LSTM model
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=0.001)  # Adam optimizer for better performance
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

lstm_model = create_lstm_model()
lstm_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=1)

# Evaluate the model on the test data
test_loss, test_accuracy = lstm_model.evaluate(X_test_scaled, y_test, verbose=0)

# Print the accuracy
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the trained model and scaler
with open("lstm_eeg_model.pkl", "wb") as f:
    pickle.dump(lstm_model, f)
    
columns = {
    'data_columns': list(X.columns)  # Extract dataset column names dynamically
}

with open("columns.json", "w") as f:
    f.write(json.dumps(columns))

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Prediction function for user input
def predict_eeg_input(user_input_data):
    # Ensure the input data is in the correct format
    user_input_data = np.array(user_input_data)
    user_input_data_features = sampen_and_rqa_transform(user_input_data)
    user_input_data_ica = ica.transform(user_input_data_features)
    user_input_data_scaled = scaler.transform(user_input_data_ica)
    user_input_data_scaled = user_input_data_scaled.reshape((user_input_data_scaled.shape[0], 1, user_input_data_scaled.shape[1]))

    # Load the model (if not already loaded)
    with open("lstm_eeg_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Make prediction
    prediction = model.predict(user_input_data_scaled)
    return "Guilty" if prediction >= 0.5 else "Innocent"

# Example usage: Input data for a single EEG sample
user_input = [
    [-0.136637929,-0.120336899,-0.238101492,-0.299465812,-0.171832929,0.016119128,-0.228545393,-0.21518379,-0.095156435,-0.055951957,-0.262127049,-0.259148102,-0.260050293,-0.202792884,-0.077706817,-0.182294312]
    # This input should have the same number of features as the model was trained on
]

prediction = predict_eeg_input(user_input)
print(f"The predicted class for the given EEG input is: {prediction}")
