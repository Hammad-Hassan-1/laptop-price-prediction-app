import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

# Load model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('columns.pkl', 'rb') as f:
    columns = pickle.load(f)

# Feature engineering functions
def extract_resolution(res):
    if isinstance(res, str):
        match = re.search(r'(\d+)x(\d+)', res)
        if match:
            return int(match.group(1)) * int(match.group(2))
    return 0

def extract_cpu_speed(cpu):
    if isinstance(cpu, str):
        match = re.search(r'(\d+\.\d+)(?:GHz)', cpu)
        if match:
            return float(match.group(1))
    return 2.0

def extract_storage(memory):
    if isinstance(memory, str) and 'SSD' in memory:
        match = re.search(r'(\d+)(GB|TB)', memory)
        if match:
            size = int(match.group(1))
            unit = match.group(2)
            return size * 1000 if unit == 'TB' else size
    return 0

def extract_gpu_type(gpu):
    if isinstance(gpu, str):
        if 'NVIDIA' in gpu:
            return 'NVIDIA'
        elif 'AMD' in gpu:
            return 'AMD'
        elif 'Intel' in gpu:
            return 'Intel'
    return 'Other'

# Streamlit app
st.title("Laptop Price Predictor")

# Input fields
company = st.selectbox("Company", ['Dell', 'HP', 'Lenovo', 'Asus', 'Acer', 'Apple'])
type_name = st.selectbox("Type", ['Ultrabook', 'Notebook', 'Gaming', '2 in 1 Convertible'])
inches = st.slider("Screen Size (inches)", 10.0, 18.0, 15.6)
resolution = st.selectbox("Screen Resolution", ['1920x1080', '1366x768', '2560x1600'])
cpu = st.selectbox("CPU", ['Intel Core i5 2.3GHz', 'Intel Core i7 2.7GHz', 'AMD Ryzen 5 2.1GHz'])
ram = st.selectbox("RAM (GB)", [4, 8, 16, 32])
memory = st.selectbox("Storage", ['128GB SSD', '256GB SSD', '512GB SSD', '1TB SSD', '1TB HDD'])
gpu = st.selectbox("GPU", ['NVIDIA GeForce RTX 3060', 'Intel UHD Graphics', 'AMD Radeon RX 6600M', 'NVIDIA GeForce GTX 1650'])
op_sys = st.selectbox("Operating System", ['Windows', 'macOS', 'Linux'])

# Create input DataFrame
input_data = pd.DataFrame({
    'Inches': [inches],
    'Ram': [int(ram)],
    'Resolution': [extract_resolution(resolution)],
    'CpuSpeed': [extract_cpu_speed(cpu)],
    'Storage': [extract_storage(memory)]
})

# Add dummy variables for categorical features
for col in columns:
    if col.startswith('Company_') or col.startswith('TypeName_') or col.startswith('OpSys_') or col.startswith('Gpu_'):
        input_data[col] = 0
input_data[f'Company_{company}'] = 1
input_data[f'TypeName_{type_name}'] = 1
input_data[f'OpSys_{op_sys}'] = 1
input_data[f'Gpu_{extract_gpu_type(gpu)}'] = 1

# Ensure all columns match training data
input_data = input_data.reindex(columns=columns, fill_value=0)

# Scale and predict
input_scaled = scaler.transform(input_data)
pred_log = model.predict(input_scaled)
pred_price = np.exp(pred_log)[0]

# Display prediction
st.write(f"Predicted Laptop Price: INR: {pred_price:.2f}")