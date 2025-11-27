
# Solar Tracker Neural Network Training - Regression Model
# Run this in Google Colab

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Load your training data
# Upload your CSV file to Colab first
print("Upload your training_data.csv file...")
from google.colab import files
uploaded = files.upload()

# Load the data
data = pd.read_csv('training_data_all.csv')

print(f"Loaded {len(data)} samples")
print(data.head())
print(f"\nData ranges:")
print(f"left_ldr: {data['left_ldr'].min()} - {data['left_ldr'].max()}")
print(f"right_ldr: {data['right_ldr'].min()} - {data['right_ldr'].max()}")
print(f"servoPosition: {data['servoPosition'].min()} - {data['servoPosition'].max()}")

# Prepare the data
X = data[['left_ldr', 'right_ldr']].values
y = data['servoPosition'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize inputs (0-1023 range)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Normalize outputs (0-180 range) - helps training
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Save scaler parameters for Arduino
input_mean = scaler_X.mean_
input_std = scaler_X.scale_
output_mean = scaler_y.mean_[0]
output_std = scaler_y.scale_[0]

print(f"\nNormalization parameters:")
print(f"Input mean: {input_mean}")
print(f"Input std: {input_std}")
print(f"Output mean: {output_mean}")
print(f"Output std: {output_std}")

# Create neural network for regression
model = keras.Sequential([
    layers.Dense(8, activation='relu', input_shape=(2,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1)  # Single output for servo position
])

model.compile(
    optimizer='adam',
    loss='mse',  # Mean Squared Error for regression
    metrics=['mae']  # Mean Absolute Error
)

print("\nModel architecture:")
model.summary()

# Train the model
print("\nTraining...")
history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=200,
    batch_size=32,
    validation_data=(X_test_scaled, y_test_scaled),
    verbose=1
)

# Evaluate
test_loss, test_mae = model.evaluate(X_test_scaled, y_test_scaled)
print(f'\nTest MAE (normalized): {test_mae:.4f}')

# Test on real scale
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_real = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))

mae_degrees = np.mean(np.abs(y_pred - y_test_real))
print(f'Test MAE (degrees): {mae_degrees:.2f}°')

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='train')
plt.plot(history.history['val_mae'], label='validation')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.title('Training MAE')

plt.tight_layout()
plt.show()

# Plot predictions vs actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test_real, y_pred, alpha=0.5)
plt.plot([0, 180], [0, 180], 'r--', label='Perfect prediction')
plt.xlabel('Actual Servo Position (degrees)')
plt.ylabel('Predicted Servo Position (degrees)')
plt.title('Prediction Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Extract weights and biases
weights_layer1 = model.layers[0].get_weights()[0]  # Shape: (2, 8)
biases_layer1 = model.layers[0].get_weights()[1]   # Shape: (8,)

weights_layer2 = model.layers[1].get_weights()[0]  # Shape: (8, 8)
biases_layer2 = model.layers[1].get_weights()[1]   # Shape: (8,)

weights_layer3 = model.layers[2].get_weights()[0]  # Shape: (8, 1)
biases_layer3 = model.layers[2].get_weights()[1]   # Shape: (1,)

# Generate Arduino header file
def generate_arduino_header(weights_layer1, biases_layer1,
                           weights_layer2, biases_layer2,
                           weights_layer3, biases_layer3,
                           input_mean, input_std,
                           output_mean, output_std):
    
    header = """// Auto-generated neural network weights
// Generated from Google Colab training
// Regression model: LDR readings -> Servo position

#ifndef NN_WEIGHTS_H
#define NN_WEIGHTS_H

// Input normalization parameters (for LDR readings)
const float input_mean[2] = {"""
    
    header += f"{input_mean[0]:.6f}, {input_mean[1]:.6f}"
    header += "};\n\nconst float input_std[2] = {"
    header += f"{input_std[0]:.6f}, {input_std[1]:.6f}"
    header += "};\n\n"
    
    # Output denormalization parameters
    header += f"// Output denormalization parameters (for servo position)\n"
    header += f"const float output_mean = {output_mean:.6f};\n"
    header += f"const float output_std = {output_std:.6f};\n\n"
    
    # Layer 1 weights (2x8)
    header += "// Layer 1: 2 inputs -> 8 neurons\n"
    header += "const float weights1[2][8] = {\n"
    for i in range(2):
        header += "  {"
        header += ", ".join([f"{w:.6f}" for w in weights_layer1[i]])
        header += "}" + ("," if i < 1 else "") + "\n"
    header += "};\n\n"
    
    header += "const float biases1[8] = {"
    header += ", ".join([f"{b:.6f}" for b in biases_layer1])
    header += "};\n\n"
    
    # Layer 2 weights (8x8)
    header += "// Layer 2: 8 neurons -> 8 neurons\n"
    header += "const float weights2[8][8] = {\n"
    for i in range(8):
        header += "  {"
        header += ", ".join([f"{w:.6f}" for w in weights_layer2[i]])
        header += "}" + ("," if i < 7 else "") + "\n"
    header += "};\n\n"
    
    header += "const float biases2[8] = {"
    header += ", ".join([f"{b:.6f}" for b in biases_layer2])
    header += "};\n\n"
    
    # Layer 3 weights (8x1)
    header += "// Layer 3: 8 neurons -> 1 output\n"
    header += "const float weights3[8] = {"
    header += ", ".join([f"{w:.6f}" for w in weights_layer3.flatten()])
    header += "};\n\n"
    
    header += f"const float bias3 = {biases_layer3[0]:.6f};\n\n"
    
    header += "#endif\n"
    
    return header

# Generate the header file
header_content = generate_arduino_header(
    weights_layer1, biases_layer1,
    weights_layer2, biases_layer2,
    weights_layer3, biases_layer3,
    input_mean, input_std,
    output_mean, output_std
)

# Save to file
with open('nn_weights.h', 'w') as f:
    f.write(header_content)

print("\n✓ Arduino header file 'nn_weights.h' generated!")
print(f"✓ Model predicts servo position with ±{mae_degrees:.2f}° accuracy")
print("\nDownloading file...")

# Download the file
files.download('nn_weights.h')

print("\n" + "="*50)
print("NEXT STEPS:")
print("="*50)
print("1. Download 'nn_weights.h' file")
print("2. Place it in your Arduino sketch folder")
print("3. Upload the Arduino code with the neural network")
print("4. Your solar tracker will use AI to control the servo!")


