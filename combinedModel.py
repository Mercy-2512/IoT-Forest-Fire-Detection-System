# import numpy as np
# import skfuzzy as fuzz
# from skfuzzy import control as ctrl

# def dummy_mfcc_model(audio_features):
#     return 0  # Placeholder for MFCC model, always returns 0

# # Define Fuzzy Variables
# # Inputs
# temperature = ctrl.Antecedent(np.arange(-30, 61, 1), 'temperature')
# humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
# mq2_value = ctrl.Antecedent(np.arange(0, 8001, 1), 'mq2_value')

# # Output
# fire_risk = ctrl.Consequent(np.arange(0, 101, 1), 'fire_risk')

# # Define Membership Functions for Temperature
# temperature['low'] = fuzz.trapmf(temperature.universe, [-30, -30, 10, 15])
# temperature['moderate'] = fuzz.trimf(temperature.universe, [10, 25, 40])
# temperature['high'] = fuzz.trapmf(temperature.universe, [35, 45, 60, 60])

# # Define Membership Functions for Humidity
# humidity['low'] = fuzz.trapmf(humidity.universe, [0, 0, 30, 40])
# humidity['moderate'] = fuzz.trimf(humidity.universe, [30, 50, 70])
# humidity['high'] = fuzz.trapmf(humidity.universe, [60, 75, 100, 100])

# # Define Membership Functions for MQ2 Value (Gas Levels)
# mq2_value['low'] = fuzz.trapmf(mq2_value.universe, [0, 0, 1500, 2000])
# mq2_value['moderate'] = fuzz.trimf(mq2_value.universe, [1500, 4000, 5000])
# mq2_value['high'] = fuzz.trapmf(mq2_value.universe, [4000, 6000, 8000, 8000])

# # Define Membership Functions for Fire Risk
# fire_risk['low'] = fuzz.trapmf(fire_risk.universe, [0, 0, 30, 50])
# fire_risk['moderate'] = fuzz.trimf(fire_risk.universe, [30, 50, 70])
# fire_risk['high'] = fuzz.trapmf(fire_risk.universe, [50, 70, 100, 100])

# # Define Fuzzy Rules
# rule1 = ctrl.Rule(temperature['low'] & humidity['high'] & mq2_value['low'], fire_risk['low'])
# rule2 = ctrl.Rule(temperature['moderate'] & humidity['moderate'] & mq2_value['moderate'], fire_risk['moderate'])
# rule3 = ctrl.Rule(temperature['high'] & humidity['low'] & mq2_value['high'], fire_risk['high'])
# rule4 = ctrl.Rule(temperature['high'] & mq2_value['moderate'] & humidity['low'], fire_risk['high'])
# rule5 = ctrl.Rule(temperature['moderate'] & humidity['low'] & mq2_value['high'], fire_risk['moderate'])
# rule6 = ctrl.Rule(temperature['low'] & humidity['low'] & mq2_value['high'], fire_risk['moderate'])
# rule7 = ctrl.Rule(temperature['high'] & humidity['moderate'] & mq2_value['high'], fire_risk['high'])
# rule8 = ctrl.Rule(temperature['moderate'] & humidity['high'] & mq2_value['low'], fire_risk['low'])
# rule9 = ctrl.Rule(temperature['low'] & mq2_value['high'], fire_risk['moderate'])
# rule10 = ctrl.Rule(temperature['high'] & mq2_value['high'] & humidity['high'], fire_risk['moderate'])

# # Create Control System
# fire_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10])
# fire_simulation = ctrl.ControlSystemSimulation(fire_ctrl)

# # Function to Compute Fire Risk
# def compute_fire_risk(temp, hum, mq2, audio_features=None):
#     fire_simulation.input['temperature'] = temp
#     fire_simulation.input['humidity'] = hum
#     fire_simulation.input['mq2_value'] = mq2
    
#     fire_simulation.compute()
#     fuzzy_risk = fire_simulation.output['fire_risk']
    
#     mfcc_risk = dummy_mfcc_model(audio_features) if audio_features is not None else 0
    
#     weighted_risk = (0.99 * fuzzy_risk) + (0.01 * mfcc_risk)
#     fire_status = "Fire Detected" if weighted_risk > 50 else "No Fire"
#     return round(weighted_risk), fire_status

# # Test the system with example data
# data = [
#     [28.2, 79.5, 1811],
#     [45.0, 20.0, 3500],
#     [60.0, 10.0, 4500],
#     [35.0, 50.0, 2000],
#     [25.0, 80.0, 1200]
# ]

# print("Timestamp, Temperature, Humidity, MQ2_Value, Fire_Risk, Status")
# for i, (temp, hum, mq2) in enumerate(data):
#     fire_risk_output, fire_status = compute_fire_risk(temp, hum, mq2)
#     print(f"{i}, {temp}, {hum}, {mq2}, {fire_risk_output}, {fire_status}")


import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Define Fuzzy Variables
# Inputs
temperature = ctrl.Antecedent(np.arange(-30, 61, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
mq2_value = ctrl.Antecedent(np.arange(0, 8001, 1), 'mq2_value')

# Output
fire_risk = ctrl.Consequent(np.arange(0, 101, 1), 'fire_risk')

# Define Membership Functions for Temperature
temperature['low'] = fuzz.trapmf(temperature.universe, [-30, -30, 10, 15])
temperature['moderate'] = fuzz.trimf(temperature.universe, [10, 25, 40])
temperature['high'] = fuzz.trapmf(temperature.universe, [35, 45, 60, 60])

# Define Membership Functions for Humidity
humidity['low'] = fuzz.trapmf(humidity.universe, [0, 0, 30, 40])
humidity['moderate'] = fuzz.trimf(humidity.universe, [30, 50, 70])
humidity['high'] = fuzz.trapmf(humidity.universe, [60, 75, 100, 100])

# Define Membership Functions for MQ2 Value (Gas Levels)
mq2_value['low'] = fuzz.trapmf(mq2_value.universe, [0, 0, 1500, 2000])
mq2_value['moderate'] = fuzz.trimf(mq2_value.universe, [1500, 4000, 5000])
mq2_value['high'] = fuzz.trapmf(mq2_value.universe, [4000, 6000, 8000, 8000])

# Define Membership Functions for Fire Risk
fire_risk['low'] = fuzz.trapmf(fire_risk.universe, [0, 0, 30, 50])
fire_risk['moderate'] = fuzz.trimf(fire_risk.universe, [30, 50, 70])
fire_risk['high'] = fuzz.trapmf(fire_risk.universe, [50, 70, 100, 100])

# Define Fuzzy Rules
rule1 = ctrl.Rule(temperature['low'] & humidity['high'] & mq2_value['low'], fire_risk['low'])
rule2 = ctrl.Rule(temperature['moderate'] & humidity['moderate'] & mq2_value['moderate'], fire_risk['moderate'])
rule3 = ctrl.Rule(temperature['high'] & humidity['low'] & mq2_value['high'], fire_risk['high'])
rule4 = ctrl.Rule(temperature['high'] & mq2_value['moderate'] & humidity['low'], fire_risk['high'])
rule5 = ctrl.Rule(temperature['moderate'] & humidity['low'] & mq2_value['high'], fire_risk['moderate'])
rule6 = ctrl.Rule(temperature['low'] & humidity['low'] & mq2_value['high'], fire_risk['moderate'])
rule7 = ctrl.Rule(temperature['high'] & humidity['moderate'] & mq2_value['high'], fire_risk['high'])
rule8 = ctrl.Rule(temperature['moderate'] & humidity['high'] & mq2_value['low'], fire_risk['low'])
rule9 = ctrl.Rule(temperature['low'] & mq2_value['high'], fire_risk['moderate'])
rule10 = ctrl.Rule(temperature['high'] & mq2_value['high'] & humidity['high'], fire_risk['moderate'])

# Create Control System
fire_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10])
fire_simulation = ctrl.ControlSystemSimulation(fire_ctrl)

def dummy_mfcc_model(audio_features):
    return 0  # Placeholder for MFCC model, always returns 0

# Updated compute_fire_risk with proper error handling
def compute_fire_risk(temp, hum, mq2, audio_features=None):
    try:
        fire_simulation.input['temperature'] = temp
        fire_simulation.input['humidity'] = hum
        fire_simulation.input['mq2_value'] = mq2
        
        fire_simulation.compute()
        fuzzy_risk = fire_simulation.output['fire_risk']
        
        mfcc_risk = dummy_mfcc_model(audio_features) if audio_features is not None else 0
        
        weighted_risk = (0.99 * fuzzy_risk) + (0.01 * mfcc_risk)
        fire_status = "Fire Detected" if weighted_risk > 50 else "No Fire"
        return round(weighted_risk), fire_status
    except Exception as e:
        print(f"Warning: Error in compute_fire_risk with inputs: Temp={temp}, Humidity={hum}, MQ2={mq2}")
        print(f"Error details: {e}")
        return 0, "Error in computation"

# Test data
data = [
    [28.2, 79.5, 1811],
    [45.0, 20.0, 3500],
    [60.0, 10.0, 4500],
    [35.0, 50.0, 2000],
    [25.0, 80.0, 1200]
]

# 1. Visualizing membership functions
def plot_membership_functions():
    plt.figure(figsize=(16, 12))
    
    plt.subplot(2, 2, 1)
    temperature.view()
    plt.title('Temperature Membership')
    
    plt.subplot(2, 2, 2)
    humidity.view()
    plt.title('Humidity Membership')
    
    plt.subplot(2, 2, 3)
    mq2_value.view()
    plt.title('MQ2 Gas Sensor Membership')
    
    plt.subplot(2, 2, 4)
    fire_risk.view()
    plt.title('Fire Risk Membership')
    
    plt.tight_layout()
    plt.savefig('membership_functions.png')
    plt.show()

# 2. 2D Heatmaps instead of 3D surfaces - more robust
def plot_2d_heatmaps():
    plt.figure(figsize=(16, 12))
    
    # Temperature vs Humidity (with MQ2 = 2000)
    temp_range = np.arange(-10, 61, 5)
    hum_range = np.arange(0, 101, 5)
    
    z = np.zeros((len(hum_range), len(temp_range)), dtype=float)
    
    for i in range(len(temp_range)):
        for j in range(len(hum_range)):
            try:
                risk, _ = compute_fire_risk(temp_range[i], hum_range[j], 2000)
                z[j, i] = risk
            except:
                z[j, i] = 0  # Use 0 instead of NaN for heatmap
    
    ax = plt.subplot(2, 2, 1)
    heatmap = ax.imshow(z, cmap='coolwarm', origin='lower', aspect='auto')
    plt.colorbar(heatmap, ax=ax)
    
    # Set tick labels
    plt.xticks(np.arange(0, len(temp_range), 2), temp_range[::2])
    plt.yticks(np.arange(0, len(hum_range), 2), hum_range[::2])
    
    plt.title('Fire Risk: Temperature vs Humidity (MQ2=2000)')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Humidity (%)')
    
    # Temperature vs MQ2 (with Humidity = 50)
    temp_range = np.arange(-10, 61, 5)
    mq2_range = np.arange(0, 8001, 400)
    
    z = np.zeros((len(mq2_range), len(temp_range)), dtype=float)
    
    for i in range(len(temp_range)):
        for j in range(len(mq2_range)):
            try:
                risk, _ = compute_fire_risk(temp_range[i], 50, mq2_range[j])
                z[j, i] = risk
            except:
                z[j, i] = 0
    
    ax = plt.subplot(2, 2, 2)
    heatmap = ax.imshow(z, cmap='coolwarm', origin='lower', aspect='auto')
    plt.colorbar(heatmap, ax=ax)
    
    # Set tick labels
    plt.xticks(np.arange(0, len(temp_range), 2), temp_range[::2])
    plt.yticks(np.arange(0, len(mq2_range), 2), mq2_range[::2])
    
    plt.title('Fire Risk: Temperature vs MQ2 (Humidity=50%)')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('MQ2 Value')
    
    # Humidity vs MQ2 (with Temperature = 30)
    hum_range = np.arange(0, 101, 5)
    mq2_range = np.arange(0, 8001, 400)
    
    z = np.zeros((len(mq2_range), len(hum_range)), dtype=float)
    
    for i in range(len(hum_range)):
        for j in range(len(mq2_range)):
            try:
                risk, _ = compute_fire_risk(30, hum_range[i], mq2_range[j])
                z[j, i] = risk
            except:
                z[j, i] = 0
    
    ax = plt.subplot(2, 2, 3)
    heatmap = ax.imshow(z, cmap='coolwarm', origin='lower', aspect='auto')
    plt.colorbar(heatmap, ax=ax)
    
    # Set tick labels
    plt.xticks(np.arange(0, len(hum_range), 2), hum_range[::2])
    plt.yticks(np.arange(0, len(mq2_range), 2), mq2_range[::2])
    
    plt.title('Fire Risk: Humidity vs MQ2 (Temperature=30°C)')
    plt.xlabel('Humidity (%)')
    plt.ylabel('MQ2 Value')
    
    plt.tight_layout()
    plt.savefig('fire_risk_heatmaps.png')
    plt.show()

# 3. Fixed Test Data Heatmap
def plot_test_data_heatmap():
    # Compute fire risk for all test data
    results = []
    for temp, hum, mq2 in data:
        try:
            risk, status = compute_fire_risk(temp, hum, mq2)
            results.append([temp, hum, mq2, float(risk), status])
        except Exception as e:
            print(f"Warning: Error computing fire risk for: Temp={temp}, Humidity={hum}, MQ2={mq2}")
            print(f"Error details: {e}")
            results.append([temp, hum, mq2, 0.0, "Error"])
    
    # Convert to numpy array with explicit float type
    results = np.array(results, dtype=object)
    
    # Extract risk values as float array for coloring
    risk_values = np.array([float(r[3]) for r in results])
    
    # Plot bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(results)), risk_values, color=plt.cm.RdYlGn_r(risk_values/100))
    plt.title('Fire Risk Assessment for Test Data')
    plt.ylabel('Fire Risk Score')
    plt.xlabel('Test Case')
    plt.xticks(range(len(results)), [f"Case {i+1}" for i in range(len(results))])
    
    # Add value annotations
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 f"{int(risk_values[i])}\n({results[i, 4]})", 
                 ha='center', va='bottom', fontsize=10)
    
    # Add details table
    table_data = []
    for i, (temp, hum, mq2, risk, status) in enumerate(results):
        table_data.append([f"Case {i+1}", f"{temp:.1f}°C", f"{hum:.1f}%", f"{int(mq2)}", f"{int(float(risk))}", status])
    
    plt.table(cellText=table_data,
              colLabels=["Test Case", "Temp (°C)", "Humidity (%)", "MQ2 Value", "Risk Score", "Status"],
              loc='bottom',
              bbox=[0, -0.65, 1, 0.5])
    
    plt.subplots_adjust(bottom=0.35)
    plt.savefig('test_data_assessment.png')
    plt.show()

# 4. Time-series simulation of changing conditions
def plot_time_series_simulation():
    # Simulate changes over time
    num_steps = 100
    
    # Starting conditions
    temp = 25.0
    hum = 60.0
    mq2 = 1200
    
    temp_history = [temp]
    hum_history = [hum]
    mq2_history = [mq2]
    risk_history = []
    status_history = []
    
    # Simulate a fire developing
    for i in range(num_steps):
        # Temperature rises, humidity falls, gas increases
        if i < 30:
            # Normal conditions
            temp += np.random.normal(0, 0.5)
            hum += np.random.normal(0, 1)
            mq2 += np.random.normal(0, 50)
        elif i < 60:
            # Fire starting
            temp += np.random.normal(0.8, 0.5)
            hum -= np.random.normal(0.5, 0.5)
            mq2 += np.random.normal(80, 20)
        else:
            # Fire fully developed
            temp += np.random.normal(1.2, 0.7)
            hum -= np.random.normal(0.8, 0.5)
            mq2 += np.random.normal(150, 40)
        
        # Keep values in realistic range
        temp = max(-10, min(60, temp))
        hum = max(10, min(95, hum))
        mq2 = max(500, min(7800, mq2))
        
        temp_history.append(temp)
        hum_history.append(hum)
        mq2_history.append(mq2)
        
        try:
            risk, status = compute_fire_risk(temp, hum, mq2)
            risk_history.append(float(risk))
            status_history.append(status)
        except Exception as e:
            print(f"Warning: Error in time series at step {i}")
            risk_history.append(0.0)
            status_history.append("Error")
    
    # Plot the time series
    plt.figure(figsize=(15, 10))
    
    # Temperature plot
    plt.subplot(4, 1, 1)
    plt.plot(temp_history, 'r-')
    plt.title('Temperature Over Time')
    plt.ylabel('Temperature (°C)')
    plt.grid(True)
    
    # Humidity plot
    plt.subplot(4, 1, 2)
    plt.plot(hum_history, 'b-')
    plt.title('Humidity Over Time')
    plt.ylabel('Humidity (%)')
    plt.grid(True)
    
    # MQ2 plot
    plt.subplot(4, 1, 3)
    plt.plot(mq2_history, 'g-')
    plt.title('MQ2 Gas Sensor Over Time')
    plt.ylabel('MQ2 Value')
    plt.grid(True)
    
    # Fire risk plot with proper error handling for detection points
    plt.subplot(4, 1, 4)
    plt.plot(risk_history, 'k-')
    plt.title('Fire Risk Assessment Over Time')
    plt.ylabel('Risk Score')
    plt.xlabel('Time Steps')
    plt.axhline(y=50, color='r', linestyle='--', label='Fire Detection Threshold')
    plt.grid(True)
    plt.legend()
    
    # Mark fire detection points with error handling
    fire_points = [i for i, status in enumerate(status_history) if status == "Fire Detected"]
    if fire_points:
        first_fire = fire_points[0]
        plt.axvline(x=first_fire, color='r', linestyle=':', label=f'First Fire Detection at t={first_fire}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('time_series_simulation.png')
    plt.show()

# Main function to run all visualizations
def visualize_fire_detection_system():
    print("Generating membership function plots...")
    plot_membership_functions()
    
    print("Generating 2D heatmap plots...")
    plot_2d_heatmaps()
    
    print("Generating test data assessment...")
    plot_test_data_heatmap()
    
    print("Generating time series simulation...")
    plot_time_series_simulation()
    
    print("All visualizations completed!")

# Run the visualization
if __name__ == "__main__":
    visualize_fire_detection_system()