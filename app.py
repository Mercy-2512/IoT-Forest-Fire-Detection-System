from flask import Flask, request, jsonify
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from flask_cors import CORS


app = Flask(__name__)
CORS(app, supports_credentials=True)

@app.after_request
def after_request(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response
# Define Fuzzy Variables
temperature = ctrl.Antecedent(np.arange(-30, 61, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
mq2_value = ctrl.Antecedent(np.arange(0, 8001, 1), 'mq2_value')
fire_risk = ctrl.Consequent(np.arange(0, 101, 1), 'fire_risk')

# Define Membership Functions
temperature['low'] = fuzz.trapmf(temperature.universe, [-30, -30, 10, 15])
temperature['moderate'] = fuzz.trimf(temperature.universe, [10, 25, 40])
temperature['high'] = fuzz.trapmf(temperature.universe, [35, 45, 60, 60])

humidity['low'] = fuzz.trapmf(humidity.universe, [0, 0, 30, 40])
humidity['moderate'] = fuzz.trimf(humidity.universe, [30, 50, 70])
humidity['high'] = fuzz.trapmf(humidity.universe, [60, 75, 100, 100])

mq2_value['low'] = fuzz.trapmf(mq2_value.universe, [0, 0, 1500, 2000])
mq2_value['moderate'] = fuzz.trimf(mq2_value.universe, [1500, 4000, 5000])
mq2_value['high'] = fuzz.trapmf(mq2_value.universe, [4000, 6000, 8000, 8000])

fire_risk['low'] = fuzz.trapmf(fire_risk.universe, [0, 0, 30, 50])
fire_risk['moderate'] = fuzz.trimf(fire_risk.universe, [30, 50, 70])
fire_risk['high'] = fuzz.trapmf(fire_risk.universe, [50, 70, 100, 100])

# Define Rules safely
rule1 = ctrl.Rule(antecedent=(temperature['low'] & humidity['high'] & mq2_value['low']), consequent=fire_risk['low'])
rule2 = ctrl.Rule(antecedent=(temperature['moderate'] & humidity['moderate'] & mq2_value['moderate']), consequent=fire_risk['moderate'])
rule3 = ctrl.Rule(antecedent=(temperature['high'] & humidity['low'] & mq2_value['high']), consequent=fire_risk['high'])
rule4 = ctrl.Rule(antecedent=(temperature['high'] & mq2_value['moderate'] & humidity['low']), consequent=fire_risk['high'])
rule5 = ctrl.Rule(antecedent=(temperature['moderate'] & humidity['low'] & mq2_value['high']), consequent=fire_risk['moderate'])

fire_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
fire_simulation = ctrl.ControlSystemSimulation(fire_ctrl)

# Fire Risk Computation Function
def compute_fire_risk(temp, hum, mq2):
    fire_simulation.input['temperature'] = temp
    fire_simulation.input['humidity'] = hum
    fire_simulation.input['mq2_value'] = mq2
    fire_simulation.compute()
    fire_risk_value = round(fire_simulation.output['fire_risk'])
    return {"fire_risk": fire_risk_value, "status": "Fire Detected" if fire_risk_value > 50 else "No Fire"}

# Flask API Endpoint
@app.route('/fire', methods=['POST'])
def fire_risk_api():
    data = request.json
    temp = data.get("temperature")
    hum = data.get("humidity")
    mq2 = data.get("mq2_value")
    print(temp, hum, mq2)
    result = compute_fire_risk(temp, hum, mq2)
    print(result)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)