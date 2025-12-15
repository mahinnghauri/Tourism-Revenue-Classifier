from flask import Flask
import joblib

app = Flask(__name__)

model = joblib.load("tourism_rf_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

@app.route('/')
def home():
    return "<h1>It Works!</h1><p>Flask is running correctly</p>"

@app.route('/test')
def test():
    return f"<h1>Test Page</h1><p>Features: {feature_columns}</p>"

if __name__ == '__main__':
    print("Starting Flask on port 5000...")
    app.run(debug=True, port=5000)