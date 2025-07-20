from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('gold_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sp500 = float(request.form['SPX'])
        oil = float(request.form['USO'])
        silver = float(request.form['SLV'])
        euro = float(request.form['EUR/USD'])
        input_data = np.array([[sp500, oil, silver, euro]])
        prediction = model.predict(input_data)[0]
        return render_template('index.html', prediction_text=f'Predicted Gold Price: ${prediction:.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
