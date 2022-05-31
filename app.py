import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
file = open('crop.pkl', 'rb')
model = pickle.load(file)
file.close()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    output = model.predict(final_features)

    return render_template('index.html', prediction_text='Crop Yeild should be {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
