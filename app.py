from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('train_model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return "Hello World"


@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form.get('Tweet')

    input_query = np.array([tweet])

    result = model.predict(input_query)[0]

    return jsonify({'Disaster': str(result)})


if __name__ == '__main__':
    app.run(debug=True)
