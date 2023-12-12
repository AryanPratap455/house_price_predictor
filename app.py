from flask import Flask, render_template, request
import jinja2, pickle
import pandas as pd 
import numpy as np 


app = Flask(__name__)
df=pd.read_csv('Cleaned_data.csv')
pipe=pickle.load(open('RidgeModel.pkl','rb'))


@app.route('/')
def index():

    locations=sorted(df['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.form['location']
        bhk = request.form['bhk']
        bath = request.form['bath']
        sqft = request.form['total_sqft']

        input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
        prediction = pipe.predict(input_data)[0] * 1e5

        result_message = f"Predicted Price: Rs. {np.round(prediction, 2)}"
        return render_template('index.html', result=result_message)

    except Exception as e:
        error_message = f"Error: {str(e)}"
        return render_template('index.html', result=error_message)

if __name__ == "__main__":
    app.run(debug=True)
