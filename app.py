from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
linear = pickle.load(open('model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')



@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        GRE_Score = int(request.form['GRE_Score'])
        TOEFL_Score = int(request.form['TOEFL_Score'])
        SOP = float(request.form['SOP'])
        LOR  = float(request.form['LOR'])
        CGPA = float(request.form['CGPA'])
        
        data =[GRE_Score,TOEFL_Score,SOP,LOR,CGPA]
        data= np.array(data)
        data =data.reshape(1,-1)
        my_prediction = linear.predict(data)
        output=my_prediction
        if output<0.60:
         return render_template('index.html',prediction_text="Sorry,Try next time harder{}".format(output))
        else:
         return render_template('index.html',prediction_text="You are lucky that you got a chance {}".format(output))
    else:
        return render_template('index.html')
        
     
if __name__=="__main__":
    app.run(debug=True)
