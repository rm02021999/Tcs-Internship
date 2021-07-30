import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json
import pandas as pd
import plotly
import plotly.express as px
from sklearn import preprocessing

app = Flask(__name__)
model = pickle.load(open('rfmodel.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(int_features)
    age = int_features[0]
    experience = int_features[1]

    prediction = model.predict(final_features)
    if prediction == 0:
        output = 'less than or equal to ₹50,000'
    if prediction == 1:
        output = 'greater than  to ₹50,000'
    return render_template('index.html', prediction_text='Employee Salary should be {}'.format(output))

@app.route('/callback', methods=['POST', 'GET'])
def callback():
    graphname = request.args.get("graphs")
    data = graphplt(graphname)

    return data

def graphplt(graph):
    pth = "salarydata.csv"
    df = pd.read_csv(pth)
    df =  df[["age","education-num","salary"]]
    label_encoder = preprocessing.LabelEncoder()
    df['salary'] = label_encoder.fit_transform(df['salary'])
    df["Experience"] = df["age"] - df['education-num'] - 8
    df = df[['age', 'Experience', 'salary']]
    grph = request.form.get('graphs')
    print("Graph Object :" ,grph)
    if grph == 'piechart':
        print(grph)
        df['bins'] = ""
        labels=["17-20","21-26","27-32","33-38","39-44","45-50","50+"]
        df['bins'] = pd.cut(df['age'],bins=[17,20,26,32,38,44,50,90], labels=["17-20","21-26","27-32","33-38","39-44","45-50","50+"])


        fig = px.pie(values=df.groupby('bins').size(), names=labels, title='Age Groups pie chart')
        
    if grph == 'sal_barchart':
        
        x = df['salary'].value_counts()
        fig = px.bar(x)
    if grph == 'accurcy_barchart':
        dictionary = {'Algorithm': ['Logistic_Reg',
                                        'Decision Tree',
                                        'Random Forest',
                                       
                                        'Guassian NB',
],
                                        'Accuracy': [0.6974110032362459,
                                        0.7349691545307443,
                                        0.7349817961165048,
                                       
                                        0.6518507281553398]}
        fig = px.bar(dictionary, x = 'Algorithm', y = 'Accuracy', title = "Classification Algorithms Accuracy")
        fig.update_layout(title_x=0.5, font_color = '#910050')

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return  graphJSON

if __name__ == "__main__":
    app.run()