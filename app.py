from flask import Flask, request, render_template
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)
model = joblib.load("model.joblib")


@app.route("/")
def home():
    return render_template("index.html")


@app.route('/predict',methods=['POST'])
def predict():
    data = [x for x in request.form.values()]
    df = pd.DataFrame([data])
    X = df.iloc[:,:].values
    le = LabelEncoder()
    X[:,0] = le.fit_transform(X[:,0])
    X[:,1] = le.fit_transform(X[:,1])
    X[:,5] = le.fit_transform(X[:,5])
    sc = StandardScaler()
    X_test = sc.fit_transform(X)
    predict =model.predict(X_test)
    return render_template('index.html', prediction_text='The price range would be {}'.format(predict))


if __name__ == "__main__":
    app.run(debug=True)