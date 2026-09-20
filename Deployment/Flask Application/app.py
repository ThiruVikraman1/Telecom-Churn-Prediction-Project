from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import warnings

app = Flask(__name__)

# Load model
model = pickle.load(open("models/Customer_Churn_Prediction_Best_Model_7.pkl","rb"))

# Load scaler
scaler = pickle.load(open("models/scaler_7.pkl","rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    tenure = float(request.form["tenure"])
    paperless_billing = float(request.form["paperless_billing"])
    total_charges = float(request.form["total_charges"])
    fiber_optic = float(request.form["fiber_optic_internet_service"])
    no_internet_service = float(request.form["no_internet_service"])
    two_year_contract = float(request.form["two_year_contract"])
    electronic_check_payment_method = float(request.form["electronic_check_payment_method"])

    data = pd.DataFrame(
        np.array([[tenure, paperless_billing, total_charges, fiber_optic,
                     no_internet_service, two_year_contract, electronic_check_payment_method]]),
        columns=['tenure','PaperlessBilling','TotalCharges','InternetService_Fiber optic','InternetService_No','Contract_Two year','PaymentMethod_Electronic check']
    )

    num_col = ['tenure','TotalCharges']

    data[num_col] = scaler.transform(data[num_col])
    prediction = model.predict(data)[0]

    if prediction == 1:
        result = "Customer is likely to churn"
    else:
        result = "Customer is likely to retain their services"

    return render_template('result.html', prediction=result)


if __name__ == "__main__":
    app.run(debug=True)



