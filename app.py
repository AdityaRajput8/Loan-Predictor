from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)
# Loading the "Brain" (Model & Scaler)
# We load these globally so we don't have to reload them for every single user request.
# This makes the app faster.
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

#-->Routes
#Home page
@app.route('/',methods=['GET','POST'])
def home():
    prediction_text=""
    # If the user hit the "Predict" button (sent a POST request)
    if request.method=='POST':
        try:
            # 1. Capture Inputs from the HTML Form
            # We use request.form.get('name_attribute_in_html')
            age = float(request.form['age'])
            gender=int(request.form['gender'])
            marital = int(request.form['marital'])
            education = int(request.form['education'])
            monthly_income = float(request.form['monthly_income'])
            employment = int(request.form['employment'])
            dti = float(request.form['dti'])
            credit_score = float(request.form['credit_score'])
            loan_amount = float(request.form['loan_amount'])
            loan_purpose = int(request.form['loan_purpose'])
            interest_rate = float(request.form['interest_rate'])
            loan_term = float(request.form['loan_term'])
            installment = float(request.form['installment'])
            grade = int(request.form['grade'])
            open_acc = float(request.form['open_acc'])
            total_limit = float(request.form['total_limit'])
            current_balance = float(request.form['current_balance'])
            delinq_hist = int(request.form['delinq_hist'])
            public_records = float(request.form['public_records'])
            num_delinq = float(request.form['num_delinq'])
            annual_income = float(request.form['annual_income'])
            # 2. Preprocessing (Feature Engineering)
            # Log transform the annual income just like we did in the notebook in analysis.ipynb
            income_log=np.log(annual_income) if annual_income>0 else 0
            # 3. Create the Feature Array
            # The order MUST match your X_train columns exactly!
            features = np.array([[
                age, gender, marital, education, monthly_income, employment, dti, credit_score,
                loan_amount, loan_purpose, interest_rate, loan_term, installment, grade,
                open_acc, total_limit, current_balance, delinq_hist, public_records, num_delinq, income_log
            ]])
            # 4. Scale the Features
            features_scaled=scaler.transform(features)
            # 5. Make Prediction
            prediction = model.predict(features_scaled)
            probability = model.predict_proba(features_scaled)[0][1] # Probability of CLASS 1 (Approved)

            # 6. Format the Result
            # We will pass structured data to the template for better UI handling
            if prediction[0] == 1:
                prediction_label = "Approved"
                prediction_class = "safe"
                # If approved, score is high (probability of 1)
                prediction_score = round(probability * 100) 
            else:
                prediction_label = "High Risk"
                prediction_class = "risk"
                # If risk, score is low (probability of 1 is low)
                # Or we can show "Risk Score" where high is bad. 
                # Let's stick to "Credit Score / Safety Score" where 100 is good.
                prediction_score = round(probability * 100)

        except Exception as e:
            prediction_label = "Error"
            prediction_class = "error"
            prediction_score = 0
            prediction_text = str(e) # Fallback for error message

    # Render the HTML page and pass structured data
    return render_template('index.html', 
                           prediction_text=prediction_text if 'prediction_text' in locals() else None,
                           result_label=prediction_label if 'prediction_label' in locals() else None,
                           result_class=prediction_class if 'prediction_class' in locals() else None,
                           result_score=prediction_score if 'prediction_score' in locals() else None)
if __name__=='__main__':
    app.run(debug=True)

