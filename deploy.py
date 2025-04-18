import joblib
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

def load_joblib(filename):
    return joblib.load(filename)

def predict_with_model(model, user_input):
    prediction = model.predict(user_input)
    return prediction[0]

def main():
    # Load model dan encoder
    model_filename = 'xgb_model.pkl'           # Ganti jika file joblib = 'xgb_model.joblib'
    encoder_filename = 'onehot_encoder.pkl'    # Ganti jika file joblib = 'onehot_encoder.joblib'

    model = load_joblib(model_filename)
    encoder = load_joblib(encoder_filename)

    all_columns = [
        'person_age', 'person_gender', 'person_education', 'person_income',
        'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 
        'credit_score', 'previous_loan_defaults_on_file'
    ]

    def encode(gender, previous_loan): 
        gender_enc = 1 if gender.lower() == "male" else 0
        previous_loan_enc = 1 if previous_loan.lower() == "yes" else 0
        return gender_enc, previous_loan_enc

    # Simulasi input user
    user_input = [34, 'female', 'Associate', 6457.0, 3, 'OWN', 
                  67000.0, 'PERSONAL', 20.2, 0.32, 2, 589, 'Yes']
    
    gender, previous = encode(user_input[1], user_input[12])
    user_input[1] = gender
    user_input[12] = previous

    df_input = pd.DataFrame([user_input], columns=all_columns)

    ohe_cols = ['person_home_ownership', 'loan_intent', 'person_education']
    df_ohe = pd.DataFrame(
        encoder.transform(df_input[ohe_cols]),
        columns=encoder.get_feature_names_out(ohe_cols),
        index=df_input.index
    )

    df_encoded = pd.concat([df_input.drop(columns=ohe_cols), df_ohe], axis=1)

    prediction = predict_with_model(model, df_encoded)
    print(f"The predicted output is: {prediction}")

if __name__ == "__main__":
    main()
