import joblib
import warnings
import pandas as pd
import streamlit as st

warnings.filterwarnings('ignore')

def load_joblib(filename):
    return joblib.load(filename)

def predict_with_model(model, user_input):
    prediction = model.predict(user_input)
    return prediction[0]

def main():
    st.title("Loan Prediction App")

    model_filename = 'xgboost_best_model.pkl' 
    encoder_filename = 'one_hot_encoder.pkl'

    model = load_joblib(model_filename)
    encoder = load_joblib(encoder_filename)

    st.subheader("Masukkan informasi peminjam:")

    person_age = st.number_input("Usia", min_value=20, max_value=90, value=30)
    person_gender = st.selectbox("Jenis Kelamin", ["male", "female"])
    person_education = st.selectbox("Pendidikan", ["Associate", "Bachelor", "Doctorate", "High School", "Master"])
    person_income = st.number_input("Pendapatan", min_value=0.0, value=5000.0)
    person_emp_exp = st.number_input("Pengalaman Kerja (tahun)", min_value=0, value=3)
    person_home_ownership = st.selectbox("Status Kepemilikan Rumah", ["MORTGAGE", "OWN", "RENT", "OTHER"])
    loan_amnt = st.number_input("Jumlah Pinjaman", min_value=1000.0, value=10000.0)
    loan_intent = st.selectbox("Tujuan Pinjaman", ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE" ])
    loan_int_rate = st.number_input("Bunga Pinjaman (%)", min_value=0.0, value=10.0)
    loan_percent_income = st.number_input("Persentase Pinjaman dari Pendapatan", min_value=0.0, value=0.25)
    cb_hist_len = st.number_input("Lama Riwayat Kredit", min_value=0, value=2)
    credit_score = st.number_input("Skor Kredit", min_value=0, value=600)
    previous_loan = st.selectbox("Ada Pinjaman Sebelumnya?", ["Yes", "No"])

    if st.button("Prediksi"):
        gender_enc = 1 if person_gender == "female" else 0
        prev_loan_enc = 1 if previous_loan == "Yes" else 0

        all_columns = [
            'person_age', 'person_gender', 'person_education', 'person_income',
            'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
            'credit_score', 'previous_loan_defaults_on_file'
        ]

        user_input = [
            person_age, gender_enc, person_education, person_income,
            person_emp_exp, person_home_ownership, loan_amnt, loan_intent,
            loan_int_rate, loan_percent_income, cb_hist_len,
            credit_score, prev_loan_enc
        ]

        df_input = pd.DataFrame([user_input], columns=all_columns)

        ohe_cols = ['person_home_ownership', 'loan_intent', 'person_education']
        df_ohe = pd.DataFrame(
            encoder.transform(df_input[ohe_cols]),
            columns=encoder.get_feature_names_out(ohe_cols),
            index=df_input.index
        )

        df_encoded = pd.concat([df_input.drop(columns=ohe_cols), df_ohe], axis=1)

        prediction = predict_with_model(model, df_encoded)

        st.success(f"Hasil prediksi: {'Lolos Pinjaman' if prediction == 1 else 'Tidak Lolos Pinjaman'}")

if __name__ == '__main__':
    main()
