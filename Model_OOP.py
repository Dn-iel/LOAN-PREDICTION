import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
import joblib

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None

    def check_nulls(self):
        if self.data is not None:
            print("Jumlah nilai null per kolom:\n", self.data.isnull().sum())

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        self.check_nulls()
        
    def create_input_output(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)

class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.createModel()
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5
    
    def boxplot(self):
        numeric_columns = self.x_train.select_dtypes(include=['int64', 'float64']).columns
        self.x_train[numeric_columns].plot(kind='box', subplots=True, layout=(len(numeric_columns)//2+1, 2), figsize=(15,10), sharex=False)
        plt.tight_layout()
        plt.show()

    def dataConvertToNumeric(self, columns):
        for col in columns:
            self.x_train[col] = self.x_train[col].astype('int64')
            self.x_test[col] = self.x_test[col].astype('int64')

    def replaceGC(self, column):
        gender_replacements = {
            'fe male': 'female',
            'Male': 'male'
        }
        self.x_train[column] = self.x_train[column].replace(gender_replacements)
        self.x_test[column] = self.x_test[column].replace(gender_replacements)


    def imputeWithGroupMedian(self, group_column, target_column):
        median_impute = self.x_train.groupby(group_column)[target_column].median()

        self.x_train[target_column] = self.x_train[target_column].fillna(
            self.x_train[group_column].map(median_impute)
        )

        self.x_test[target_column] = self.x_test[target_column].fillna(
            self.x_test[group_column].map(median_impute)
        )

    def removeAge(self, column='person_age', age=70):
        to_drop = self.x_train[self.x_train[column] > age].index

        self.x_train = self.x_train.drop(to_drop).reset_index(drop=True)
        self.y_train = self.y_train.drop(to_drop).reset_index(drop=True)

    def encodeColumn(self, encode_dict):
        self.x_train = self.x_train.replace(encode_dict)
        self.x_test = self.x_test.replace(encode_dict)

    def OneHotEncoding(self, columns):
        self.encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        self.encoder.fit(self.x_train[columns])

        x_train_ohe = pd.DataFrame(
            self.encoder.transform(self.x_train[columns]),
            columns=self.encoder.get_feature_names_out(columns),
            index=self.x_train.index
        )

        x_test_ohe = pd.DataFrame(
            self.encoder.transform(self.x_test[columns]),
            columns=self.encoder.get_feature_names_out(columns),
            index=self.x_test.index
        )

        self.x_train = pd.concat([self.x_train.drop(columns=columns), x_train_ohe], axis=1)
        self.x_test = pd.concat([self.x_test.drop(columns=columns), x_test_ohe], axis=1)

    def createModel(self, random_state=42):
        self.model = XGBClassifier(random_state=random_state)

    def makePrediction(self):
        self.y_predict = self.model.predict(self.x_test) 
        
    def createReport(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test, self.y_predict, target_names=['0', '1']))
            
    def split_data(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.output_data, test_size=test_size, random_state=random_state)

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        predictions = self.model.predict(self.x_test)
        return accuracy_score(self.y_test, predictions)
    
    def tuningParameter(self):
        param_grid = {
            'n_estimators': [150, 200, 250],
            'max_depth': [2, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
        }
       
        xgb = XGBClassifier()
        grid_search= GridSearchCV(xgb ,
                            param_grid = param_grid,  
                            scoring='accuracy',        
                            cv=5, verbose=1, n_jobs=1)
        
        grid_search.fit(self.x_train,self.y_train)
       
        print("Tuned Hyperparameters :", grid_search.best_params_)
       
        self.model = grid_search.best_estimator_


    def save_model_and_encoder(self, model_filename, encoder_filename):
        joblib.dump(self.model, model_filename)
        joblib.dump(self.encoder, encoder_filename)





file_path = 'Dataset_A_loan.csv'  
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.create_input_output('loan_status')
input_df = data_handler.input_df
output_df = data_handler.output_df

model_handler = ModelHandler(input_df, output_df)
model_handler.split_data()
model_handler.dataConvertToNumeric(['person_age', 'cb_person_cred_hist_length'])
model_handler.replaceGC(column='person_gender')
model_handler.removeAge()
model_handler.imputeWithGroupMedian('person_education', 'person_income')


pg_en = {"person_gender" : {"male":0, "female":1}}
model_handler.encodeColumn(pg_en)

pl_en = {"previous_loan_defaults_on_file" : {"No":0, "Yes":1}}
model_handler.encodeColumn(pl_en)


model_handler.OneHotEncoding(['person_home_ownership', 'loan_intent', 'person_education'])

print("Before Tuning Parameter")
model_handler.train_model()
print("Model Accuracy:", model_handler.evaluate_model())
model_handler.makePrediction()
model_handler.createReport()

print("After Tuning Parameter")
model_handler.tuningParameter()
model_handler.train_model()
print("Model Accuracy :", model_handler.evaluate_model())
model_handler.makePrediction()
model_handler.createReport()




model_handler.save_model_and_encoder('xgboost_best_model.pkl', 'one_hot_encoder.pkl')