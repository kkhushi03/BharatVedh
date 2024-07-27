import gradio as gr
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the data
heart = pd.read_csv('heart.dat', header=None, sep=' ', names=['age', 'sex', 'cp', 'trestbps', 'chol',
                                                            'fbs', 'restecg', 'thalach', 'exang',
                                                            'oldpeak', 'slope', 'ca', 'thal', 'heart disease'])

# Load the saved models with error handling
def load_model(filename):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

models = {
    'Tree': load_model('Tree.pkl'),
    'SVM': load_model('svm.pkl'),
    'QDA': load_model('QDA.pkl'),
    'MLP': load_model('MLP.pkl'),
    'Log': load_model('Log.pkl'),
    'LDA': load_model('LDA.pkl'),
    'For': load_model('For.pkl')
}

# Define the function to make predictions
def make_prediction(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, model_name):
    # Create a pandas DataFrame from the inputs
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })

    # Feature scaling
    X = heart.drop('heart disease', axis=1)
    y = heart['heart disease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)

    # Choose the model and make prediction
    model = models.get(model_name)
    if model is None:
        return "Model not found or failed to load."

    input_data_std = scaler.transform(input_data)
    probas = model.predict_proba(input_data_std)
    return {f"Probability of Class {i+1}": proba for i, proba in enumerate(probas[0])}

# Create the Gradio interface
inputs = [
    gr.Number(label='age'),
    gr.Radio(choices=[0,1], label='sex(m=1, f=0)'),
    gr.Dropdown(choices=[1,2,3,4], label='chest pain type'),
    gr.Number(label='resting blood pressure(NR=120/80)'),
    gr.Number(label='serum cholesterol(NR=<200mg/dl)'),
    gr.Radio(choices=[0,1], label='fasting blood sugar'),
    gr.Radio(choices=[0,1,2], label='resting electrocardiographic'),
    gr.Number(label='maximum heart rate'),
    gr.Radio(choices=[0,1], label='exercise induced angina'),
    gr.Number(label='oldpeak'),
    gr.Dropdown(choices=[1,2,3], label='slope ST'),
    gr.Dropdown(choices=[0,1,2,3], label='major vessels'),
    gr.Dropdown(choices=[3,6,7], label='thallessemia'),
    gr.Dropdown(choices=['Tree', 'QDA', 'MLP', 'Log', 'LDA', 'For', 'SVM'], label='Select the model')
]

outputs = gr.Label(label='Predicted class probabilities')

gr.Interface(fn=make_prediction, inputs=inputs, outputs=outputs).launch()
