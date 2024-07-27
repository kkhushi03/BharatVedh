import gradio as gr
import pickle
import pandas as pd

# load the data
heart=pd.read_csv('heart.dat', header=None, sep=' ', names=['age', 'sex', 'cp', 'trestbps', 'chol',
                                                            'fbs', 'restecg', 'thalach', 'exang',
                                                            'oldpeak', 'slope', 'ca', 'thal', 'heart disease'])

# load the saved models
with open('Tree.pkl', 'rb') as f:
    tree_model = pickle.load(f)

with open('svm.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open('QDA.pkl', 'rb') as f:
    qda_model = pickle.load(f)

with open('MLP.pkl', 'rb') as f:
    mlp_model = pickle.load(f)

with open('Log.pkl', 'rb') as f:
    log_model = pickle.load(f)

with open('LDA.pkl', 'rb') as f:
    lda_model = pickle.load(f)

with open('For.pkl', 'rb') as f:
    for_model = pickle.load(f)

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

    # feature scaling
    from sklearn.model_selection import train_test_split
    X = heart.drop('heart disease', axis=1)
    y = heart['heart disease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)

    # choose the model and make prediction
    model_dict = {'Decision_Tree': tree_model,
                  'QDA': qda_model,
                  'Artificial_Neural_Networks': mlp_model,
                  'Logistic_Regression': log_model,
                  'LDA': lda_model,
                  'Random_Forest': for_model,
                  'SVM': svm_model}
    model = model_dict[model_name]
    input_data_std = scaler.transform(input_data)
    probas = model.predict_proba(input_data_std)
    outtext={1:'no heart_disease', 2:'heart disease'}  
    return {f"Probability of Class {i+1}": proba for i, proba in enumerate(probas[0])}

# Create the Gradio interface
inputs = [
    gr.inputs.Number(label='age'), 
    gr.inputs.Radio(choices=[0,1], label='sex'),
    gr.inputs.Dropdown(choices=[1,2,3,4], label='chest pain type'),
    gr.inputs.Number(label='resting blood pressure'),
    gr.inputs.Number(label='serum cholestoral'),
    gr.inputs.Radio(choices=[0,1], label='fasting blood sugar'),
    gr.inputs.Radio(choices=[0,1,2], label='resting electrocardiographic'),
    gr.inputs.Number(label='maximum heart rate'),
    gr.inputs.Radio(choices=[0,1], label='exercise induced angina'),
    gr.inputs.Number(label='oldpeak'),
    gr.inputs.Dropdown(choices=[1,2,3], label='slope ST'),
    gr.inputs.Dropdown(choices=[0,1,2,3], label='major vessels'),
    gr.inputs.Dropdown(choices=[3,6,7], label='thal'),
    gr.inputs.Dropdown(choices=['Decision_Tree', 'QDA', 'Artificial_Neural_Networks', 'Logistic_Regression', 'LDA', 'Random_Forest', 'SVM'], label='Select the model')
]

outputs = gr.outputs.Label(label='Predicted class probabilities')

gr.Interface(fn=make_prediction, inputs=inputs, outputs=outputs).launch()