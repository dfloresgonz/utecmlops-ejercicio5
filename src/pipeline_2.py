# filepath: /ml-pipeline-project/ml-pipeline-project/src/pipelineml.py
import pandas as pd
from load_data_2 import load_data
from data_preparation_2 import prepare_data
from model_trainer_2 import train_model
from model_regestry_2 import register_model

def main():
    # Load data
    data = load_data("/Users/diego/Documents/projects/utec-mlops/ejercicio5/data/in/application_data.csv")
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    # Train model
    model, accuracy = train_model(X_train, y_train, X_test, y_test)
    n_estimators = model.n_estimators
    model_name = "RandomForestClassifier"

    # Register model with MLflow
    register_model(model, model_name, n_estimators, accuracy)

if __name__ == "__main__":
    main()