import numpy as np
from numpy.matlib import repmat
import scipy
from scipy.stats import mode
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, log_loss
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder

from matplotlib.colors import LinearSegmentedColormap
from mne import create_info
from mne.viz import plot_topomap


def create_all_intensity_observator(df, clusters):
    # Create an empty list to hold the concatenated data for each ID
    classes = []
    concatenated_data = []

    # Loop through each unique ID
    for ID in df['ID'].unique():
        # Filter the dataframe for the current ID
        df_temp = df[df['ID'] == ID]
        
        # List to hold the data arrays for all classes
        class_arrays = []
        
        # Loop through each unique class for the current ID
        max_c = df_temp['Class'].value_counts().max()
        min_c = df_temp['Class'].value_counts().min()
        rep_factor = max_c//min_c+1 
        for cls in df_temp['Class'].unique():
            # Filter the dataframe for the current class
            df_class = df_temp[df_temp['Class'] == cls].reset_index(drop=True)
            
            # Drop unnecessary columns
            df_class = df_class.drop(columns=['channels', 'ID', 'Class', 'Epoch'])
        
            # Convert the dataframe to a numpy array
            class_array = df_class.to_numpy()
            class_array = repmat(class_array, rep_factor, 1)
            class_array = class_array[:max_c,:]
            class_arrays.append(class_array)
        print(f'ID{ID:02d}: Repetition factor {rep_factor:02d} - Max is: {max_c:02d} - Min is: {min_c:02d}') 
        classes = classes+ [clusters[ID-1]]*max_c
        concatenated_data.append(np.concatenate((class_arrays),axis=1).tolist())
    final_array = np.concatenate((concatenated_data))

    channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'AFz', 'CPz', 'POz']
    patterns = ['_d1', '_d2', '_d3', '_d4', '_d5', '_a1']
    mod_intensity = ['_A1','_A2','_A3','_A4','_V1','_V2','_V3','_V4','_C1','_C2','_C3','_C4']
    column_names = [f'{ch}_{pt}{mi}' for ch in channels for mi in mod_intensity for pt in patterns ]
    
    # Convert the final array to a dataframe
    final_df = pd.DataFrame(final_array, columns = column_names)

    return final_df, classes

def print_scores(name, scores):
    mean_test_accuracy = np.mean(scores['test_accuracy']) * 100
    std_test_accuracy = np.std(scores['test_accuracy']) * 100
    mean_test_precision = np.mean(scores['test_precision'])
    mean_test_recall = np.mean(scores['test_recall'])
    mean_test_f1 = np.mean(scores['test_f1'])
    mean_test_roc_auc = np.mean(scores['test_roc_auc'])
    
    print(f"------------------------------------------------------------")
    print(f"Average K-Fold Test Accuracy of {name}: {mean_test_accuracy:.2f}%")
    print(f"Std Dev of K-Fold Test Accuracy of {name}: {std_test_accuracy:.2f}%")
    print(f"Average K-Fold Test Precision of {name}: {mean_test_precision:.2f}")
    print(f"Average K-Fold Test Recall of {name}: {mean_test_recall:.2f}")
    print(f"Average K-Fold Test F1 Score of {name}: {mean_test_f1:.2f}")
    print(f"Average K-Fold Test ROC AUC of {name}: {mean_test_roc_auc:.2f}")
    print(f"------------------------------------------------------------")
    
def RForest_train(X_train, y_train, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', validation='standard', cv=5, echo=True):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                   max_features=max_features, random_state=42)
    
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)

    model.fit(X_train, y_train)
    
    if echo:
        print_scores('Random Forest', scores)
    return scores, model


def KNN_train(X_train, y_train, hot_encode=False, n_neighbors=20, p=1, weights='distance', leaf_size=20, algorithm='auto', validation='standard', cv=5, echo=True):
    model = KNeighborsClassifier(n_neighbors=n_neighbors, p=p, weights=weights, leaf_size=leaf_size, algorithm=algorithm)

    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)
    
    model.fit(X_train, y_train)
    
    if echo:
        print_scores('KNN', scores)
    return scores, model


def MLP_train(X_train, y_train,  hot_encode=False, hidden_layer_sizes=(100,), activation='relu', solver='adam', validation='standard', alpha=0.001, cv=5, max_iter=200, echo=True):
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, batch_size='auto', learning_rate='constant', 
                              learning_rate_init=0.001, power_t=0.5, max_iter=max_iter, shuffle=True, random_state=None, tol=0.0001, verbose=False, 
                              warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, 
                              beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    scores = cross_validate(model,X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)
    
    model.fit(X_train, y_train)

    if echo:
        print_scores('MLP', scores)
    return scores, model

def train_all(X_train, y_train, cv = 10, echo = True):

    models = []
    scores = []
    
    score, model = KNN_train(X_train, y_train, cv = cv, echo = echo)
    models.append(model)
    scores.append(score)
    
    score, model =RForest_train(X_train, y_train,  cv = cv, echo = echo)
    models.append(model)
    scores.append(score)
    
    score, model = MLP_train(X_train, y_train, hidden_layer_sizes=(200, 100, 50), max_iter=10000, alpha=0.001, cv=cv, echo = echo)
    models.append(model)
    scores.append(score)
    
    return models, scores

def calculate_acc(X_test, y_test, models):
    
    le = LabelEncoder()
    y_test_encoded = le.fit_transform(y_test)
    ensemble_preds = np.zeros((len(y_test_encoded), len(models)))
    accs = []
    # Loop over each model to get predictions and calculate individual accuracies
    for i, (model, name) in enumerate(zip(models, ['KNN', 'Random Forest','MLP'])):
        pred = model.predict(X_test)
        
        # Convert string predictions to numerical form
        pred_encoded = le.transform(pred)
        
        acc = np.mean(pred_encoded == y_test_encoded)
        accs.append(acc)
        print(f"{name} Classification Accuracy: {acc}")
        
        # Store the numerical predictions for ensemble averaging
        ensemble_preds[:, i] = pred_encoded

    # Average the predictions using mode for classification|
    ensemble_final_preds_encoded = mode(ensemble_preds, axis=1)[0].flatten()
    
    # Convert numerical predictions back to string labels
    ensemble_final_preds = le.inverse_transform(ensemble_final_preds_encoded.astype(int))
    
    # Calculate the ensemble accuracy
    ensemble_acc = np.mean(ensemble_final_preds == y_test)
    print(f"Ensemble Classification Accuracy: {ensemble_acc}")
    return accs, ensemble_preds

def get_mutualInfo(X_train, X_test, y_train, number_of_f=100):
    # Calculate mutual information scores
    mutual_info_scores = mutual_info_classif(X_train, y_train)
    if number_of_f > 0:
        # Sort the mutual information scores in descending order and select the top n features
        sorted_indices = np.argsort(mutual_info_scores)[::-1]
        selected_features = sorted_indices[:number_of_f]
    else:
        raise ValueError("Number of features must be greater than 0.")

    # Filter the training and test sets to include only the selected features
    X_train_sf = X_train[:, selected_features]
    X_test_sf = X_test[:, selected_features]

    # Define the channels and patterns
    channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'AFz', 'CPz', 'POz']
    patterns = ['d1', 'd2', 'd3', 'd4', 'd5', 'a1']
    mod_intensity = ['A1','A2','A3','A4','V1','V2','V3','V4','C1','C2','C3','C4']
    feature_names = [f'{ch}_{pt}_{mi}' for ch in channels for pt in patterns for mi in mod_intensity]
    
    # Ensure that feature_names has the same length as mutual_info_scores
    if len(feature_names) != len(mutual_info_scores):
        raise ValueError(f"The number of generated feature names ({len(feature_names)}) does not match the number of features ({len(mutual_info_scores)}) in the data.")
    
    # Print the mutual information scores of the top n features
    print(f"Top {number_of_f} most important features:")
    for i in selected_features:
        print(f"{feature_names[i]}: Mutual Information Score: {mutual_info_scores[i]:.5f}")
    
    # Print the number of selected features
    print(f"Number of features selected: {len(selected_features)}")
    
    return X_train_sf, X_test_sf
