import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, confusion_matrix

class Terminator:
    def __init__(self, filename,feature_select = 'entropy', Hyperparam = False, method = 'dwt_energy', 
                 Types = [1,2,3,4,5,6,7,8,9,10,11,12], mod = ['Air','Air','Air','Air','Vib','Vib','Vib','Vib','Car','Car','Car','Car']):          

        if method == 'dwt_energy':
            self.read_dwt(filename, mod, Types)
        else:
            self.read_data(filename)
            self.filter_data(5, Types)
            self.class_generator(mod)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_array, 
                                                                                    self.class_labels, test_size=0.2, random_state=42)      
        self.MLP_train(validation='kfold', hidden_layer_sizes=(200,100,50,20,10,), max_iter = 10000)
 
        if feature_select == 'entropy':
            self.feature_select_entropy(threshold=0.001, echo = False, plot = True, number_of_f = 70)
            self.MLP_train(validation='kfold', hidden_layer_sizes=(200,100,50,20,10,), max_iter = 10000)
        if Hyperparam:     
            self.HyperParametrizeMLP()
        #    self.HyperParametrizeKNN()

## Preprocessing

    def read_data(self, filename):
        self.df = pd.read_csv(filename)
        new_column_names = ['channels', 'ID', 'Class', 'Epoch']
        self.df.rename(columns=dict(zip(self.df.columns[:4], new_column_names)), inplace=True)
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

    def filter_data(self, channel, Types):
        filtered_class_labels = self.df[(self.df['channels'] == channel) & (self.df['Class'].isin(Types))]
        filtered_class_labels = filtered_class_labels.drop(columns=['ID', 'channels', 'Epoch'])
        self.df = self.df.drop(columns=['channels', 'ID', 'Epoch'])
        self.class_labels_original = filtered_class_labels.reset_index(drop=True)
        self.data_array = filtered_class_labels.iloc[:, 1:].values

    def class_generator(self, mod):
        class_labels = self.class_labels_original['Class'].copy()
        for i, replacement in enumerate(mod, start=1):
            class_labels[class_labels == i] = replacement
        self.class_labels = class_labels

    def read_dwt(self, filename, mod, Types):
        self.df = pd.read_csv(filename)
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(inplace=True)

        channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'AFz', 'CPz', 'POz']
        patterns = ['_d1', '_d2', '_d3', '_d4', '_d5', '_a1']
        column_names = ['channels', 'ID', 'Class', 'Epoch'] + [f'{ch}{pt}' for ch in channels for pt in patterns]

        if len(column_names) != len(self.df.columns):
            raise ValueError(f"Number of column names ({len(column_names)}) does not match number of columns in DataFrame ({len(self.df.columns)})")

        self.df.columns = column_names

        class_mapping = dict(zip(Types, mod))
        self.df['Class'] = self.df['Class'].map(class_mapping)
        self.df = self.df.drop(columns=['channels', 'ID', 'Epoch'])

        self.data_array = self.df.drop(['Class'], axis=1).values
        self.class_labels = self.df['Class'].values

    def time_filter_band(self,lowcut = 2.0, highcut = 30.0, fs_original = 250.0, fs_downsampled = 60 ):
        nyquist = 0.5 * fs_original
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(4, [low, high], btype='band')
        filtered_data_array = np.apply_along_axis(lambda x: filtfilt(b, a, x), axis=1, arr=self.data_array)
        downsample_factor = int(fs_original / fs_downsampled)
        self.downsampled_data_array = resample(filtered_data_array, len(filtered_data_array[0]) // downsample_factor, axis=1)

## Feature Selection

    def feature_select_entropy(self, threshold=0.001, echo=True, plot = True, number_of_f = 0):
        print("Selecting features based on mutual information")
        mutual_info_scores = mutual_info_classif(np.vstack((self.X_train, self.X_test)), np.concatenate((self.y_train, self.y_test)))

        if number_of_f > 0:
            # Select the top number_of_f features based on mutual information
            sorted_indices = np.argsort(mutual_info_scores)[::-1]
            selected_features = sorted_indices[:number_of_f]
        else:
            # Select features based on the threshold
            selected_features = np.where(mutual_info_scores > threshold)[0]

        channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'AFz', 'CPz', 'POz']
        patterns = ['d1', 'd2', 'd3', 'd4', 'd5', 'a1']
        feature_names = [f'{ch}_{pt}' for ch in channels for pt in patterns]

        if echo:
            for feature, score in zip(feature_names, mutual_info_scores):
                print(f"{feature}: Mutual Information Score: {score:.5f}")

        self.X_train = self.X_train[:, selected_features]
        self.X_test = self.X_test[:, selected_features]

        if plot:
            print(f"Number of features selected: {len(selected_features)}")
            scores_matrix = mutual_info_scores.reshape(len(channels), len(patterns))
            scores_df = pd.DataFrame(scores_matrix, index=channels, columns=patterns)
            plt.figure(figsize=(10, 8))
            sns.heatmap(scores_df, annot=True, fmt=".3f", cmap="YlGnBu")
            plt.title('Heatmap of Mutual Information Scores')
            plt.xlabel('Patterns')
            plt.ylabel('Channels')
            plt.show()

    def feature_select_fdr(self, echo=True):
        print(f"Selected features based on FDR values higher then median")
        fdr_values = {}

        classes = self.df['Class'].unique()

        for column in self.df.columns[1:]:  
            sum_of_mean_diffs_squared = 0
            sum_of_variances = 0

            overall_mean = self.df[column].mean()

            for cls in classes:
                class_subset = self.df[self.df['Class'] == cls][column]

                class_mean = class_subset.mean()
                class_variance = class_subset.var()

                n_k = len(class_subset)

                sum_of_mean_diffs_squared += n_k * (class_mean - overall_mean) ** 2
                sum_of_variances += (n_k - 1) * class_variance

            # Calculate FDR for the current feature
            fdr = sum_of_mean_diffs_squared / sum_of_variances if sum_of_variances != 0 else 0
            fdr_values[column] = fdr

        # Sort features by FDR in descending order and get their indices
        self.sorted_features = sorted(fdr_values.items(), key=lambda x: x[1], reverse=True)
        sorted_feature_indices = [self.df.columns[1:].get_loc(feature) for feature, _ in self.sorted_features]

        # Print the sorted features, their FDR values, and indices
        if echo:
            print("Features sorted by FDR (in descending order):")
            for i, (feature, value) in enumerate(self.sorted_features):
                print(f"{i}. {feature} (Index: {sorted_feature_indices[i]}) - FDR: {value}")

        # Update X_train and X_test to only include selected features
        self.X_train = self.X_train[:, sorted_feature_indices]
        self.X_test = self.X_test[:, sorted_feature_indices]
    
    def select_features_above_median_fdr(self):

        # Find the median FDR value
        fdr_values = [fdr for _, fdr in self.sorted_features]
        median_fdr = np.median(fdr_values)

        # Select features above the median FDR
        selected_features = [feature for feature, value in self.sorted_features if value > median_fdr]

        # Get the indices of the selected features
        selected_indices = [self.df.columns[1:].get_loc(feature) for feature in selected_features]
        
        # Assuming self.X_train and self.X_test are numpy arrays
        self.X_train = self.X_train[:, selected_indices]
        self.X_test = self.X_test[:, selected_indices]

    def apply_pca_and_select_components(self):
        pca = PCA().fit(self.X_train)
        
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.where(cumulative_variance >= 0.95)[0][0] + 1  # Add 1 because indices start at 0
        
        # Refit PCA with the selected number of components
        pca = PCA(n_components=n_components).fit(self.X_train)
        self.X_train = pca.transform(self.X_train)
        self.X_test = pca.transform(self.X_test)

        # Step 4: Reconstruction is not typical with PCA for feature selection. If needed:
        self.X_train = pca.inverse_transform(self.X_train)
        self.X_test = pca.inverse_transform(self.X_test)

        print(f"Number of principal components selected: {n_components}")

## Training methods

    def RForest_train(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', validation='standard', n_splits=5):
        rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                          min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                          max_features=max_features, random_state=42)
        
        if validation == 'standard':
            rf_model.fit(self.X_train, self.y_train)
            self.y_pred = rf_model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, self.y_pred)
            print(f"------------------------------------------------------------")
            print(f"Accuracy of Random Forest: {accuracy * 100:.2f}%")
            self.conf_mat()
            print(f"------------------------------------------------------------")
            return accuracy * 100
        elif validation == 'kfold':
            scores = cross_validate(rf_model, np.vstack((self.X_train, self.X_test)), np.concatenate((self.y_train, self.y_test)), cv=n_splits, scoring='accuracy', return_train_score=True)
            print(f"------------------------------------------------------------")
            print(f"Average K-Fold Test Accuracy of Random Forest: {np.mean(scores['test_score']) * 100:.2f}%")
            print(f"Average K-Fold Train Accuracy of Random Forest: {np.mean(scores['train_score']) * 100:.2f}%")
            print(f"Std Dev of K-Fold Test Accuracy of Random Forest: {np.std(scores['test_score']) * 100:.2f}%")
            print(f"Std Dev of K-Fold Train Accuracy of Random Forest: {np.std(scores['train_score']) * 100:.2f}%")
            print(f"------------------------------------------------------------")
            return np.mean(scores['train_score']) * 100
        else:
            print("Invalid validation method specified.")
    
    def KNN_train(self, n_neighbors=20, p=1, weights='distance', leaf_size=20, algorithm='auto', validation='standard', n_splits=5):
            knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, p=p, weights=weights, leaf_size=leaf_size, algorithm=algorithm)
            
            if validation == 'standard':
                knn_model.fit(self.X_train, self.y_train)
                self.y_pred = knn_model.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, self.y_pred)
                print(f"------------------------------------------------------------")
                print(f"Accuracy of KNN: {accuracy * 100:.2f}%")
                self.conf_mat()
                print(f"------------------------------------------------------------")
                return accuracy * 100
            elif validation == 'kfold':
                scores = cross_validate(knn_model, np.vstack((self.X_train, self.X_test)), np.concatenate((self.y_train, self.y_test)), cv=n_splits, scoring='accuracy', return_train_score=True)
                print(f"------------------------------------------------------------")
                print(f"Average K-Fold Test Accuracy of KNN: {np.mean(scores['test_score']) * 100:.2f}%")
                print(f"Average K-Fold Train Accuracy of KNN: {np.mean(scores['train_score']) * 100:.2f}%")
                print(f"Std Dev of K-Fold Test Accuracy of KNN: {np.std(scores['test_score']) * 100:.2f}%")
                print(f"Std Dev of K-Fold Train Accuracy of KNN: {np.std(scores['train_score']) * 100:.2f}%")
                print(f"------------------------------------------------------------")
                return np.mean(scores['train_score']) * 100
            else:
                print("Invalid validation method specified.")         

    def SVM_train(self, C=1.0, kernel='rbf', gamma='scale', validation='standard', n_splits=5):
        svm_model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)

        if validation == 'standard':
            svm_model.fit(self.X_train, self.y_train)
            self.y_pred = svm_model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, self.y_pred)
            print(f"------------------------------------------------------------")
            print(f"Accuracy of SVM: {accuracy * 100:.2f}%")
            self.conf_mat()
            print(f"------------------------------------------------------------")
            return accuracy * 100
        elif validation == 'kfold':
            scores = cross_validate(svm_model, np.vstack((self.X_train, self.X_test)), np.concatenate((self.y_train, self.y_test)), cv=n_splits, scoring='accuracy', return_train_score=True)
            print(f"------------------------------------------------------------")
            print(f"Average K-Fold Test Accuracy of SVM: {np.mean(scores['test_score']) * 100:.2f}%")
            print(f"Average K-Fold Train Accuracy of SVM: {np.mean(scores['train_score']) * 100:.2f}%")
            print(f"Std Dev of K-Fold Test Accuracy of SVM: {np.std(scores['test_score']) * 100:.2f}%")
            print(f"Std Dev of K-Fold Train Accuracy of SVM: {np.std(scores['train_score']) * 100:.2f}%")
            print(f"------------------------------------------------------------")
            return np.mean(scores['test_score']) * 100
        else:
            print("Invalid validation method specified.")

    def MLP_train(self, hidden_layer_sizes=(100,), activation='relu', solver='adam', validation='standard', n_splits=5, max_iter = 200):
        mlp_model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=0.001, batch_size='auto', learning_rate='constant', 
                                  learning_rate_init=0.001, power_t=0.5, max_iter=max_iter, shuffle=True, random_state=None, tol=0.0001, verbose=False, 
                                  warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, 
                                  beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

        if validation == 'standard':
            mlp_model.fit(self.X_train, self.y_train)
            self.y_pred = mlp_model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, self.y_pred)
            print(f"------------------------------------------------------------")
            print(f"Accuracy of MLP: {accuracy * 100:.2f}%")
            self.conf_mat()
            print(f"------------------------------------------------------------")
            return accuracy * 100
        elif validation == 'kfold':
            scores = cross_validate(mlp_model, np.vstack((self.X_train, self.X_test)), np.concatenate((self.y_train, self.y_test)), cv=n_splits, scoring='accuracy', return_train_score=True)
            print(f"------------------------------------------------------------")
            print(f"Average K-Fold Test Accuracy of MLP: {np.mean(scores['test_score']) * 100:.2f}%")
            print(f"Average K-Fold Train Accuracy of MLP: {np.mean(scores['train_score']) * 100:.2f}%")
            print(f"Std Dev of K-Fold Test Accuracy of MLP: {np.std(scores['test_score']) * 100:.2f}%")
            print(f"Std Dev of K-Fold Train Accuracy of MLP: {np.std(scores['train_score']) * 100:.2f}%")
            print(f"------------------------------------------------------------")
            return np.mean(scores['test_score']) * 100
        else:
            print("Invalid validation method specified.")

    def GaussianNB_train(self, validation='standard', n_splits=5):
        nb_model = GaussianNB()

        if validation == 'standard':
            nb_model.fit(self.X_train, self.y_train)
            self.y_pred = nb_model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, self.y_pred)
            print(f"------------------------------------------------------------")
            print(f"Accuracy of GaussianNB: {accuracy * 100:.2f}%")
            self.conf_mat()
            print(f"------------------------------------------------------------")
            return accuracy * 100
        elif validation == 'kfold':
            scores = cross_validate(nb_model, np.vstack((self.X_train, self.X_test)), np.concatenate((self.y_train, self.y_test)), cv=n_splits, scoring='accuracy', return_train_score=True)
            print(f"------------------------------------------------------------")
            print(f"Average K-Fold Test Accuracy of GaussianNB: {np.mean(scores['test_score']) * 100:.2f}%")
            print(f"Average K-Fold Train Accuracy of GaussianNB: {np.mean(scores['train_score']) * 100:.2f}%")
            print(f"Std Dev of K-Fold Test Accuracy of GaussianNB: {np.std(scores['test_score']) * 100:.2f}%")
            print(f"Std Dev of K-Fold Train Accuracy of GaussianNB: {np.std(scores['train_score']) * 100:.2f}%")
            print(f"------------------------------------------------------------")
            return np.mean(scores['test_score']) * 100
        else:
            print("Invalid validation method specified.")

    def plot_featureselection(self, filename):
            knn_acc = []
            rf_acc = []
            #SVM_acc = []
            MLP_acc = []
            #NB_acc = []

            for number_of_f in np.arange(1, 130, 5): 
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_array, 
                                                                                        self.class_labels, test_size=0.2, random_state=42)   
                self.feature_select_entropy(echo=False, plot=False, number_of_f=number_of_f)
                knn_accuracy = self.KNN_train() 
                rf_accuracy = self.RForest_train()  
                #svm_accuracy = self.SVM_train()
                mlp_accuracy = self.MLP_train(hidden_layer_sizes=(200,100,50,20,10,), max_iter = 10000)
                #NB_accuracy = self.GaussianNB_train()

                knn_acc.append(knn_accuracy)
                rf_acc.append(rf_accuracy)
                #SVM_acc.append(svm_accuracy)
                MLP_acc.append(mlp_accuracy)
                #NB_acc.append(NB_accuracy)

            plt.figure(figsize=(10, 8))
            
            sns.set_theme(style="darkgrid")

            sns.lineplot(x = np.arange(1, 130, 5), y = knn_acc, label='KNN')
            sns.lineplot(x = np.arange(1, 130, 5), y = rf_acc, label='Random Forest')
            #sns.lineplot(x = np.arange(1, 130, 5), y = SVM_acc, label='SVM')
            sns.lineplot(x = np.arange(1, 130, 5), y = MLP_acc, label='MLP')
            #sns.lineplot(x = np.arange(1, 130, 5), y = NB_acc, label='NB')

            plt.title('Accuracy based on the number of retained features')
            plt.xlabel('Number of features')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()

## Hyperparametrization

    def HyperParametrizeRForest(self):
        print(f"Computing Hyperparametrization of random forest classifier ")
        param_grid = {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }

        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=10, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)

        best_params = grid_search.best_params_
        print("Best parameters for Random Forest:", best_params)
        print("Best cross-validation accuracy: {:.2f}%".format(grid_search.best_score_ * 100))

        self.RForest_train(**best_params)

    def HyperParametrizeKNN(self):
        print(f"Computing Hyperparametrization of KNN classifier ")

        param_grid = {
            'n_neighbors': list(range(10, 21)),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [20, 30, 40],
            'p': [1, 2]
        }

        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        
        best_params = grid_search.best_params_
        print("Best parameters:", best_params)
        print("Best cross-validation accuracy: {:.2f}%".format(grid_search.best_score_ * 100))
        self.KNN_train(**best_params)

    def HyperParametrizeMLP(self):
        print(f"Computing Hyperparametrization of MLP classifier ")

        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100),(200, 100), (200, 100, 50), (200, 100, 50, 20) ],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001],  # L2 penalty (regularization term) parameter
            'learning_rate': ['constant', 'adaptive'],
        }

        mlp = MLPClassifier(max_iter=5000, random_state=42)

        grid_search = GridSearchCV(mlp, param_grid, n_jobs=-1, cv=5, scoring='accuracy')

        grid_search.fit(self.X_train, self.y_train)
        best_params = grid_search.best_params_

        print("Best parameters found: ", best_params)
        print("Best cross-validation accuracy: {:.2f}%".format(grid_search.best_score_ * 100))
        self.MLP_train(**best_params)

        return grid_search.best_estimator_
    

    def conf_matold(self, porcentage = False):
        cm = confusion_matrix(self.y_test, self.y_pred)


        labels = ['Air', 'Vib', 'Car']  
        print(f"{'':10}{labels[0]:^10}{labels[1]:^10}{labels[2]:^10}")
        for i, row in enumerate(cm):
            print(f"{labels[i]:10}{row[0]:^10}{row[1]:^10}{row[2]:^10}")

    def conf_mat(self, percentage=True):
            cm = confusion_matrix(self.y_test, self.y_pred)
            if percentage:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

            labels = ['Air', 'Vib', 'Caress']
            print(f"{'':10}{labels[0]:^10}{labels[1]:^10}{labels[2]:^10}")
            for i, row in enumerate(cm):
                if percentage:
                    print(f"{labels[i]:10}{row[0]:^10.2f}{row[1]:^10.2f}{row[2]:^10.2f}")
                else:
                    print(f"{labels[i]:10}{row[0]:^10}{row[1]:^10}{row[2]:^10}")            

## Tester

if __name__ == "__main__":
    t13 = Terminator("D:\shared_git\MaestriaThesis\FeaturesTabs\pp01_t13.csv", method = 'dwt_energy',Hyperparam = False, 
                      Types = [1,2,3,4,5,6,7,8,9,10,11,12], 
                      mod = ['Air','Air','Air','Air','Vib','Vib','Vib','Vib','Car','Car','Car','Car'])

