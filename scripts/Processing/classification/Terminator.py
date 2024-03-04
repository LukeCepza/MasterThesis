import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, confusion_matrix

class Terminator:
    def __init__(self, filename):
        self.read_data(filename)
        self.filter_data(5, [1,2,3,4,5,6,7,8,9,10,11,12])
        self.class_generator(['Air','Air','Air','Air','Vib','Vib','Vib','Vib','Car','Car','Car','Car'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_array, self.class_labels, test_size=0.2, random_state=42)
        self.KNN_train()
        self.RForest_train()
        self.feature_select_entropy(echo = False)
        self.KNN_train()
        self.RForest_train()       
        self.HyperParametrizeRForest()
        self.RForest_train()              
        self.HyperParametrizeKNN()
        self.KNN_train()
        
    def read_data(self, filename):
        self.df = pd.read_csv(filename)
        new_column_names = ['channels', 'ID', 'Type', 'Epoch']
        self.df.rename(columns=dict(zip(self.df.columns[:4], new_column_names)), inplace=True)
    
    def filter_data(self,channel, Types):
        filtered_class_labels = self.df[self.df['channels'] == channel]
        filtered_class_labels = self.df[self.df['Type'].isin(Types)]
        filtered_class_labels = filtered_class_labels.drop(columns=['ID'])
        filtered_class_labels = filtered_class_labels.drop(columns=['channels', 'Epoch'])
        self.class_labels_original = filtered_class_labels.reset_index(drop=True)
        self.data_array = filtered_class_labels.iloc[:, 4:].values
    
    def class_generator(self, mod):
        class_labels = self.class_labels_original['Type'].copy() 
        for i, replacement in enumerate(mod, start=1):
            class_labels[class_labels == i] = replacement
        self.class_labels = class_labels

    def time_filter_band(self,lowcut = 2.0, highcut = 30.0, fs_original = 250.0, fs_downsampled = 60 ):
        nyquist = 0.5 * fs_original
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(4, [low, high], btype='band')
        filtered_data_array = np.apply_along_axis(lambda x: filtfilt(b, a, x), axis=1, arr=self.data_array)
        downsample_factor = int(fs_original / fs_downsampled)
        self.downsampled_data_array = resample(filtered_data_array, len(filtered_data_array[0]) // downsample_factor, axis=1)

    def feature_select_entropy(self, threshold = 0.0001, echo = True):
        mutual_info_scores = mutual_info_classif(self.X_train, self.y_train)

        if echo:
            for feature, score in zip(range(self.X_train.shape[1]), mutual_info_scores):
                print(f"Feature index: {feature}, Mutual Information Score: {score}")
      
        selected_features = np.where(mutual_info_scores > threshold)[0]

        self.X_train = self.X_train[:, selected_features]
        self.X_test = self.X_test[:, selected_features]
        print(f"Selecting features based on mutual information")

    def KNN_train(self, n_neighbors=20,p=1,weights='distance',leaf_size=20,algorithm='auto'):
        knn_model = KNeighborsClassifier(n_neighbors=n_neighbors,p=p,weights=weights,leaf_size=leaf_size,algorithm=algorithm)
        knn_model.fit(self.X_train, self.y_train)
        self.y_pred = knn_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f"------------------------------------------------------------")
        print(f"Average Accuracy of KNN:  {accuracy * 100:.2f}%")
        self.conf_mat()
        print(f"------------------------------------------------------------")

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
        
    def RForest_train(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt'):
        rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                               min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                               max_features=max_features, random_state=42)
        rf_model.fit(self.X_train, self.y_train)
        self.y_pred = rf_model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f"------------------------------------------------------------")
        print(f"Average Accuracy of Random Forest: {self.accuracy * 100:.2f}%")
        self.conf_mat()
        print(f"------------------------------------------------------------")

    def HyperParametrizeRForest(self):
        print(f"Computing Hyperparametrization of random forest classifier ")
        param_grid = {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }

        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)

        best_params = grid_search.best_params_
        print("Best parameters for Random Forest:", best_params)
        print("Best cross-validation accuracy: {:.2f}%".format(grid_search.best_score_ * 100))

        self.RForest_train(**best_params)

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

if __name__ == "__main__":
    t1 = Terminator("D:\shared_git\MaestriaThesis\FeaturesTabs\pp01_t3.csv")
