from ITMO_FS.wrappers import RecursiveElimination
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import numpy as np
dataset = make_classification(n_samples=1000, n_features=20)
data, target = np.array(dataset[0]), np.array(dataset[1])
model = SVC(kernel='linear')
rfe = RecursiveElimination(model, 5)
rfe.fit(data, target)
print("Resulting features: ", rfe.__features__)