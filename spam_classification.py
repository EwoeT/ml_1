
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import model_selection, datasets, metrics
#SVM
from sklearn.svm import SVC

data_train=np.genfromtxt("/home/ewoe/Downloads/sklearn-lab-material/spambase/train-data.csv",delimiter=",")
targets_train=np.genfromtxt("/home/ewoe/Downloads/sklearn-lab-material/spambase/train-targets.csv")
data_test=np.genfromtxt("/home/ewoe/Downloads/sklearn-lab-material/spambase/test-data.csv",delimiter=",")
X, y, X_test_data = data_train, targets_train,data_test
data_test.shape


# In[2]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Let's check the length of the two sets
len(X_train), len(X_test)


# In[3]:


from sklearn.svm import SVC

# Specify the parameters in the constructor.
# C is the parameter of the primal problem of the SVM;
# The rbf kernel is the Radial Basis Function;
# The rbf kernel takes one parameter: gamma
clf = SVC(C=10, kernel='rbf', gamma=0.02)


# In[4]:


from sklearn.model_selection import KFold, cross_val_score

# 3-fold cross-validation
# random_state ensures same split for each value of gamma
kf = KFold(n_splits=3, shuffle=True, random_state=42)

gamma_values = [0.1, 0.05, 0.02, 0.01]
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# Do model selection over all the possible values of gamma 
for gamma in gamma_values:
    
    # Train a classifier with current gamma
    clf = SVC(C=10, kernel='rbf', gamma=gamma)

    # Compute cross-validated accuracy scores
    scores_accuracy = cross_val_score(clf, X_train, y_train, cv=kf.split(X_train), scoring='accuracy')
    scores_precision = cross_val_score(clf, X_train, y_train, cv=kf.split(X_train), scoring='precision')
    scores_recall = cross_val_score(clf, X_train, y_train, cv=kf.split(X_train), scoring='recall')
    scores_f1 = cross_val_score(clf, X_train, y_train, cv=kf.split(X_train), scoring='f1')
    
    # Compute the mean accuracy and keep track of it
    accuracy_score = scores_accuracy.mean()
    accuracy_scores.append(accuracy_score)
    
    # Compute the mean precision and keep track of it
    precision_score = scores_precision.mean()
    precision_scores.append(precision_score)
    
    # Compute the mean recall and keep track of it
    recall_score = scores_recall.mean()
    recall_scores.append(recall_score)
    
    # Compute the mean f1 and keep track of it
    f1_score = scores_f1.mean()
    f1_scores.append(f1_score)
    
#Mean evaluation scores
print("average accuracy ", accuracy_score)
print("average precision ", precision_score)
print("average recall ", recall_score)
print("average f1  ", f1_score)

# Get the gamma with highest mean accuracy
best_index = np.array(accuracy_scores).argmax()
best_gamma = gamma_values[best_index]
print("gamma with highest mean accuracy ", best_gamma)


# Train over the full training set with the best gamma
clf = SVC(C=10, kernel='rbf', gamma=best_gamma)
clf.fit(X_train, y_train)







# In[5]:


# Evaluate on the test set
y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)

accuracy


# In[6]:


from sklearn import metrics

report = metrics.classification_report(y_test, y_pred)

# the support is the number of instances having the given label in y_test
print(report)


# In[ ]:





# In[7]:


from sklearn.model_selection import learning_curve

plt.figure()
plt.title("Learning curve")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.grid()

clf = SVC(C=10, kernel='rbf', gamma=best_gamma)

# Compute the scores of the learning curve
# by default the (relative) dataset sizes are: 10%, 32.5%, 55%, 77.5%, 100% 
train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train, scoring='accuracy')

# Get the mean and std of train and test scores along the varying dataset sizes
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot the mean and std for the training scores
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")

# Plot the mean and std for the cross-validation scores
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")

plt.legend()
plt.show()


# In[29]:



# Train over the full training set with the best gamma
clf = SVC(C=10, kernel='rbf', gamma=best_gamma)
clf.fit(data_train, targets_train)

# Predicting examples test set
y_pred = clf.predict(X_test_data)

for yp in y_pred:
    print(int(yp))

