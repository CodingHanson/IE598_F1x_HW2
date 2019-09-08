import pandas as pd

df = pd.read_csv('C:/Users/wangh/Desktop/MSFE/machine_learning/HW2/Treasury_Squeeze_test.csv')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
# Create arrays for the features and the response variable
y = df['squeeze'].values
X = df.drop(['rowindex','contract','squeeze'], axis = 1).values

# using 0,3 for the size of the test and use a random state of 16
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=16, stratify=y)
# Create a k-NN classifier with multiple values of neighbors
k_range = range(1,26)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    # Predict the labels for the training data X_test
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))

max_k = scores.index(max(scores)) +1
max_score = max(scores)
print("The best performance k for KNN is {} and its score is {}".format(max_k,max_score))
import matplotlib.pyplot as plt
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(k_range, scores, label = 'Accuracy Scores')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
print(list(scores))

#Classification Tree
from sklearn.model_selection import cross_val_score
SEED = 16
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=6, random_state= SEED)
dt.fit(X_train, y_train)
# Predict test set labels
y_pred = dt.predict(X_test)
print(y_pred[0:5])
acc = metrics.accuracy_score(y_test, y_pred)
print("Test set accuracy: {:.2f}".format(acc))

#Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=1)
log_reg.fit(X_train,y_train)
y_pred_log = dt.predict(X_test)
print(y_pred_log[0:5])
acc_log = metrics.accuracy_score(y_test, y_pred_log)
print("Test set log_accuracy: {:.2f}".format(acc_log))

#using entropy/ Gini
dt_entropy = DecisionTreeClassifier(max_depth=8, criterion='entropy', random_state=1)
# Fit dt_entropy to the training set
dt_entropy.fit(X_train, y_train)
# Use dt_entropy to predict test set labels
y_pred= dt_entropy.predict(X_test)
# Evaluate accuracy_entropy
accuracy_entropy = metrics.accuracy_score(y_test,y_pred)
# Print accuracy_entropy
print('Accuracy achieved by using entropy: ', accuracy_entropy)
dt_gini = DecisionTreeClassifier(max_depth=8, criterion='gini', random_state=1)
# Fit dt_entropy to the training set
dt_gini.fit(X_train, y_train)
# Use dt_entropy to predict test set labels
y_pred_gini= dt_gini.predict(X_test)
# Evaluate accuracy_entropy
accuracy_gini = metrics.accuracy_score(y_test,y_pred)
# Print accuracy_gini
print('Accuracy achieved by using the gini index: ', accuracy_gini)

#using regression tree
from sklearn.tree import DecisionTreeRegressor

# Instantiate dt
dt_reg = DecisionTreeRegressor(max_depth=8,
             min_samples_leaf=0.13,
            random_state=3)
# Fit dt to the training set
dt_reg.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error as MSE
# Compute y_pred
y_pred_reg = dt_reg.predict(X_test)
# Compute mse_dt
mse_dt = MSE(y_test, y_pred_reg)
# Compute rmse_dt
rmse_dt = mse_dt **(1/2)
# Print rmse_dt
print("Test set RMSE of dt: {:.2f}".format(rmse_dt))

#compare with Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train,y_train)
# Predict test set labels
y_pred_lr = lr.predict(X_test)
# Compute mse_lr
mse_lr = MSE(y_test, y_pred_lr)
# Compute rmse_lr
rmse_lr = mse_lr **(1/2)
# Print rmse_lr
print('Test set RMSE of lr: {:.2f}'.format(rmse_lr))

#Random_Forest
# Import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
# Instantiate rf
rf = RandomForestRegressor(n_estimators=25,
                           random_state=2)
# Fit rf to the training set
rf.fit(X_train, y_train)
# Predict the test set labels
y_pred = rf.predict(X_test)
# Evaluate the test set RMSE
rmse_test = MSE(y_test,y_pred) ** (1/2)
# Print rmse_test
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))

#AdaBoost Classifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
ada = AdaBoostClassifier(base_estimator=dt, n_estimators=180, random_state=1)
# Fit ada to the training set
ada.fit(X_train,y_train)
# Compute the probabilities of obtaining the positive class
y_pred_proba = ada.predict_proba(X_test)[:,1]
# Evaluate test-set roc_auc_score
ada_roc_auc = roc_auc_score(y_test, y_pred_proba)
# Print roc_auc_score
print('ROC AUC score: {:.2f}'.format(ada_roc_auc))

#GB regressor
from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor(max_depth=4,
            n_estimators=200,
            random_state=2)
# Fit gb to the training set
gb.fit(X_train,y_train)
# Predict test set labels
y_pred_GB = gb.predict(X_test)
# Compute MSE
mse_test_GB = MSE(y_test,y_pred_GB)
# Compute RMSE
rmse_test_GB = mse_test_GB ** (1/2)
# Print RMSE
print('Test set RMSE of gb: {:.3f}'.format(rmse_test_GB))

#Stochastic Gradient Boosting (SGB)
from sklearn.ensemble import GradientBoostingRegressor
sgbr = GradientBoostingRegressor(max_depth=4,
            subsample=0.9,
            max_features=0.75,
            n_estimators=200,
            random_state=2)
# Fit sgbr to the training set
sgbr.fit(X_train,y_train)
# Predict test set labels
y_pred_SGB = sgbr.predict(X_test)
# Compute test set MSE
mse_test_SGB = MSE(y_test,y_pred_SGB)
# Compute test set RMSE
rmse_test_SGB = mse_test_SGB ** (1/2)
# Print rmse_test
print('Test set RMSE of sgbr: {:.3f}'.format(rmse_test_SGB))

print("-------------------------------------------------------------------------")
print("My name is Han Wang")
print("My NetID is: 'hanw8'")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")