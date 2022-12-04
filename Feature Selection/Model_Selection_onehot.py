import numpy as np
import math
import pickle
from collections import Counter
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.kernel_approximation import RBFSampler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean
list1=[]
data = pd.read_csv(r'C:\Users\Brendan\Desktop\datathon22_team02_v1.csv')
y= data[['onehot_act_1','onehot_act_2','onehot_act_3','onehot_act_4']]
x = data[["shock_index","age","gender","weight","readmission","sirs","elixhauser_vanwalraven","MechVent","heartrate","respiratoryrate","spo2","temperature","sbp","dbp","mbp","lactate","bicarbonate","pao2","paco2"
,"pH","hemoglobin","baseexcess","chloride","glucose","calcium","ionized_calcium","albumin","potassium","sodium","co2","pao2fio2ratio","wbc","platelet","bun","creatinine","ptt","pt","inr","ast","alt","bilirubin_total","gcs",
"fio2","urine_output","output_total","sofa_24hours","magnesium","bloc"]]


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=12)
sel = RandomForestClassifier(n_estimators = 100)

sel.fit(X_train, y_train)
importances= sel.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure()
plt.title("Feature importances")
plt.bar(range(x.shape[1]), importances[indices],
       color="r", align="center")
plt.xticks(range(x.shape[1]), indices)
plt.xlim([-1, x.shape[1]])
plt.axhline(y=mean(sel.feature_importances_),color='r',linestyle='-')
plt.show()