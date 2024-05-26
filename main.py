# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from neo4j import GraphDatabase
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



driver = GraphDatabase.driver("bolt://localhost:7687", auth = ("neo4j","12345678"))

with driver.session() as session:
    result = session.run('''MATCH (n:Account) RETURN n.accountId AS id, n.customerId AS customer_id, n.initBalance AS 
    init_balance, n.tx_behavior_id AS tx_behavior_id, n.isFraud AS isFraud ''')

    data = result.data()

data = np.array([[row["id"],row["customer_id"],row["init_balance"],row["tx_behavior_id"],row["isFraud"]] for row in data])


X = data[:,:-1]
Y = data[:,-1]
label_encoders = {}
X_encoded = np.empty(X.shape)

# Кодирование 'customer_id'
label_encoders['customer_id'] = LabelEncoder()
X_encoded[:, 1] = label_encoders['customer_id'].fit_transform(X[:, 1])

# Один шаг кодирования
one_hot_encoder = OneHotEncoder()
X_encoded = one_hot_encoder.fit_transform(X_encoded)

# Теперь вы можете разделить свои данные и подогнать модель
Y = np.array(Y, dtype=int)

X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:,1]

f1 = f1_score(Y_test,y_pred, average="weighted")

accuracy = accuracy_score(Y_test, y_pred)
precision = precision_score(Y_test, y_pred, zero_division=1)
recall = recall_score(Y_test, y_pred,zero_division=1)

print("f1_scores ", f1)
print("accuracy_score ",accuracy)
print("presition_score ", precision)
print("recall_score ", recall)

fpr, tpr, thresholds = roc_curve(Y_test, y_pred_prob)

roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0,1],[0,1],label='Random guess', linestyle='--' )
plt.xlim([0.0,1.0])
plt.ylim([0.0,0.03])
plt.xlabel("False positive Rate")
plt.ylabel("True positive Rate")
plt.title("Receiver characteristic")
plt.legend(loc='lower right')

metrics = ['Accuracy','Precision','Recall','F1_score']
values = [accuracy, precision, recall, f1]

plt.bar(range(len(metrics)), values, tick_label=metrics)

plt.title('Model perfomance')
plt.xlabel('Metrics')
plt.ylabel('Values')

plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
