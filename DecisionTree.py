df = datasets["Churn"]

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Store user ids separately
userids = df['userid']
df = df.drop(columns=['userid', 'net_transfers_prev_month'])

# Fill missing values
df['income'].fillna(df['income'].mode()[0], inplace=True)
df['aum_bucket'].fillna(df['aum_bucket'].mode()[0], inplace=True)
df.fillna(0, inplace=True)

# Create weighted columns and drop the originals
# df['time_spent_min'] = df['time_spent_min']
# df['withdrawal_over_aum__prev_month'] = df['withdrawal_over_aum__prev_month']
# df['rel_perfomance'] = df['rel_perfomance'] 
df['aum_bucket'] = df['aum_bucket']  * 5

# Prepare features and target
X = df.drop('churned_yn', axis=1)  # Features
y = df['churned_yn']  # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# max_depth = 5 for pruning
clf = DecisionTreeClassifier(class_weight='balanced', random_state=42, max_depth=5)
clf.fit(X_train, y_train)

# Predictions and evaluation
y_pred = clf.predict(X_test)

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("Report:")
print(classification_report(y_test, y_pred))

# ROC AUC score
y_pred_prob = clf.predict_proba(X_test)[:, 1]
print("ROC Score:", roc_auc_score(y_test, y_pred_prob))

# Predicting churn probabilities for the entire dataset
y_pred_prob_full = clf.predict_proba(X)[:, 1] 

#--------

X_train.columns

#--------
# Precision-Recall Tradeoff Chart

from sklearn.metrics import precision_recall_curve

# Get the probabilities for the churned class (class 1)
y_pred_prob_full = clf.predict_proba(X)[:, 1]

# Compute precision, recall, and thresholds
precision, recall, thresholds = precision_recall_curve(df['churned_yn'], y_pred_prob_full)

# Plot precision-recall curve to visualize trade-off
import matplotlib.pyplot as plt

plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Precision / Recall')
plt.legend()
plt.title('Precision-Recall Trade-off')
plt.show()

#--------
# List userids at risk of churn

threshold = 0.95  # precision vs recall trade-off

at_risk_users = userids[(y_pred_prob_full >= threshold) & (df['churned_yn'] == 0)]

print("userids of those at risk of churning (excluding already churned):")
print(at_risk_users)

len(at_risk_users)

#--------
# Format list as comma separated strings

formatted_userids = "('" + "', '".join(at_risk_users) + "')"
print(formatted_userids)

#-------
# Feature Importance

importances = clf.feature_importances_

import pandas as pd
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print(feature_importance_df)

#--------
# Tree Visualization

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(40, 60), dpi=300) 
plot_tree(clf, 
          filled=True, 
          rounded=True, 
          feature_names=X.columns, 
          class_names=['Non-Churned', 'Churned'],
          fontsize=10)
plt.show()

#---------
# Print Tree Rules as txt

from sklearn.tree import export_text

tree_rules = export_text(clf, feature_names=list(X.columns))
print(tree_rules)
