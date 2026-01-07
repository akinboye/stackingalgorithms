'''
This program compares HBL Stacking (traditional ML algorithms) 
and EBL Stacking (ensemble algorithms) for heart disease classification.
'''
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from numpy import mean, std
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
# Boosting Algorithms
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
# Bagging Algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix,
                             matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
                             log_loss, brier_score_loss)
from sklearn.preprocessing import StandardScaler

# Generate the dataset
def getDataset():
    df = pd.read_csv("heart.csv")
    return df

def dataExploration(df):
    # Heart Disease by Gender
    plt.figure(figsize=(12,8))
    ax = sns.countplot(data=df, x="target", hue="sex", palette=["salmon", "lightblue"])
    plt.title("Heart Disease Frequency by Gender")
    plt.xlabel("Target (0 = No Disease, 1 = Disease)")
    plt.ylabel("Count")
    plt.legend(["Female", "Male"])
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', label_type='edge', padding=3, fontsize=10)
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

    # Age Distribution
    plt.figure(figsize=(12,8))
    ax = sns.histplot(data=df, x="age", bins=20, kde=True, color="skyblue")
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    for patch in ax.patches:
        height = patch.get_height()
        if height > 0:
            ax.text(patch.get_x() + patch.get_width() / 2, height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9, color='black')
    plt.show()

    # Age vs. Heart Disease (Gender Differentiated)
    plt.figure(figsize=(12,8))
    sns.boxplot(data=df, x="target", y="age", hue="sex", palette=["salmon", "lightblue"])
    plt.title("Age Distribution by Heart Disease and Gender")
    plt.xlabel("Target (0 = No Disease, 1 = Disease)")
    plt.ylabel("Age")
    plt.show()

    # Cholesterol Distribution
    plt.figure(figsize=(12,8))
    ax = sns.histplot(data=df, x="chol", bins=25, kde=True, color="orange")
    plt.title("Cholesterol Level Distribution")
    plt.xlabel("Cholesterol")
    plt.ylabel("Frequency")
    for patch in ax.patches:
        height = patch.get_height()
        if height > 0:
            ax.text(patch.get_x() + patch.get_width() / 2, height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9, color='black')
    plt.show()

    # Age vs Max Heart Rate by Heart Disease
    plt.figure(figsize=(12,8))
    sns.scatterplot(data=df, x="age", y="thalach", hue="target", palette="coolwarm")
    plt.title("Age vs. Maximum Heart Rate by Heart Disease Status")
    plt.xlabel("Age")
    plt.ylabel("Max Heart Rate")
    plt.show()

def splitDataset(df):
    X = df.drop("target", axis=1)
    y = df.target.values
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_Val, X_test, y_Val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Scale the data for better convergence
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_Val = scaler.transform(X_Val)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_Val, y_Val, X_test, y_test

def dataPreprocess(df):
    df.target.value_counts(normalize=True)
    return df

# Define HBL base models (traditional ML algorithms)
def get_hbl_base_models():
    models = dict()
    models['lr'] = LogisticRegression(max_iter=1000, random_state=42)
    models['knn'] = KNeighborsClassifier()
    models['cart'] = DecisionTreeClassifier(random_state=42)
    models['svm'] = SVC(probability=True, random_state=42)
    models['bayes'] = GaussianNB()
    return models

# Define EBL base models (ensemble algorithms)
def get_ebl_base_models():
    models = dict()
    models['rf'] = RandomForestClassifier(random_state=42)
    models['bc'] = BaggingClassifier(random_state=42)
    models['xtree'] = ExtraTreesClassifier(random_state=42)
    models['ada'] = AdaBoostClassifier(random_state=42)
    models['xgb'] = XGBClassifier(random_state=42)
    models['gbm'] = GradientBoostingClassifier(random_state=42)
    models['lgbm'] = LGBMClassifier(random_state=42)
    models['cat'] = CatBoostClassifier(random_state=42, verbose=0)
    return models

# Define HBL stacking ensemble
def get_hbl_stacking():
    level0 = list()
    level0.append(('lr', LogisticRegression(max_iter=1000, random_state=42)))
    level0.append(('knn', KNeighborsClassifier()))
    level0.append(('cart', DecisionTreeClassifier(random_state=42)))
    level0.append(('svm', SVC(probability=True, random_state=42)))
    level0.append(('bayes', GaussianNB()))
    level1 = LogisticRegression(max_iter=1000, random_state=42)
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return model

# Define EBL stacking ensemble
def get_ebl_stacking():
    level0 = list()
    level0.append(('gbm', GradientBoostingClassifier(random_state=42)))
    level0.append(('ada', AdaBoostClassifier(random_state=42)))
    level0.append(('rf', RandomForestClassifier(random_state=42)))
    level0.append(('bc', BaggingClassifier(random_state=42)))
    level1 = XGBClassifier(random_state=42)
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return model

# Load data and prepare
dataframe = getDataset()
dataExploration(dataframe)
df = dataPreprocess(dataframe)
X_train, y_train, X_Val, y_Val, X_test, y_test = splitDataset(df)

print("\n" + "="*80)
print("EVALUATING HBL STACKING (Traditional ML Algorithms)")
print("="*80)

# Evaluate HBL base models
hbl_models = get_hbl_base_models()
hbl_results_val = {}
hbl_results_test = {}

for name, model in hbl_models.items():
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_Val)
    y_test_pred = model.predict(X_test)
    hbl_results_val[name] = accuracy_score(y_Val, y_val_pred)
    hbl_results_test[name] = accuracy_score(y_test, y_test_pred)
    print(f'>HBL Base {name}: Val Acc {hbl_results_val[name]:.3f}, Test Acc {hbl_results_test[name]:.3f}')

# Evaluate HBL stacking
hbl_stacking_model = get_hbl_stacking()
hbl_stacking_model.fit(X_train, y_train)
y_val_pred_hbl_stack = hbl_stacking_model.predict(X_Val)
y_test_pred_hbl_stack = hbl_stacking_model.predict(X_test)
hbl_results_val['HBL_stacking'] = accuracy_score(y_Val, y_val_pred_hbl_stack)
hbl_results_test['HBL_stacking'] = accuracy_score(y_test, y_test_pred_hbl_stack)
print(f'>HBL Stacking: Val Acc {hbl_results_val["HBL_stacking"]:.3f}, Test Acc {hbl_results_test["HBL_stacking"]:.3f}')

print("\n" + "="*80)
print("EVALUATING EBL STACKING (Ensemble Algorithms)")
print("="*80)

# Evaluate EBL base models
ebl_models = get_ebl_base_models()
ebl_results_val = {}
ebl_results_test = {}

for name, model in ebl_models.items():
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_Val)
    y_test_pred = model.predict(X_test)
    ebl_results_val[name] = accuracy_score(y_Val, y_val_pred)
    ebl_results_test[name] = accuracy_score(y_test, y_test_pred)
    print(f'>EBL Base {name}: Val Acc {ebl_results_val[name]:.3f}, Test Acc {ebl_results_test[name]:.3f}')

# Evaluate EBL stacking
ebl_stacking_model = get_ebl_stacking()
ebl_stacking_model.fit(X_train, y_train)
y_val_pred_ebl_stack = ebl_stacking_model.predict(X_Val)
y_test_pred_ebl_stack = ebl_stacking_model.predict(X_test)
ebl_results_val['EBL_stacking'] = accuracy_score(y_Val, y_val_pred_ebl_stack)
ebl_results_test['EBL_stacking'] = accuracy_score(y_test, y_test_pred_ebl_stack)
print(f'>EBL Stacking: Val Acc {ebl_results_val["EBL_stacking"]:.3f}, Test Acc {ebl_results_test["EBL_stacking"]:.3f}')

# ============================================================================
# COMPARISON VISUALIZATIONS
# ============================================================================

# Combined summary table
print("\n" + "="*80)
print("COMPARISON: HBL vs EBL STACKING")
print("="*80)

comparison_df = pd.DataFrame({
    'Model': ['HBL_Stacking', 'EBL_Stacking'],
    'Val Accuracy': [hbl_results_val['HBL_stacking'], ebl_results_val['EBL_stacking']],
    'Test Accuracy': [hbl_results_test['HBL_stacking'], ebl_results_test['EBL_stacking']]
})
print(comparison_df.to_string(index=False))

# Side-by-side comparison chart
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# HBL models
hbl_summary = pd.DataFrame({
    'Model': list(hbl_results_test.keys()),
    'Accuracy': list(hbl_results_test.values())
})
bars1 = axes[0].barh(hbl_summary['Model'], hbl_summary['Accuracy'], color='steelblue')
axes[0].set_xlabel('Test Accuracy')
axes[0].set_title('HBL Stacking Architecture (Traditional ML)')
axes[0].set_xlim(0, 1.1)
axes[0].axvline(x=hbl_results_test['HBL_stacking'], color='red', linestyle='--', linewidth=2, label='HBL Stacking')
axes[0].legend()
# Add value labels on bars
for i, bar in enumerate(bars1):
    width = bar.get_width()
    axes[0].text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                ha='left', va='center', fontsize=9, fontweight='bold', color='black')

# EBL models
ebl_summary = pd.DataFrame({
    'Model': list(ebl_results_test.keys()),
    'Accuracy': list(ebl_results_test.values())
})
bars2 = axes[1].barh(ebl_summary['Model'], ebl_summary['Accuracy'], color='darkgreen')
axes[1].set_xlabel('Test Accuracy')
axes[1].set_title('EBL Stacking Architecture (Ensemble Algorithms)')
axes[1].set_xlim(0, 1.1)
axes[1].axvline(x=ebl_results_test['EBL_stacking'], color='red', linestyle='--', linewidth=2, label='EBL Stacking')
axes[1].legend()
# Add value labels on bars
for i, bar in enumerate(bars2):
    width = bar.get_width()
    axes[1].text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                ha='left', va='center', fontsize=9, fontweight='bold', color='black')

plt.tight_layout()
plt.show()

# Direct comparison of stacking models
plt.figure(figsize=(10, 6))
comparison_melted = comparison_df.melt(id_vars='Model', var_name='Dataset', value_name='Accuracy')
ax = sns.barplot(data=comparison_melted, x='Model', y='Accuracy', hue='Dataset', palette='Set2')
plt.title('HBL vs EBL Stacking Performance Comparison', fontsize=16, fontweight='bold')
plt.ylabel('Accuracy')
plt.ylim(0, 1.05)
plt.legend(title='Dataset')
# Add value labels on bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', label_type='edge', padding=3, fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# DETAILED METRICS COMPARISON
# ============================================================================

print("\n" + "="*80)
print("DETAILED METRICS COMPARISON")
print("="*80)

# HBL metrics
hbl_metrics_summary = []
for name, model in hbl_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = None
    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
        # Normalize decision_function output to [0, 1] range for log_loss and brier_score
        from sklearn.preprocessing import minmax_scale
        y_proba = minmax_scale(y_proba)
    
    hbl_metrics_summary.append({
        "Model": f"HBL_{name}",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced_Acc": balanced_accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Cohen_Kappa": cohen_kappa_score(y_test, y_pred),
        "Log_Loss": log_loss(y_test, y_proba) if y_proba is not None else None,
        "Brier_Score": brier_score_loss(y_test, y_proba) if y_proba is not None else None
    })

# HBL stacking metrics
y_pred_hbl_stack = hbl_stacking_model.predict(X_test)
y_proba_hbl_stack = hbl_stacking_model.predict_proba(X_test)[:, 1]
hbl_metrics_summary.append({
    "Model": "HBL_Stacking",
    "Accuracy": accuracy_score(y_test, y_pred_hbl_stack),
    "Balanced_Acc": balanced_accuracy_score(y_test, y_pred_hbl_stack),
    "Precision": precision_score(y_test, y_pred_hbl_stack),
    "Recall": recall_score(y_test, y_pred_hbl_stack),
    "F1-Score": f1_score(y_test, y_pred_hbl_stack),
    "AUC": roc_auc_score(y_test, y_proba_hbl_stack),
    "MCC": matthews_corrcoef(y_test, y_pred_hbl_stack),
    "Cohen_Kappa": cohen_kappa_score(y_test, y_pred_hbl_stack),
    "Log_Loss": log_loss(y_test, y_proba_hbl_stack),
    "Brier_Score": brier_score_loss(y_test, y_proba_hbl_stack)
})

# EBL metrics
ebl_metrics_summary = []
for name, model in ebl_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = None
    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
        # Normalize decision_function output to [0, 1] range for log_loss and brier_score
        from sklearn.preprocessing import minmax_scale
        y_proba = minmax_scale(y_proba)
    
    ebl_metrics_summary.append({
        "Model": f"EBL_{name}",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced_Acc": balanced_accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Cohen_Kappa": cohen_kappa_score(y_test, y_pred),
        "Log_Loss": log_loss(y_test, y_proba) if y_proba is not None else None,
        "Brier_Score": brier_score_loss(y_test, y_proba) if y_proba is not None else None
    })

# EBL stacking metrics
y_pred_ebl_stack = ebl_stacking_model.predict(X_test)
y_proba_ebl_stack = ebl_stacking_model.predict_proba(X_test)[:, 1]
ebl_metrics_summary.append({
    "Model": "EBL_Stacking",
    "Accuracy": accuracy_score(y_test, y_pred_ebl_stack),
    "Balanced_Acc": balanced_accuracy_score(y_test, y_pred_ebl_stack),
    "Precision": precision_score(y_test, y_pred_ebl_stack),
    "Recall": recall_score(y_test, y_pred_ebl_stack),
    "F1-Score": f1_score(y_test, y_pred_ebl_stack),
    "AUC": roc_auc_score(y_test, y_proba_ebl_stack),
    "MCC": matthews_corrcoef(y_test, y_pred_ebl_stack),
    "Cohen_Kappa": cohen_kappa_score(y_test, y_pred_ebl_stack),
    "Log_Loss": log_loss(y_test, y_proba_ebl_stack),
    "Brier_Score": brier_score_loss(y_test, y_proba_ebl_stack)
})

# Combined metrics
all_metrics = hbl_metrics_summary + ebl_metrics_summary
metrics_df = pd.DataFrame(all_metrics)
print("\n--- All Models Performance Metrics ---")
print(metrics_df.to_string(index=False))

# Metrics comparison visualization
plt.figure(figsize=(16, 8))
metrics_melted = metrics_df[metrics_df['Model'].str.contains('Stacking')].melt(
    id_vars='Model', var_name='Metric', value_name='Score')
ax = sns.barplot(data=metrics_melted, x='Metric', y='Score', hue='Model', palette='viridis')
plt.title('HBL vs EBL Stacking: Detailed Metrics Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Score')
plt.xticks(rotation=45, ha='right')
# Add value labels on bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', label_type='edge', padding=3, fontsize=8, fontweight='bold')
plt.legend(title='Stacking Model')
plt.tight_layout()
plt.show()

# ============================================================================
# ROC CURVES COMPARISON
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# HBL ROC curves
for name, model in hbl_models.items():
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        continue
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    axes[0].plot(fpr, tpr, label=f"{name.upper()} (AUC={auc:.2f})", alpha=0.7)

fpr_hbl, tpr_hbl, _ = roc_curve(y_test, y_proba_hbl_stack)
auc_hbl = roc_auc_score(y_test, y_proba_hbl_stack)
axes[0].plot(fpr_hbl, tpr_hbl, label=f"HBL STACKING (AUC={auc_hbl:.3f})", 
             linewidth=3, color='red')
axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray')
axes[0].set_title("HBL Stacking ROC Curves", fontsize=12, fontweight='bold')
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend(loc='lower right', fontsize=8)
axes[0].grid(alpha=0.3)

# EBL ROC curves
for name, model in ebl_models.items():
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        continue
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    axes[1].plot(fpr, tpr, label=f"{name.upper()} (AUC={auc:.2f})", alpha=0.7)

fpr_ebl, tpr_ebl, _ = roc_curve(y_test, y_proba_ebl_stack)
auc_ebl = roc_auc_score(y_test, y_proba_ebl_stack)
axes[1].plot(fpr_ebl, tpr_ebl, label=f"EBL STACKING (AUC={auc_ebl:.2f})", 
             linewidth=3, color='darkgreen')
axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray')
axes[1].set_title("EBL Stacking ROC Curves", fontsize=12, fontweight='bold')
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend(loc='lower right', fontsize=8)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# CONFUSION MATRICES COMPARISON
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# HBL Stacking confusion matrix
cm_hbl = confusion_matrix(y_test, y_pred_hbl_stack)
sns.heatmap(cm_hbl, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0])
axes[0].set_title('HBL Stacking Confusion Matrix', fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# EBL Stacking confusion matrix
cm_ebl = confusion_matrix(y_test, y_pred_ebl_stack)
sns.heatmap(cm_ebl, annot=True, fmt='d', cmap='Greens', cbar=False, ax=axes[1])
axes[1].set_title('EBL Stacking Confusion Matrix', fontweight='bold')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)