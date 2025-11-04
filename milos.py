import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score


# Load the pre-split data
X_train = pd.read_csv('competition/x_train.csv')
X_test = pd.read_csv('competition/x_test.csv')
y_train = pd.read_csv('competition/y_train.csv')
y_test = pd.read_csv('competition/y_test.csv')



# Visualize smoking-related patterns
def explore_smoking_patterns(df):
    """
    Explore which features differ most between smoking groups
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.ravel()
    
    # Key biomarkers known to be affected by smoking
    key_features = [
        'hemoglobin', 'HDL_chole', 'triglyceride', 'gamma_GTP',
        'SGOT_ALT', 'SGOT_AST', 'SBP', 'DBP',
        'BLDS', 'waistline', 'tot_chole', 'LDL_chole'
    ]
    
    for idx, feature in enumerate(key_features):
        if feature in df.columns:
            # Create violin plot for each smoking status
            data_to_plot = [
                df[df['SMK_stat_type_cd'] == 1][feature].dropna(),
                df[df['SMK_stat_type_cd'] == 2][feature].dropna(),
                df[df['SMK_stat_type_cd'] == 3][feature].dropna()
            ]
            
            axes[idx].violinplot(data_to_plot, positions=[1, 2, 3])
            axes[idx].set_xticks([1, 2, 3])
            axes[idx].set_xticklabels(['Never', 'Former', 'Current'])
            axes[idx].set_title(f'{feature}')
            axes[idx].set_xlabel('Smoking Status')
            axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Biomarker Distribution by Smoking Status', fontsize=16)
    plt.tight_layout()
    plt.show()


def engineer_smoking_features(X):
    """
    Create features specifically relevant to smoking detection
    """
    print("\n" + "="*60)
    print("FEATURE ENGINEERING FOR SMOKING DETECTION")
    print("="*60)
    
    X_eng = X.copy()
    
    # 1. Cardiovascular risk indicators (smoking affects cardiovascular system)
    X_eng['pulse_pressure'] = X_eng['SBP'] - X_eng['DBP']
    X_eng['MAP'] = X_eng['DBP'] + (X_eng['SBP'] - X_eng['DBP']) / 3
    
    # 2. Cholesterol ratios (smoking affects lipid metabolism)
    X_eng['total_hdl_ratio'] = X_eng['tot_chole'] / (X_eng['HDL_chole'] + 1e-5)
    X_eng['ldl_hdl_ratio'] = X_eng['LDL_chole'] / (X_eng['HDL_chole'] + 1e-5)
    X_eng['non_hdl_cholesterol'] = X_eng['tot_chole'] - X_eng['HDL_chole']
    
    # 3. BMI and metabolic indicators
    X_eng['BMI'] = X_eng['weight'] / ((X_eng['height'] / 100) ** 2)
    X_eng['waist_height_ratio'] = X_eng['waistline'] / X_eng['height']
    
    # 4. Liver function combinations (smoking affects liver)
    X_eng['ast_alt_ratio'] = X_eng['SGOT_AST'] / (X_eng['SGOT_ALT'] + 1e-5)
    X_eng['liver_enzyme_sum'] = X_eng['SGOT_AST'] + X_eng['SGOT_ALT'] + X_eng['gamma_GTP']
    
    # 5. Age-related risk factors
    X_eng['age_bmi_interaction'] = X_eng['age'] * X_eng['BMI']
    X_eng['age_bp_risk'] = X_eng['age'] * X_eng['SBP'] / 100
    
    # 6. Gender-specific indicators
    X_eng['male_waist_risk'] = X_eng['sex_encoded'] * X_eng['waistline']
    X_eng['male_hdl_risk'] = X_eng['sex_encoded'] * (50 - X_eng['HDL_chole'])
    
    # 7. Composite health scores
    # Metabolic syndrome components (often correlated with smoking)
    X_eng['metabolic_risk_score'] = (
        (X_eng['waistline'] > 90).astype(int) +  # High waist
        (X_eng['SBP'] > 130).astype(int) +        # High BP
        (X_eng['BLDS'] > 100).astype(int) +       # High glucose
        (X_eng['triglyceride'] > 150).astype(int) + # High triglycerides
        (X_eng['HDL_chole'] < 40).astype(int)     # Low HDL
    )
    
    # 8. Smoking-specific biomarker flags
    X_eng['high_hemoglobin'] = (X_eng['hemoglobin'] > 16).astype(int)
    X_eng['low_hdl'] = (X_eng['HDL_chole'] < 40).astype(int)
    X_eng['high_triglyceride'] = (X_eng['triglyceride'] > 150).astype(int)
    
    print(f"Original features: {X.shape[1]}")
    print(f"Engineered features: {X_eng.shape[1]}")
    print(f"New features added: {X_eng.shape[1] - X.shape[1]}")
    
    return X_eng

# Feature selection specific to smoking
def select_smoking_features(X, y, n_features=25):
    """
    Select most relevant features for smoking prediction
    """
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    from sklearn.ensemble import RandomForestClassifier
    
    print("\n" + "="*60)
    print("FEATURE SELECTION FOR SMOKING PREDICTION")
    print("="*60)
    
    # Method 1: ANOVA F-statistic
    selector_f = SelectKBest(score_func=f_classif, k=n_features)
    selector_f.fit(X, y)
    
    # Method 2: Mutual Information
    selector_mi = SelectKBest(score_func=mutual_info_classif, k=n_features)
    selector_mi.fit(X, y)
    
    # Method 3: Random Forest importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Combine scores
    feature_scores = pd.DataFrame({
        'feature': X.columns,
        'f_score': selector_f.scores_,
        'mi_score': selector_mi.scores_,
        'rf_importance': rf.feature_importances_
    })
    
    # Normalize scores
    for col in ['f_score', 'mi_score', 'rf_importance']:
        feature_scores[col] = feature_scores[col] / feature_scores[col].max()
    
    # Combined score
    feature_scores['combined_score'] = (
        feature_scores['f_score'] + 
        feature_scores['mi_score'] + 
        feature_scores['rf_importance']
    ) / 3
    
    # Sort by combined score
    feature_scores = feature_scores.sort_values('combined_score', ascending=False)
    
    print("\nTop 15 Features for Smoking Prediction:")
    print(feature_scores[['feature', 'combined_score']].head(15))
    
    # Select top features
    top_features = feature_scores['feature'].head(n_features).tolist()
    
    return X[top_features], top_features, feature_scores



def train_smoking_models(x_train, x_test, y_train, y_test):
    """
    Train and optimize models specifically for smoking prediction
    """
    print("\n" + "="*60)
    print("MODEL TRAINING FOR SMOKING PREDICTION")
    print("="*60)
    
    # Custom scorer prioritizing current smoker detection
    def smoking_f1_score(y_true, y_pred):
        # Weight current smokers (class 3) higher
        return f1_score(y_true, y_pred, average='weighted', labels=[1,2,3],
                       sample_weight=(y_true == 3).astype(int) * 2 + 1)
    
    scorer = make_scorer(smoking_f1_score)
    
    models = {}
    results = {}
    
    # 1. LOGISTIC REGRESSION (Multinomial)
    print("\n[1/5] Training Logistic Regression...")
    lr_params = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'multi_class': ['multinomial'],
        'solver': ['lbfgs'],
        'max_iter': [1000]
    }
    
    lr = GridSearchCV(
        LogisticRegression(random_state=42),
        lr_params,
        cv=5,
        scoring=scorer,
        n_jobs=-1
    )
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr.best_estimator_
    
    lr_justification = """
    JUSTIFICATION - Logistic Regression for Smoking:
    ✓ Multinomial classification handles 3 smoking categories naturally
    ✓ Provides probability scores for risk stratification
    ✓ Coefficients show which biomarkers increase smoking likelihood
    ✓ Fast inference for point-of-care screening
    ✓ Baseline for comparing complex models
    """
    
    # 2. RANDOM FOREST
    print("\n[2/5] Training Random Forest...")
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced', None]  # Handle class imbalance
    }
    
    rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_params,
        cv=5,
        scoring=scorer,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf.best_estimator_
    
    rf_justification = """
    JUSTIFICATION - Random Forest for Smoking:
    ✓ Captures non-linear relationships between biomarkers
    ✓ 'balanced' class weight handles smoking prevalence imbalance
    ✓ Feature importance identifies key smoking biomarkers
    ✓ Robust to outliers in medical measurements
    ✓ No preprocessing needed for raw lab values
    """
    
    # 3. DECISION TREE
    print("\n[3/5] Training Decision Tree...")
    dt_params = {
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy'],
        'class_weight': ['balanced', None]
    }
    
    dt = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=5, scoring=scorer, n_jobs=-1)
    dt.fit(X_train, y_train)
    models['Decision Tree'] = dt.best_estimator_
    print("✓ Decision Tree trained")
    
    
    # 4. SVM
    print("\n[4/5] Training SVM...")
    svm_params = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'poly'],
        'class_weight': ['balanced']  # Important for imbalanced data
    }
    
    svm = GridSearchCV(
        SVC(random_state=42, probability=True),
        svm_params,
        cv=5,
        scoring=scorer,
        n_jobs=-1
    )
    svm.fit(X_train, y_train)
    models['SVM'] = svm.best_estimator_
    
    svm_justification = """
    JUSTIFICATION - SVM for Smoking:
    ✓ RBF kernel captures complex biomarker interactions
    ✓ 'balanced' class weight crucial for detecting current smokers
    ✓ Effective with standardized medical measurements
    ✓ Good generalization with limited samples
    ✓ Probability estimates for risk assessment
    """
    
    # 5. NEURAL NETWORK
    print("\n[5/5] Training Neural Network...")
    nn_params = {
        'hidden_layer_sizes': [(100,50), (100,50,25)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.001, 0.01],  # Regularization
        'learning_rate': ['adaptive'],
        'max_iter': [500]
    }
    
    nn = GridSearchCV(
        MLPClassifier(random_state=42, early_stopping=True),
        nn_params,
        cv=5,
        scoring=scorer,
        n_jobs=-1
    )
    nn.fit(X_train, y_train)
    models['Neural Network'] = nn.best_estimator_
    
    nn_justification = """
    JUSTIFICATION - Neural Network for Smoking:
    ✓ Learns complex biomarker patterns automatically
    ✓ Multiple layers capture hierarchical health indicators
    ✓ Handles high-dimensional engineered features
    ✓ Non-linear activation models biochemical interactions
    ✓ Regularization prevents overfitting to training smokers
    """
    
    # Print justifications
    justifications = {
        'Logistic Regression': lr_justification,
        'Random Forest': rf_justification,
        'Gradient Boosting': gb_justification,
        'SVM': svm_justification,
        'Neural Network': nn_justification
    }
    
    for name, just in justifications.items():
        print(f"\n{just}")
    
    return models

















def evaluate_smoking_models(models, X_test, y_test, X_train, y_train):
    """
    Evaluate models with focus on smoking detection performance
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION FOR SMOKING DETECTION")
    print("="*60)
    
    results_summary = []
    
    for name, model in models.items():
        print(f"\n{'-'*40}")
        print(f"{name}")
        print(f"{'-'*40}")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Overall metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, labels=[1, 2, 3], average=None
        )
        
        # Weighted metrics
        weighted_precision = np.average(precision, weights=support)
        weighted_recall = np.average(recall, weights=support)
        weighted_f1 = np.average(f1, weights=support)
        
        # Specific focus on current smoker detection (class 3)
        current_smoker_recall = recall[2]  # Index 2 for class 3
        current_smoker_precision = precision[2]
        current_smoker_f1 = f1[2]
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
        
        # Store results
        results_summary.append({
            'Model': name,
            'Accuracy': accuracy,
            'Weighted F1': weighted_f1,
            'Current Smoker Recall': current_smoker_recall,
            'Current Smoker Precision': current_smoker_precision,
            'Current Smoker F1': current_smoker_f1,
            'CV Mean': cv_scores.mean(),
            'CV Std': cv_scores.std()
        })
        
        # Print detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Never Smoked', 'Former Smoker', 'Current Smoker']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate sensitivity and specificity for current smokers
        tn = cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1]  # True negatives for class 3
        fp = cm[0,2] + cm[1,2]  # False positives for class 3
        fn = cm[2,0] + cm[2,1]  # False negatives for class 3
        tp = cm[2,2]  # True positives for class 3
        
        sensitivity_current = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity_current = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"\nCurrent Smoker Detection Performance:")
        print(f"  Sensitivity (Recall): {sensitivity_current:.3f}")
        print(f"  Specificity: {specificity_current:.3f}")
        print(f"  Precision: {current_smoker_precision:.3f}")
        print(f"  F1-Score: {current_smoker_f1:.3f}")
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_features_idx = np.argsort(importances)[-10:]
            print(f"\nTop 10 Important Features (indices): {top_features_idx}")
    
    # Create summary dataframe
    results_df = pd.DataFrame(results_summary)
    results_df = results_df.sort_values('Current Smoker F1', ascending=False)
    
    print("\n" + "="*60)
    print("SUMMARY: MODEL PERFORMANCE COMPARISON")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Best model for current smoker detection
    best_model = results_df.iloc[0]['Model']
    print(f"\n✓ BEST MODEL FOR SMOKING DETECTION: {best_model}")
    print(f"  Current Smoker F1: {results_df.iloc[0]['Current Smoker F1']:.3f}")
    print(f"  Current Smoker Recall: {results_df.iloc[0]['Current Smoker Recall']:.3f}")
    
    return results_df





def visualize_smoking_results(models, X_test, y_test, feature_names):
    """
    Create comprehensive visualizations for clinical interpretation
    """
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Confusion Matrices
    for idx, (name, model) in enumerate(models.items(), 1):
        ax = plt.subplot(3, 5, idx)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Never', 'Former', 'Current'],
                   yticklabels=['Never', 'Former', 'Current'])
        ax.set_title(f'{name}')
        ax.set_ylabel('True' if idx == 1 else '')
        ax.set_xlabel('Predicted')
    
    # 2. Feature Importance (for tree-based models)
    importance_models = ['Random Forest', 'Gradient Boosting']
    for idx, model_name in enumerate(importance_models, 6):
        if model_name in models:
            ax = plt.subplot(3, 5, idx)
            model = models[model_name]
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                top_idx = np.argsort(importances)[-15:]
                top_features = [feature_names[i] for i in top_idx]
                top_importances = importances[top_idx]
                
                ax.barh(range(len(top_features)), top_importances)
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features, fontsize=8)
                ax.set_xlabel('Importance')
                ax.set_title(f'{model_name} - Top Features')
    
    # 3. ROC Curves for Current Smoker Detection (One vs Rest)
    from sklearn.metrics import roc_curve, auc
    
    ax = plt.subplot(3, 5, 11)
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
            # ROC for current smokers (class 3)
            fpr, tpr, _ = roc_curve((y_test == 3).astype(int), y_prob[:, 2])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Current Smoker Detection')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    plt.suptitle('Smoking Prediction Model Analysis', fontsize=16)
    plt.tight_layout()
    plt.show()








def generate_clinical_report(models, results_df, feature_scores):
    """
    Generate a clinical interpretation report
    """
    print("\n" + "="*60)
    print("CLINICAL INSIGHTS REPORT")
    print("="*60)
    
    print("\n1. KEY BIOMARKERS FOR SMOKING DETECTION:")
    print("-" * 40)
    top_biomarkers = feature_scores.head(10)
    for idx, row in top_biomarkers.iterrows():
        feature = row['feature']
        score = row['combined_score']
        
        # Provide clinical interpretation
        interpretations = {
            'gamma_GTP': 'Elevated in smokers due to oxidative stress',
            'hemoglobin': 'Often elevated in smokers (compensatory mechanism)',
            'HDL_chole': 'Typically decreased in smokers',
            'triglyceride': 'Often elevated due to smoking-induced metabolic changes',
            'waistline': 'Central adiposity associated with smoking',
            'SGOT_ALT': 'Liver enzyme affected by smoking',
            'age': 'Smoking patterns vary with age',
            'sex_encoded': 'Gender differences in smoking prevalence'
        }
        
        interpretation = interpretations.get(feature, 'Associated with smoking status')
        print(f"  • {feature}: Score={score:.3f} - {interpretation}")
    
    print("\n2. MODEL RECOMMENDATIONS FOR CLINICAL USE:")
    print("-" * 40)
    
    best_model = results_df.iloc[0]
    print(f"  Recommended Model: {best_model['Model']}")
    print(f"  - Current Smoker Detection Rate: {best_model['Current Smoker Recall']:.1%}")
    print(f"  - False Positive Rate: {1 - best_model['Current Smoker Precision']:.1%}")
    print(f"  - Overall Accuracy: {best_model['Accuracy']:.1%}")
    
    print("\n3. CLINICAL APPLICATION GUIDELINES:")
    print("-" * 40)
    print("  • Use as screening tool, not definitive diagnosis")
    print("  • High probability (>0.7) → Counsel on smoking cessation")
    print("  • Medium probability (0.3-0.7) → Further assessment needed")
    print("  • Consider patient history and clinical context")
    print("  • Regular model retraining with new data recommended")
    
    print("\n4. LIMITATIONS:")
    print("-" * 40)
    print("  • Model accuracy limited by self-reported training data")
    print("  • Cannot distinguish occasional from heavy smokers")
    print("  • Former smoker detection less reliable than current")
    print("  • Biomarkers influenced by other health conditions")


def complete_smoking_prediction_pipeline(df):
    """
    Execute the complete pipeline for smoking prediction
    """
    print("="*70)
    print("SMOKING STATUS PREDICTION FROM HEALTH BIOMARKERS")
    print("="*70)
    
    # 1. Data Preparation
    X, y = prepare_smoking_data(df)
    
    # 2. Explore patterns
    explore_smoking_patterns(df)
    
    # 3. Feature Engineering
    X_engineered = engineer_smoking_features(X)
    
    # 4. Feature Selection
    X_selected, selected_features, feature_scores = select_smoking_features(
        X_engineered, y, n_features=25
    )
    
    # 5. Handle class imbalance with SMOTE
    print("\n" + "="*60)
    print("HANDLING CLASS IMBALANCE")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply SMOTE to training data
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"Original training distribution: {np.bincount(y_train)}")
    print(f"Balanced training distribution: {np.bincount(y_train_balanced)}")
    
    # 6. Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    # 7. Train models
    models = train_smoking_models(
        X_train_scaled, X_test_scaled, 
        y_train_balanced, y_test
    )
    
    # 8. Evaluate models
    results_df = evaluate_smoking_models(
        models, X_test_scaled, y_test,
        X_train_scaled, y_train_balanced
    )
    
    # 9. Visualize results
    visualize_smoking_results(
        models, X_test_scaled, y_test, selected_features
    )
    
    # 10. Generate clinical report
    generate_clinical_report(models, results_df, feature_scores)
    
    return models, results_df, selected_features

# Run the complete pipeline
# models, results, features = complete_smoking_prediction_pipeline(your_df)