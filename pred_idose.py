import pandas as pd 
import numpy as np 
import xgboost as xgb 
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from jinja2 import Environment, FileSystemLoader
import pdfkit
import os
import webbrowser
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
import seaborn as sns
from scipy.stats import pearsonr 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler 
import umap 
import seaborn as sns
from code_groupings import new_feats
import plotnine as p9
import hdbscan 
from scipy.stats import f_oneway
import shap 
import joblib
from datetime import datetime
from sklearn.inspection import partial_dependence, permutation_importance
from xgboost import plot_tree
from sklearn.pipeline import make_pipeline

LESS_RELATED = ['CORNEAL_STUFF', 'REFRACTIVE_DISORDERS', 'GENERAL_OP', 'OCULAR_PROSTHETICS', 'RETINAL_PROCEDURES', 'GENERAL_PROCEDURES']
DIAGNOSES = ['PREGLAUCOMA', 'OPEN_ANGLE_BORDERLINE', 'ANGLE_CLOSURE', 'OC_HYPERTENSION', 'PRIMARY_OPEN_ANGLE_MILD', 'PRIMARY_OPEN_ANGLE_MOD',
             'PRIMARY_OPEN_ANGLE_SEV', 'PRIMARY_OPEN_ANGLE_UNSP']
COLUMNS_TO_REMOVE = ['Ml All Patients', '0660T Patients', 'J7355 Patients', 'J7351 Patients', 
                     '0661T Patients', 'Medscout Profile', 'Medscout Profile_x', 'Medscout Profile_y',
                     'Idose (0660T, J7355) Patients In 2022', 'Idose (0660T, J7355) Patients In 2023',
                     'Idose (0660T, J7355) Patients In 2024', 'Idose (0660T, J7355) Patients In 2025',
                     'Ml All Patients In 2022', 'Ml All Patients In 2023', 'Ml All Patients In 2024', 'Ml All Patients In 2025',
                     'J7355 Patients In 2022', 'J7355 Patients In 2023', 'J7355 Patients In 2024', 'J7355 Patients In 2025',
                     'J7351 Patients In 2022','J7351 Patients In 2023','J7351 Patients In 2024','J7351 Patients In 2025',
                     '0661T Patients In 2022', '0661T Patients In 2023', '0661T Patients In 2024','0661T Patients In 2025',
                     '0660T Patients In 2022', '0660T Patients In 2023', '0660T Patients In 2024','0660T Patients In 2025',
                     'index']
IDOS_VAL_COLUMN = 'Idose (0660T, J7355)'
XGB_TEST_PARAMS = {
    'n_estimators': [100, 300, 400],
    'max_depth': [2, 3, 6, 9],
    'learning_rate': [0.01, 0.1], 
    'subsample': [0.7, 0.8, 0.9], 
    'gamma': [0]
}
XGB_PARAMS = {
    'n_estimators': 300, 
    'max_depth': 6, 
    'learning_rate': 0.05, 
    'subsample': 0.8
}

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='Data file containing medscope information')
    parser.add_argument('--pred_idos_val', action='store_true', help='If the model should predict the number of iDose patients')
    parser.add_argument('--pred_idos_bool', action='store_true', help='If the model should predict if iDose is/should be used')
    parser.add_argument('--unsupervised_clusters', action='store_true', help='If unsupervised clustering should be run')
    parser.add_argument('--grid_search', action='store_true', help='If a grid search of parameters should be run')
    parser.add_argument('--data_consolidation_level', type=int, default=1, 
                        help="""How much the data should be consolidated: 
                        0 -> no consolidation, 
                        1 -> basic consolidation based on similarity, 
                        2 -> similarity grouping + removing seemingly unrelated categories, 
                        3 -> similarity grouping + removing unrelated + removing diagnoses,
                        4 -> similarity grouping + removing unrelated + removing diagnoses + removing imaging,
                        5 -> similarity grouping, with custom columns kept specified with --custom_feats
                        """)
    parser.add_argument('--custom_feats', help=f'File containing list of custom features wanted, options are: {new_feats.keys()}')
    parser.add_argument('--extra_idose', nargs='+', help=f'iDose data to add to the original data')
    parser.add_argument('--extra_non', nargs='+', help=f'Non-iDose data to add to the original data')
    parser.add_argument('--time_features', nargs=2, help=f'Start year and end year to be considered over')
    parser.add_argument('--predict', help='File containing physicians that you want to predict on')
    parser.add_argument('--classifier', help='Saved model file to use when predicting')
    parser.add_argument('--save_model', action='store_true', help='If the model should be saved for future prediction')
    parser.add_argument('--totals', action='store_true', help='Whether to use the raw total values instead of proportions for features', default=False)
    parser.add_argument('--props', action='store_true', help='Whether to use the proportions for feature values', default=True)
    
    args = parser.parse_args()
    
    if args.predict is not None and args.classifier is not None: 
        clf = joblib.load(args.classifier)
    else: 
        if args.data is None: 
            print('--data must be specified if running --predict with no classifier')
            exit(1)
            
        try: 
            if '.csv' == args.data[-4:]: 
                data = pd.read_csv(args.data)
                for drop in COLUMNS_TO_REMOVE: 
                    if drop in data.columns:
                        data = data.drop(drop, axis=1)
                
                if args.extra_idose is not None:        
                    for extra in args.extra_idose: 
                        ex_df = pd.read_csv(extra, index_col=0)
                        if 'NPI / CCN' in ex_df.columns: 
                            ex_df = ex_df.rename(columns={'NPI / CCN':'NPI'})
                        for drop in COLUMNS_TO_REMOVE: 
                            if drop in ex_df.columns:
                                ex_df = ex_df.drop(drop, axis=1)
                                
                        if IDOS_VAL_COLUMN + ' Patients' not in ex_df.columns: 
                            ex_df[IDOS_VAL_COLUMN + ' Patients'] = 1
                        
                        diff = [val for val in data.columns if val not in ex_df.columns]
                        diff2 = [val for val in ex_df.columns if val not in data.columns]
                        
                        if len(diff) > 0 or len(diff2) > 0:
                            raise Exception(f'Data columns must match: {diff, diff2}')
                        
                        data = pd.concat([data, ex_df], axis=0).reset_index(drop=True)
            elif '.xlsx' == args.data[-5:]:
                data = pd.read_excel(args.data, index_col=0)
        except Exception as e: 
            print('Data file could not be read in properly')
            print(f'Error: {e}')
            exit(1)
            
        X, y = prep_data(data, args.data_consolidation_level, 
                        args.time_features is not None, 
                        args.time_features[0] if args.time_features else None,
                        args.time_features[1] if args.time_features else None,
                        args.custom_feats, 
                        args.props, args.totals)
        
        print(f'iDose Features: {sum(y > 0)}')
        print(f'Non iDose Features: {sum(y == 0)}')
        
        if args.pred_idos_val: 
            clf = pred_idos_val(X, y, args.grid_search, args.data_consolidation_level)
        elif args.pred_idos_bool: 
            clf = pred_idos_bool(X, y, args.grid_search, args.data_consolidation_level)
        elif args.unsupervised_clusters: 
            run_unsupervised(X, y)
        else:
            print('Please select a model to use with --pred_idos_val, --pred_idos_bool, or --unsupervised_clusters')
            exit(1)


    if args.predict is not None: 
        pred_df = pd.read_csv(args.predict)
        npi_dict = dict(zip(pred_df['NPI'], pred_df['Name']))
        for drop in COLUMNS_TO_REMOVE: 
            if drop in pred_df.columns:
                pred_df = pred_df.drop(drop, axis=1)
                
        X_pred, _ = prep_data(pred_df, args.data_consolidation_level, 
                    args.time_features is not None, 
                    args.time_features[0] if args.time_features else None,
                    args.time_features[1] if args.time_features else None,
                    args.custom_feats,
                    args.props, args.totals)
        
        run_prediction(X_pred, clf, npi_dict)    

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.save_model: 
        joblib.dump(clf, f'xgb_model_cons{args.data_consolidation_level}_{timestamp}.pkl')
        

def combine_cols(df, combine_dict, start_year=None, end_year=None): 
    total_processed = 0
    which_processed = []
    
    new_cols = {}
    
    for header, cols in combine_dict.items(): 
        total_processed += len(cols)
        which_processed += cols 
        new_cols[header] = (df[cols].sum(axis=1))
        
        if start_year is not None and end_year is not None: 
            for year in range(int(start_year), int(end_year)+1):
                year_cols = [f'{col} In {year}' for col in cols]
                new_cols[f'{header} In {year}'] = df[year_cols].sum(axis=1)
                total_processed += len(year_cols)
                which_processed += year_cols
    
    new_df = pd.DataFrame(new_cols)
    
    new_df[IDOS_VAL_COLUMN] = df[IDOS_VAL_COLUMN]
    return new_df, total_processed, which_processed


def calculate_time_features(X, start_year, end_year): 
    features = X.columns.str.replace(r' In 20\d{2}','', regex=True).drop_duplicates()
    time_feats = {}
    
    for feature in features: 
        vals_for_feature = np.column_stack([X[f'{feature} In {year}'] for year in range(int(start_year), int(end_year)+1)])
        time_feats[f'{feature} Rate of Change'] = (X[f'{feature} In {end_year}'] - X[f'{feature} In {start_year}'])/(end_year-start_year)
        time_feats[f'{feature} Median'] = np.median(vals_for_feature, axis=1)
        time_feats[f'{feature} Standard Deviation'] = np.std(vals_for_feature, axis=1)
        time_feats[f'{feature} Range'] = np.max(vals_for_feature, axis=1) - np.min(vals_for_feature, axis=1)
        
    time_features = pd.DataFrame(time_feats)   
        
    return time_features


def prep_data(data, data_consolidation_level, time_features=False, start_year=None, end_year=None, custom_feats=None, props=True, totals=False):     
    df = data.set_index('NPI').drop('Name', axis=1).replace('<11', 5).astype(int)
    
    if time_features:
        time_df = pd.DataFrame()
        start_year = int(start_year)
        end_year = int(end_year)
        time_df = df.filter(like=' Patients In', axis=1)
        time_df.columns = time_df.columns.str.replace(" Patients In","", regex=False)
    else:
        df = df.loc[:, ~df.columns.str.contains(' Patients In')]
        
    df.columns = df.columns.str.replace(" Patients","", regex=False)
    
    new_df = pd.DataFrame()
    
    if data_consolidation_level == 0:
        new_df = df
        total_processed = 0
    elif data_consolidation_level == 1:
        new_df, total_processed, which_processed = combine_cols(df, new_feats, start_year, end_year)
    elif data_consolidation_level == 2: 
        new_df, total_processed, which_processed = combine_cols(df, new_feats, start_year, end_year)
        new_df = new_df.drop(LESS_RELATED, axis=1)
        if time_features:
            for year in range(start_year, end_year+1):
                year_cols = [f'{val} In {year}' for val in LESS_RELATED]
                new_df = new_df.drop(year_cols, axis=1)
    elif data_consolidation_level == 3: 
        new_df, total_processed, which_processed = combine_cols(df, new_feats, start_year, end_year)
        new_df = new_df.drop(LESS_RELATED, axis=1)
        new_df = new_df.drop(DIAGNOSES, axis=1)
        if time_features:
            for year in range(start_year, end_year+1):
                year_cols = [f'{val} In {year}' for val in LESS_RELATED]
                year_cols2 = [f'{val} In {year}' for val in DIAGNOSES]
                new_df = new_df.drop(year_cols, axis=1)
                new_df = new_df.drop(year_cols2, axis=1)
    elif data_consolidation_level == 4: 
        new_df, total_processed, which_processed = combine_cols(df, new_feats, start_year, end_year)
        new_df = new_df.drop(LESS_RELATED, axis=1)
        new_df = new_df.drop(DIAGNOSES, axis=1)
        new_df = new_df.drop('DIAGNOSTIC_IMAGING', axis=1)
        if time_features:
            for year in range(start_year, end_year+1):
                year_cols = [f'{val} In {year}' for val in LESS_RELATED]
                year_cols2 = [f'{val} In {year}' for val in DIAGNOSES]
                new_df = new_df.drop(year_cols, axis=1)
                new_df = new_df.drop(year_cols2, axis=1)
                new_df = new_df.drop(f'DIAGNOSTIC_IMAGING In {year}', axis=1)
    elif data_consolidation_level == 5: 
        new_df, total_processed, which_processed = combine_cols(df, new_feats, start_year, end_year)
        if custom_feats != None: 
            feature_set = [IDOS_VAL_COLUMN]
            with open(custom_feats, 'r') as infile: 
                for line in infile:
                    line = line.strip()
                    if line not in new_feats.keys(): 
                        print(f'Invalid Feature: {line}')
                        print(f'Available Features: {new_feats.keys()}')
                    else: 
                        feature_set.append(line) 
            new_df = new_df[feature_set]
        else: 
            print('You must supply custom features with --custom_feats when using consolidation level 5')
            exit(1)
    else: 
        print('Please use a consolidation level between 0-5')
        exit(1)
        
    total_feat_before = len(df.drop(IDOS_VAL_COLUMN, axis=1).columns.tolist())
    if total_processed != total_feat_before and data_consolidation_level > 0:
        print(f'Processed: {total_processed}, Total Before: {total_feat_before}')
        missing_cols = []
        for col in df.drop(IDOS_VAL_COLUMN, axis=1).columns.tolist(): 
            if col not in which_processed: 
                missing_cols.append(col)
        print(f'Missing Columns: {missing_cols}')
        
            
    std_df = new_df.loc[:, ~new_df.columns.str.contains(' 20')]
    
    # Making the features a proportion of all the features still included 
    if props: 
        X = std_df.drop(IDOS_VAL_COLUMN, axis=1).astype(float)
        prop_df = X.div(X.sum(axis=1), axis=0)
        prop_df.columns = [f'{col} Proportion' for col in prop_df.columns]
        if not totals:
            X = prop_df
    if totals: 
        X.columns = [f'{col} Total' for col in X.columns]
    if props and totals: 
        X = X.join(prop_df)
    
    
    if time_features:
        time_df = new_df.loc[:, new_df.columns.str.contains('In 20')]
        X = pd.concat([X, calculate_time_features(time_df, start_year, end_year)], axis=1)
        
    y = new_df[IDOS_VAL_COLUMN]
    return X, y
 

def run_prediction(X_pred, clf, npi_dict): 
    preds = clf.predict(X_pred)
    probs = clf.predict_proba(X_pred)
    pred_class_probs = probs.max(axis=1)
    
    X_pred['Prediction'] = preds 
    X_pred['Probability'] = pred_class_probs
    
    df = X_pred.sort_values(by=['Prediction','Probability'], ascending=[False, False])
    df['NPI'] = df.index
    df['Name'] = df['NPI'].map(npi_dict)
    
    df = df[['Name', 'NPI', 'Prediction', 'Probability']]
    df.to_csv('predictions.csv', index=0)
    
    
        
def pred_idos_val(X, y, grid_search, consolidation_level): 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
    
    #clf = xgb.XGBRegressor(objective = 'reg:squarederror', use_label_encoder=False)
    
    if grid_search: 
        clf = xgb.XGBRegressor(objective = 'reg:squarederror')
        grid = GridSearchCV(clf, param_grid=XGB_TEST_PARAMS, scoring='accuracy', cv=5)
        grid.fit(X_train, y_train)
        
        print(grid.best_params_)
        
        clf = grid.best_estimator_
    else: 
        clf = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=XGB_PARAMS['n_estimators'], subsample=XGB_PARAMS['subsample'],
                                max_depth=XGB_PARAMS['max_depth'], learning_rate=XGB_PARAMS['learning_rate'])
        clf.fit(X_train, y_train)
    
    co_path = f"{os.getcwd()}\\correlation_plot.png"
    plot_correlation(clf, X_test, y_test, co_path)
    generate_model_report(clf, X_test, y_test, X, y, co_path, 'Regression', output_path=f'xgb_report_consol{consolidation_level}.pdf', top_n_features=20)
    
    clf.fit(X, y)
    return clf


def pred_idos_bool(X, y, grid_search, consolidation_level): 
    y = y > 0
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
    
    if grid_search: 
        clf = xgb.XGBClassifier(objective = 'binary:logistic')
        grid = GridSearchCV(clf, param_grid=XGB_TEST_PARAMS, scoring='accuracy', cv=5)
        grid.fit(X_train, y_train)
        
        print(grid.best_params_)
        
        clf = grid.best_estimator_
    else: 
        clf = xgb.XGBClassifier(objective='binary:logistic', n_estimators=XGB_PARAMS['n_estimators'], subsample=XGB_PARAMS['subsample'],
                                max_depth=XGB_PARAMS['max_depth'], learning_rate=XGB_PARAMS['learning_rate'])
        clf.fit(X_train, y_train)
    
    # y_pred = clf.predict(X_test)
    # acc = clf.score(X_test, y_test)
    #print(acc)
    #get_importances(clf, 10)
    cm_path = f"{os.getcwd()}\\confusion_matrix.png"
    plot_confusion_matrix(clf, X_test, y_test, cm_path) 
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")
    generate_model_report(clf, X_test, y_test, X, y, cm_path, 'Binary', output_path=f'xgb_report_consol{consolidation_level}_{formatted_datetime}.pdf', top_n_features=20)
    
    clf.fit(X, y)
    return clf


def plot_importance(clf, importance_type, max_num_features): 
    importances = clf.get_booster().get_score(importance_type=importance_type)
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:max_num_features]
    features, scores = zip(*sorted_importances)
    if max_num_features > len(features):
        max_num_features = len(features)
    
    plt.figure(figsize=(10, 0.4*max_num_features))
    bars = plt.barh(range(len(scores)), scores, color='skyblue')
    
    plt.yticks(range(len(features)), features)
    plt.gca().invert_yaxis()
    
    for i, score in enumerate(scores):
        plt.text(score + max(scores) * 0.01, i, f'{score:.2f}', va='center')
        
    plt.xlabel(f'Importance ({importance_type})', fontsize=12)
    plt.title(f'Top {max_num_features} Features', fontsize=14)
    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.5)


def get_importances(clf, max_num_features):
    importances = dict(sorted(clf.get_booster().get_score(importance_type='gain').items(), key=lambda item: item[1], reverse=True))
    #print(list(importances.items()))
    plot_importance(clf, importance_type='gain', max_num_features=max_num_features)
    plt.savefig('importances.png') 
    plt.close()
    
    contributions = np.array(list(importances.values()))/sum(importances.values())*100
    feature_df = pd.DataFrame({
        'Feature': importances.keys(), 
        'Importance': importances.values(), 
        'Contribution': contributions
    }).sort_values(by="Importance", ascending=False)   
    
    return feature_df
   

def plot_confusion_matrix(clf, X_val, y_val, path):
    #print(clf)
    y_pred = clf.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    #ConfusionMatrixDisplay.from_estimator(clf, X_val, y_val, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    
    
def plot_correlation(clf, X_val, y_val, path): 
    y_pred = clf.predict(X_val)
    r, _ = pearsonr(y_val, y_pred)
    
    sns.set(style='whitegrid')
    
    plt.figure(figsize=(8,6))
    sns.regplot(x=y_val, y=y_pred, ci=None, scatter_kws={"s": 30, "alpha": 0.7})
    
    min_val = min(np.min(y_val), np.min(y_pred))
    max_val = min(np.max(y_val), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Ideal Fit")
    
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("True vs Predictions")
    plt.text(0.05, 0.95, f'r = {r:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
 
 
def generate_model_report(clf, X_val, y_val, X_full, y_full, image_path, clf_type, output_path='xgb_report.pdf', top_n_features=10): 
    #### GENERATE METRIC TABLE ####
    
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1] if hasattr(clf, "predict_proba") else None
    
    class_counts = pd.Series(y_full).value_counts().to_dict()
    class_labels = {1:'iDose Users', 0:'Non iDose Users'}
    class_summary = ", ".join([f'{class_labels.get(cl, f"Class {cl}")} ({count})' for cl, count in class_counts.items()])
    
    if clf_type == 'Binary':
        metrics = {
            "Accuracy": round(accuracy_score(y_val, y_pred), 4),
            "Precision": round(precision_score(y_val, y_pred), 4),
            "Recall": round(recall_score(y_val, y_pred), 4),
            "F1 Score": round(f1_score(y_val, y_pred), 4),
            "ROC AUC": round(roc_auc_score(y_val, y_proba), 4) if y_proba is not None else "N/A"
        }  
    elif clf_type == 'Regression':
        metrics = {
            "MAE": round(mean_absolute_error(y_val, y_pred), 4),
            "MSE": round(mean_squared_error(y_val, y_pred), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_val, y_pred)), 4),
            "R2 Score": round(r2_score(y_val, y_pred), 4)
        }
    
    
    #### FEATURE IMPORTANCE ####
    
    clf.fit(X_full, y_full)
    feature_df = get_importances(clf, top_n_features)
    if feature_df is not None: 
        fi_path = f'{os.getcwd()}\\importances.png'
       
       
    #### SHAP ANALYSIS ####
    
    explainer = shap.TreeExplainer(clf, X_full, model_output='probability')
    shap_values = explainer(X_full)
    shap_matrix = shap_values.values
    
    shap.summary_plot(shap_values, X_full, max_display=X_full.shape[1], show=False)
    shap_summary_filename = f'{os.getcwd()}\\shap_summary.png'
    plt.savefig(shap_summary_filename, bbox_inches='tight')
    plt.close()
        
    shap_df = pd.DataFrame(shap_values.values, columns=X_full.columns)
    mean_shap = shap_df.abs().mean().sort_values(ascending=False)
    shap_importance = pd.Series(mean_shap, index=X_full.columns).sort_values(ascending=False)
    xgb_importance = pd.Series(clf.feature_importances_, index=X_full.columns).sort_values(ascending=False)
    importance_df = pd.DataFrame({
        'Feature': xgb_importance.index.sort_values(),
        'SHAP': shap_importance,
        'XGB': xgb_importance
    })
    
    feature_df = pd.merge(feature_df, importance_df.drop('XGB',axis=1), on='Feature').sort_values(by='SHAP', ascending=False)
    
    top_features = mean_shap.head(3).index.tolist()
    top_example_tables = {} 
    top_example_means = {}
    top_example_maxes = {}
    top_example_mins = {}
    
    for feature in top_features: 
        i = X_full.columns.get_loc(feature)
        shap_vals = shap_matrix[:, i]
        top_indices = np.argsort(np.abs(shap_vals))[::-1][:10]
        top_example_means[feature] = np.mean(X_full[feature])
        top_example_maxes[feature] = np.max(X_full[feature])
        top_example_mins[feature] = np.min(X_full[feature])
        
        top_examples = []
        for idx in top_indices: 
            top_examples.append({
                "NPI": X_full.index[idx],
                "SHAP": shap_vals[idx],
                "FeatureValue": X_full.iloc[idx, i],
                "Prediction": clf.predict_proba(X_full)[idx, 1]
            })
            
        top_example_tables[feature] = top_examples
    
    shap_force_images = {}
    for feature in top_features: 
        i = X_full.columns.get_loc(feature)
        shap_vals = shap_matrix[:, i]
        top_idx = np.argmax(np.abs(shap_vals))
        
        shap.plots.waterfall(shap_values[top_idx], show=False)
        filename = f'{os.getcwd()}\\shap_force_{feature}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        
        shap_force_images[feature] = filename
        
    shap_feature_table = mean_shap.reset_index()
    shap_feature_table.columns = ['Feature', 'Mean_SHAP']
    shap_feature_table = shap_feature_table.loc[shap_feature_table['Feature'].isin(top_features)]
    shap_feature_table = shap_feature_table.to_dict(orient='records')
    
    
    #### PARTIAL DEPENDENCE PLOT ###

    from sklearn.ensemble import GradientBoostingClassifier 
    from sklearn.inspection import PartialDependenceDisplay
    
    temp_clf = GradientBoostingClassifier(**XGB_PARAMS)
    temp_clf.fit(X_full, y_full)
    new_import = pd.DataFrame({
        'Feature': X_full.columns, 
        'Importance': temp_clf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    top_six = new_import['Feature'].to_list()[:6]

    fig, axs = plt.subplots(3, 2, figsize=(15,15))
    PartialDependenceDisplay.from_estimator(temp_clf, X_full, features=top_six, ax=axs, response_method='predict_proba', method='brute')
    plt.tight_layout()
    par_dep_path = f'{os.getcwd()}\\partial_dependence.png'
    plt.savefig(par_dep_path)
    plt.close()
    
    
    #### PERMUTATION IMPORTANCE ####
    
    results = permutation_importance(clf, X_full, y_full, n_repeats=10)
    importances = results.importances_mean 
    std = results.importances_std 
    indices = importances.argsort()[::-1]
    feature_names = X_full.columns[indices]
    
    plt.figure(figsize=(8,6))
    plt.barh(feature_names, importances[indices], xerr=std[indices], align='center')
    plt.xlabel('Mean Decrease in Accuracy')
    plt.title('Permutation Importance')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    perm_import_path = f'{os.getcwd()}\\permutation_importance.png'
    plt.savefig(perm_import_path)
    plt.close()
    
    #### TREE VISUALIZATION ####
    
    # fig, ax = plt.subplots(figsize=(20,10), dpi=300)
    # plot_tree(clf, num_trees=0, rankdir='UT', ax=ax)
    tree_path = f'{os.getcwd()}\\xgb_tree.png'
    plot_xgb_tree_manual(clf, tree_path, 0, (50, 10))
    
    #### OUTPUT PDF REPORT ####
        
    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template("report_template.html")
    html_content = template.render(
        metrics=metrics, 
        feature_table=feature_df.to_dict(orient="records"),
        prediction_labels_image=image_path,
        feature_importance_image=fi_path, 
        class_summary=class_summary, 
        shap_summary_image=shap_summary_filename,
        shap_feature_table=shap_feature_table,
        top_features=top_features,
        shap_force_images=shap_force_images,
        top_example_tables=top_example_tables,
        top_example_means=top_example_means, 
        top_example_maxes=top_example_maxes,
        top_example_mins=top_example_mins, 
        par_dep_path=par_dep_path, 
        perm_import_path=perm_import_path, 
        tree_path=tree_path
    )
    
    path_wkhtmltopdf = r'wkhtmltopdf.exe'
    config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
    options = {'enable-local-file-access': None}
    pdfkit.from_string(html_content, output_path, configuration=config, options=options)
    webbrowser.open_new_tab(f'{os.getcwd()}\\{output_path}')

import networkx as nx 
from scipy.special import expit

    # if root is None: 
    #     roots = [n for n,d in G.in_degree() if d==0]
    #     root = roots[0] if roots else list(G.nodes)[0]
        
    # from functools import lru_cache
    # @lru_cache(None)
    # def leaf_count(node): 
    #     children = list(G.successors(node))
    #     if not children: 
    #         return 1
    #     return sum(leaf_count(c) for c in children)
    
    # total_leaves = leaf_count(root)

def plot_xgb_tree_manual(clf, tree_path, tree_index=0, figsize=(30,10)): 
    
    def parse_node_id(x):
        if isinstance(x, (int, float)): 
            return int(x)
        s = str(x)
        if '-' in s: 
            return int(s.split('-',1)[1])
        return int(s)
    
    df = clf.get_booster().trees_to_dataframe()
    tree_df = df[df['Tree'] == tree_index]
    
    G =  nx.DiGraph()
    
    for _, row in tree_df.iterrows():
        node_id = parse_node_id(row['Node'])
        if row['Feature'] == 'Leaf': 
            leaf_value = float(row['Gain'])
            prob = expit(leaf_value)
            pred = int(prob >= 0.5)
            label = f'Leaf\nPred: {pred}\nProb: {prob:.2f}'
        else: 
            split = float(row['Split'])
            label = f'{row["Feature"]}\n< {split:.3f}'
            
        G.add_node(node_id, label=label)
        
    for _, row in tree_df.iterrows(): 
        if row['Feature'] != 'Leaf': 
            parent = parse_node_id(row['Node'])
            yes_id = parse_node_id(row['Yes'])
            no_id = parse_node_id(row['No'])
            G.add_edge(parent, yes_id, label='Yes')
            G.add_edge(parent, no_id, label='No')
            
    #print(G)
    #print(G.nodes)
    
    pos = hierarchy_pos(G, root=0)
        
    plt.figure(figsize=figsize)
    
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=5000, node_shape='s')
    nx.draw_networkx_edges(G, pos)
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels, font_size=10)
    
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    
    plt.axis('off')
    plt.title(f'XGBoost Tree {tree_index}')
    plt.tight_layout()
    plt.savefig(tree_path)
    plt.close()

def hierarchy_pos(
    G, root=None, width=1500.0, vert_gap=0.2, vert_loc=0, xcenter=None, sibling_gap=0.15
):
    if root is None:
        root = list(G.nodes)[0]
    if xcenter is None:
        xcenter = width / 2.0

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def leaf_count(node):
        children = list(G.successors(node))
        if not children:
            return 1
        return sum(leaf_count(child) for child in children)

    def _hierarchy_pos(node, left, right, vert_loc, pos):
        x = (left + right) / 2.0
        pos[node] = (x, vert_loc)
        children = list(G.successors(node))
        if not children:
            return pos

        total_leaves = sum(leaf_count(child) for child in children)
        total_gap = sibling_gap * (len(children)-1) if len(children) > 1 else 0 
        width_available = (right-left) - total_gap
        
        start = left
        for child in children:
            child_leaves = leaf_count(child)
            child_width = (width_available) * (child_leaves / total_leaves)
            child_left = start
            child_right = start + child_width
            pos = _hierarchy_pos(child, child_left, child_right, vert_loc - vert_gap, pos)
            start += child_width + sibling_gap
        return pos

    return _hierarchy_pos(root, 0, width, vert_loc, pos={})


def run_unsupervised(X, y):
    #print(X.columns.to_list())
    #X = X.sample(n=500)
    #y = y.loc[X.index]
    #print(X.shape[0], X.shape[1])
    y = y > 0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Standardize 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    new_scaled = scaler.fit_transform(X_test)
    
    # PCA 
    pca = PCA().fit(X_scaled)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.searchsorted(cumulative_variance, 0.95) + 1
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    new_pca = pca.transform(new_scaled)
    
    # UMAP 
    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=10,
        min_dist=0.8,
        metric='euclidean',
        target_metric='categorical',
        low_memory=False
        )
    #X_umap = umap_model.fit_transform(X_pca, y=y)
    X_umap = umap_model.fit_transform(X_pca, y_train)
    new_umap = umap_model.transform(new_pca)
    
    # Plot embedding 
    df = pd.DataFrame({
        'UMAP1': X_umap[:, 0],
        'UMAP2': X_umap[:, 1], 
        'label': y_train.astype(str)
    })
    test_df = pd.DataFrame({
        'UMAP1': new_umap[:, 0],
        'UMAP2': new_umap[:, 1], 
        'label': y_test.astype(str)
    })
    
    plot = (
        p9.ggplot(df, p9.aes(x='UMAP1', y='UMAP2', color='label')) + 
        p9.geom_point(alpha=0.6, size=2) + 
        p9.theme_bw() + 
        p9.scale_color_brewer(type='qual', palette='Set1') + 
        p9.ggtitle('UMAP projection after PCA') + 
        p9.labs(color='iDose')
    )
    plot.save('UMAP.png')
    
    test_plot = (
        p9.ggplot(test_df, p9.aes(x='UMAP1', y='UMAP2', color='label')) + 
        p9.geom_point(alpha=0.6, size=2) + 
        p9.theme_bw() + 
        p9.scale_color_brewer(type='qual', palette='Set1') + 
        p9.ggtitle('Test-UMAP projection after PCA') + 
        p9.labs(color='iDose')
    )
    test_plot.save('UMAP_test.png')
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=20, prediction_data=True)
    cluster_labels = clusterer.fit_predict(new_umap)
    
    test_df['cluster'] = cluster_labels.astype(str)
    test_df['cluster'] = test_df['cluster'].fillna(-1).astype(int).astype(str)
    # unique_clusters = df['cluster'].unique()
    # palette = sns.color_palette('tab10', n_colors = len(unique_clusters))
    # cluster_to_color = {
    #     c: "#d3d3d3" if c == '-1' else palette[i % len(palette)]
    #     for i, c in enumerate(unique_clusters)
    # }
    
    cluster_plot = (
        p9.ggplot(test_df, p9.aes(x='UMAP1', y='UMAP2', color='cluster')) + 
        p9.geom_point(alpha=0.7, size=2) + 
        p9.theme_bw() + 
        #p9.scale_color_manual(values=cluster_to_color) + 
        p9.ggtitle('HDBSCAN Clusters on UMAP Embedding') + 
        p9.labs(color='Cluster')
    )
    
    cluster_plot.save('UMAP_clusters.png')
    
    cluster_df = X_test
    cluster_df['cluster'] = cluster_labels.astype(str)
    
    print(cluster_df.groupby('cluster').mean())
    print(cluster_df.groupby('cluster').median())
    print(cluster_df.groupby('cluster').std())
    print(cluster_df['cluster'].value_counts())
    
    for col in cluster_df.columns[:-1]: 
        groups = [group[col].values for name, group in cluster_df.groupby('cluster')]
        stat, p = f_oneway(*groups)
        if p < 0.05: 
            print(f'{col}: p={p:.4f} (likely differs across clusters)')
    
    print(pd.crosstab(cluster_df['cluster'], y_test))
    
    features_to_plot = ['GONIOTOMY',
                        'SHUNT',
                        'EXTERNAL_DRAIN_DEV',
                        'TRABS',
                        'MIGS',
                        'CATARACTS',
                        'LASER_PROCEDURES',
                        'COMBO_CAT',
                        'MIOTICS',
                        'DURYSTA',
                        'DIAGNOSTIC_IMAGING',
                        'PROSTAGLANDIN_ANALOGS',
                        'RHO_KINASE_INHIB']
    
    n = len(features_to_plot)
    ncols = 3
    nrows = (n+ncols-1)//ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols,4*nrows), squeeze=False)
    
    for i, feature in enumerate(features_to_plot): 
        ax = axes[i // ncols][i % ncols] 
        sns.violinplot(data=cluster_df, x='cluster', y=feature, ax=ax)
        ax.set_title(f'{feature} by Cluster')
    
    plt.tight_layout()
    plt.savefig('violin_plots.png')
    
    for cluster, clus_df in cluster_df.groupby('cluster'): 
        clus_df.to_csv(f'cluster_{cluster}.csv')
    
    
if __name__ == '__main__': 
    main()
