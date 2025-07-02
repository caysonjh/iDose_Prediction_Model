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

from code_groupings import new_feats

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
    parser.add_argument('--data', type=str, help='Data file containing medscope information', required=True)
    parser.add_argument('--pred_idos_val', action='store_true', help='If the model should predict the number of iDose patients')
    parser.add_argument('--pred_idos_bool', action='store_true', help='If the model should predict if iDose is/should be used')
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
    
    args = parser.parse_args()
    
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
                    args.custom_feats)
    
    print(f'iDose Features: {sum(y > 0)}')
    print(f'Non iDose Features: {sum(y == 0)}')
    
    if args.pred_idos_val: 
        pred_idos_val(X, y, args.grid_search, args.data_consolidation_level)
    elif args.pred_idos_bool: 
        pred_idos_bool(X, y, args.grid_search, args.data_consolidation_level)
    else:
        print('Please select a model to use with --pred_idos_val or --pred_idos_bool')


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


def prep_data(data, data_consolidation_level, time_features=False, start_year=None, end_year=None, custom_feats=None):     
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
    
    X = std_df.drop(IDOS_VAL_COLUMN, axis=1)
    X = X.div(X.sum(axis=1), axis=0)
    #print(X.sum(axis=1))
    #X.columns = [col + ' Total' for col in X.columns]
    #X['Total Number of Patients'] = X.sum(axis=1)
    
    if time_features:
        time_df = new_df.loc[:, new_df.columns.str.contains('In 20')]
        X = pd.concat([X, calculate_time_features(time_df, start_year, end_year)], axis=1)
        
    y = new_df[IDOS_VAL_COLUMN]
    return X, y
 
   
        
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
    
    co_path = "C:\\Users\\chamilton\\Cayson_Dirs\\idose_model\\correlation_plot.png"
    plot_correlation(clf, X_test, y_test, co_path)
    generate_model_report(clf, X_test, y_test, X, y, co_path, 'Regression', output_path=f'xgb_report_consol{consolidation_level}.pdf', top_n_features=20)


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
    cm_path = "C:\\Users\\chamilton\\Cayson_Dirs\\idose_model\\confusion_matrix.png"
    plot_confusion_matrix(clf, X_test, y_test, cm_path) 
    generate_model_report(clf, X_test, y_test, X, y, cm_path, 'Binary', output_path=f'xgb_report_consol{consolidation_level}.pdf', top_n_features=20)


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
    #print(list(importances.items())[:10])
    plot_importance(clf, importance_type='gain', max_num_features=max_num_features)
    plt.savefig('importances.png') 
    
    contributions = np.array(list(importances.values()))/sum(importances.values())*100
    feature_df = pd.DataFrame({
        'Feature': importances.keys(), 
        'Importance': importances.values(), 
        'Contribution': contributions
    }).sort_values(by="Importance", ascending=False)   
    
    return feature_df
   

def plot_confusion_matrix(clf, X_val, y_val, path):
    ConfusionMatrixDisplay.from_estimator(clf, X_val, y_val, cmap='Blues')
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
    
    clf.fit(X_full, y_full)
    feature_df = get_importances(clf, top_n_features)
    if feature_df is not None: 
        fi_path = 'C:\\Users\\chamilton\\Cayson_Dirs\\idose_model\\importances.png'
    
    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template("report_template.html")
    html_content = template.render(
        metrics=metrics, 
        feature_table=feature_df.to_dict(orient="records"),
        prediction_labels_image=image_path,
        feature_importance_image=fi_path, 
        class_summary=class_summary
    )
    
    path_wkhtmltopdf = r'C:\\Users\\chamilton\\Cayson_Dirs\\wkhtmltox\\bin\\wkhtmltopdf.exe'
    config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
    options = {'enable-local-file-access': None}
    pdfkit.from_string(html_content, output_path, configuration=config, options=options)
    webbrowser.open_new_tab(f'C:\\Users\\chamilton\\Cayson_Dirs\\idose_model\\{output_path}')

    
if __name__ == '__main__': 
    main()
