# iDose_Prediction_Model

## Environment Set-up

Program usage depends first and foremost on preparing the python environment and including the required packages. 

### 1. Clone this directory to your machine

First, you will need to [install Git](https://git-scm.com/downloads/win), which will allow me to share the code with you, update it as needed, and get you the updated version.

Next, open the terminal app on your machine and run the following command to ensure git was installed correctly: 

```
git --version
```

You should see a confirmation of the version that is installed. Then, run this command to clone this directory onto your local machine: 

```
git clone git@github.com:caysonjh/iDose_Prediction_Model.git
```

### 2. Install Anaconda

Then, download the latest version of [Anaconda](https://www.anaconda.com/download) by following the link and providing your email. Anaconda is a package manager for python, which allows for easy installation of the machine learning packages we are using. 

### 3. Install VSCode

Once Anaconda has been installed, install [VSCode](https://code.visualstudio.com/download), which serves as a code/text editor and will be the platform through which we run the program. 

### 4. Install Required Packages

Open the folder that was downloaded to your commputer in VSCode. Create a terminal by selecting **Terminal** → **New Terminal**. The required packages can be installed by running the following 2 commands: 

```
conda env create -f environment.yml
conda activate iDose_Prediction
```

You are now ready to run the program

## Program Execution 

### Basic Use

First, ensure that `all_data_v2.csv` is located in the folder that you are currently in, you should be able to see it in the file explorer on the left side of VSCode. 

In the terminal you opened in VSCode, the program can be run using the following command: 

```
python .\pred_idose.py --data .\all_data_v2.csv --pred_idos_bool --data_consolidation_level 5 --custom_feats custom_features.txt --save_model
```

The program will automatically create and open a file called `xgb_report_consol5.pdf`, which contains information on the performance of the model and the feature importances. 

The --save_model flag will save the model in the directory you are working in as `xgb_model_cons[consolidation_level]_[date/time].pkl`, which can be used to make predictions on new data to identify new iDose users. 

For more information on what the `--` option flags denote, see the description down below. 

### Customization 

To change which features are included in the training of the model, you can modify the file `custom_features.txt`

**NOTE:** The names of the features in `custom_features.txt` must exactly match those found in `code_groupings.py`. 

If you would like to change which codes are combined into which categories, you can modify `code_groupings.py`, just ensure to keep the same formatting (eg. each code/prescription must be within single quotes, each one must be within the brackets corresponding to the group title -- see example ).
```
new_feats = {
"GROUP_TITLE" : ['Code1', 'Code2', 'Code3'],
"SECOND_GROUP" : ['Medicine1', 'Treatment2']
}
```

## Run Prediction

In order to run prediction on new data, you can run a command like the following: 

```
python pred_idose.py --pred_idos_bool --data_consolidation_level 5 --custom_feats custom_features.txt --predict your_prediction_file.csv --classifier your_model.pkl
```

If you don't yet have a classifier model saved, you can instead specify the `--data` flag as before, and the model will first train on the data and then use that same model to run predictions. 

## All Program Functionality 

#### Basic Model Training
- `--data`: Must be followed by the name of the file containing the information regarding each physician **(REQUIRED unless running `--predict` with `--classifier`)**
- `--pred_idos_bool`: Indicates that the model will predict a binary yes/no if the physician uses iDose
- `--pred_idos_val`: Indicates that the model will predict how many iDose cases a doctor has had
- `--data_consolidation_level`: Indicates how much to consolidate the individual code data into groups, must be followed by a number from 0-5 **(REQUIRED)**
    - `0` → No consolidation will be done, every code is evaluated separate
    - `1` → Codes are consolidated based on similarity 
    - `2` → Similarity consolidation and categories unrelated to glaucoma are removed
    - `3` → Similarity consolidation, unrelated removed, and diagnoses are removed 
    - `4` → Similarity consolidation, unrelated removed, diagnoses removed, and diagnostic imaging is removed
    - `5` → Fully custom feature selection, with selected values provided with `--custom_feats`
- `--custom_feats`: Must be followed by the file containing the names of the features/groupings that the model should train on **(REQUIRED IF `--data_consolidation_level 5`)**
- `--save_model`: Indicates that you want to save the trained model for running predictions in the future

#### Running Prediction
- `--predict`: Indicates that you want to run prediction on new data, the filename of which must be included
- `--classifier`: Indicates that there is a saved classifier that you want to use to make predictions on new data **(REQUIRED if running `--predict` without `--data`)**
- MUST ALSO INCLUDE: `--pred_idos_bool` or `--pred_idos_val`, `--data` or `--classifier` for prediction

#### Running Unsupervised Learning
- `--unsupervised_clusters`: Indicates that you want to run unsupervised clustering using the training data supplied by `--data`. This will generate multiple images showing various clustering analyses, but the generating of the full report isn't complete

#### Extra Customizations
- `--time_features`: Indicates that you want to include statistical features to evaluate the feature trends over time, must include start year and end year (eg. `--time_features 2022 2025`)
- `--extra_idose`: If you have a separate file with more iDose users that you want to combine with the rest of the dataset
- `--extra_non`: If you have a separate file with more Non-iDose users that you want to combine with the rest of the dataset
- `--grid_search`: Indicate if you want to try and identify the best parameters to train the XGBoost model