There are 2 different datasets

They are given in the folder: DATASETS

For EDA: USE            ---->      btc_full_feature_set_daily.csv 

For running models: USE ---->      btc_clean.csv

If you wish to reproduce this thesis, set up your folder structure however you like. 
Mine looks like the folder structure in this repository but I like to paste btc_clean.csv (For running models) or btc_full_feature_set_daily.csv (for EDA) in every single directory with its .py file so that I won't have to change the file directory in the code. 
(I DID NOT PASTE btc_clean.csv or btc_full_feature_set_daily.csv in every file directory in this repository)
If both the csv and the .py file are in the same directory, I just need to 'df = pd.read_csv(csv_path)' and it runs seemlessly. 




