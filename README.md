# SCOS-manuscript-2

This repository contains the data and codes needed to replicate the results in our recent paper using SCOS to estimate blood pressure. 

Codes:
'five_fold_CV_BP_estimation_manuscript' predicts BP from original 30 subjects using 5 fold cross validation (manuscript figure 2) 
'longitudinal_BP_estimation_manuscript' predicts BP for 20 subject measuremed several weeks after first measurement (manuscript figure 3)
'BP_avg' - function to average predicted BP
'error_comp'- function to compile errors into table and save
'feat_selection_manuscript' - function to select features 

Data:
'all_subj_features_all_loc_incl_nan' - BFi + PPG features for 30 original subjects (5 fold CV)
'all_subj_features_all_loc_incl_nan_PPG' - PPG features for 30 original subjects (5 fold CV)
'all_subj_features_cold' - BFi + PPG features for 5 cold pressor subjects (5 fold CV)
'all_subj_features_cold_PPG' - PPG features for 5 cold pressor subjects (5 fold CV)
'all_subj_features_R'- BFi + PPG features for 20 repeat subjects (longitudinal measurement)
'all_subj_features_R_PPG'- PPG features for 20 repeat subjects (longitudinal measurement)

Each .mat file contains the same variables:

X - features over time for each subject
Y - BP over time for each subject
t_all_subject - timing of each pulse for each subject

The .mat files for the longitudingal measurement also contain the data for the second measurement:

X_R - features during second measurement 
Y_R - BP during second measurement 
t_r_all_subject - timing of each pulse during second measurement

Hyperparameter files: 
best_params_30.npy- hyperparameters for all 30 subjects, BFi + PPG features, SBP
best_params_30_PPG.npy- hyperparameters for all 30 subjects, PPG features, SBP
best_params_30_dbp.npy- hyperparameters for all 30 subjects, BFi + PPG features, DBP
best_params_30_PPG_dbp.npy- hyperparameters for all 30 subjects, PPG features, DBP
best_params_cold.npy- hyperparameters for cold pressor subjects, BFi + PPG features, SBP
best_params_cold_dbp.npy- hyperparameters for cold pressor subjects, BFi + PPG features, DBP
best_params_cold_PPG.npy- hyperparameters for cold pressor subjects, PPG features, SBP
best_params_cold_PPG_dbp.npy- hyperparameters for cold pressor subjects, PPG features, DBP
