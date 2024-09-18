# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 10:40:40 2024

@author: arianeg
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import scipy.io as spio
from BP_avg import BP_avg
import xgboost as xg
from feat_selection_manuscript import feat_selection
from ccc import concordance_correlation_coefficient
from error_comp_manuscript import error_comp_manuscript


#user set parameters
meas_types = ['PPG', 'BFi + PPG'] #set measurement type. 'PPG' builds model with only PPG features. 'BFi + PPG' uses all features
experiment_name = 'data' #save excel file name
save_flag = 1 # set as 1 to save file, other to not save file
feat_select_flag = 1 # set as 1 to use feature selection
max_features = 25 #set number of features to include in model
cal_flag = 1 #set as 1 for calibrated results, 0 for uncalibrated results
num_avg = 15 #set number of BP points to be averaged
stride = 5 #set stride for BP averaging 
bp_type = 1 #set as 0 to predict SBP, 1 to predict DBP
num_subjects = 10 #set number of subjects
repeat_subjects = [2, 3, 6, 7, 9, 11, 13, 22, 24, 25, 29, 1, 12, 17, 30, 21, 27, 4, 5, 23] #indices of repeat subjects (to extract correct hyperparameter file)

if bp_type == 0:
    bp_add = '' #set hyperparameter file name depending on SBP vs DBP
    #initialize arrays for storinng SBP- these are only used for time trace plots
    Y_real_SBP_both_in = np.array([])
    Y_predicted_SBP_both_in = np.array([])
    Y_real_SBP_ppg_in = np.array([])
    Y_predicted_SBP_ppg_in = np.array([])
elif bp_type == 1:
    bp_add = '_dbp' #set hyperparameter file name depending on SBP vs DBP
    end_idx = np.array([0]) #save length of each subject's data for time trace plots
    #initialize arrays for storinng DBP- these are only used for time trace plots
    Y_real_DBP_both_in = np.array([])
    Y_predicted_DBP_both_in = np.array([])
    Y_real_DBP_ppg_in = np.array([])
    Y_predicted_DBP_ppg_in = np.array([])


#initialize arrays to store errors
rmse_store = []
mae_store = []
me_store = []
sd_store = []
table_data = []
me = np.array([])
sd = np.array([])
mae = np.array([])
rmse = np.array([])
ccc = np.array([])
index = np.array([])
num_pulses = np.array([])
diffs_store = np.array([])
mae_subj_store = []


 
#predict BP for measurement types specified above 
for z in range(len(meas_types)):
    meas_type = meas_types[z]

    
    
    all_subjects = list(range(0, num_subjects))

    #load data
    if meas_type == 'PPG':
        matData = spio.loadmat('all_subj_features_R_PPG.mat')
        feature_names_data = spio.loadmat('feature_names_type.mat')
        best_params = np.load('best_params_repeat_new_hyperopt'+ '116'+ '_30_PPG' + bp_add + '.npy', allow_pickle= True)

    else:
        matData = spio.loadmat('all_subj_features_R.mat')
        feature_names_data = spio.loadmat('feature_names_type_no_nan.mat')    
        best_params = np.load('best_params_repeat_new_hyperopt' + '116'+ '_30' + bp_add + '.npy', allow_pickle= True)

        
    
    
    feature_names = (feature_names_data['names_all'])
    feature_names = feature_names[0]
    
    # initialize arrays to store error for specific measurement type
    mae_all_subj = np.array([])
    mae_std_all_subj = np.array([])
    error_all_subj = np.array([])
    std_all_subj = np.array([])
    r_all_subj = np.array([])
    rmse_all_subj = np.array([])
    Y_pred_store = np.array([])
    Y_real_store = np.array([])

    

    #predict BP for each subject 
    for i in all_subjects:
        test_subjects = [i]
  

  
        X_all = np.array(matData['X']) #features from 1st measurement
        Y_all = np.array(matData['Y']) #BP values from 1st measurement
        X_all_R = np.array(matData['X_R']) #features from 2nd measurement
        Y_all_R = np.array(matData['Y_R']) #BP values from 2nd measurement
            
        #calculate feature #
        feat_size = np.shape(X_all[0, i])[0];
        
        #initialize arrays to store training and testing data
        X_test, Y_test = np.empty((feat_size, 0)), np.empty((1, 0))
        X_train, Y_train = np.empty((feat_size, 0)), np.empty((1, 0))
            
        #select training and testing data
        for s in test_subjects:
            X_test = np.append(X_test, X_all_R[0, s], axis = 1) #test on 2nd measurement
            Y_test = np.append(Y_test, [Y_all_R[0, s][bp_type]], axis = 1)
            X_train = np.append(X_train, X_all[0, s], axis = 1) #train on 1st measurement
            Y_train = np.append(Y_train, [Y_all[0, s][bp_type]], axis = 1)
            
        X_test = np.transpose(X_test)
        Y_test = np.transpose(Y_test)
        X_train = np.transpose(X_train)
        Y_train = np.transpose(Y_train)

        #initialize arrays to store BP values
        Y_predicted, Y_real, Y_train_predicted = np.array([]), np.array([]), np.array([])
        
        
        #build xgboost model
        r_subj = repeat_subjects[i]-1        
        params = best_params[r_subj]
        params['max_depth'] = int(params['max_depth'])
        reg = xg.XGBRegressor(objective ='reg:squarederror',  seed = 123, n_estimators = 500,  **params)
                              
        #perform features selection if flag is set to 1
        if feat_select_flag == 1:
            X_test, X_train = feat_selection(X_train, Y_train, X_test, Y_test, reg, feature_names, max_features)
            

        #predict BP
        reg.fit(X_train, Y_train[:])    
        Y_predicted = np.append(Y_predicted, reg.predict(X_test)) #predict BP
        Y_train_predicted = np.append(Y_train_predicted, reg.predict(X_train)) #in case you want to check how model worked on training data
        Y_real = np.append(Y_real, Y_test[:]) #store true BP values

        
        #calibrate predicted BP values
        if cal_flag == 1:
                  
            #slope calibration 1st and last point
            reg_cal = LinearRegression().fit(np.transpose([[0,  np.mean([Y_real[-1], Y_real[0]]) ]]), np.transpose([[0, np.mean([Y_predicted[-1], Y_predicted[0]])]]))
            b1 = reg_cal.intercept_[0]
            Y_predicted = np.multiply(Y_predicted, 1/reg_cal.coef_[0])
            reg_cal = LinearRegression().fit(np.transpose([[0, np.mean([Y_predicted[-1], Y_predicted[0]])]]), np.transpose([[0,  np.mean([Y_real[-1], Y_real[0]])]]))
            b2 = reg_cal.intercept_[0]
            Y_predicted = Y_predicted - b2
      
        #average predicted bp values
        Y_predicted, Y_real = BP_avg(num_avg, stride, Y_predicted, Y_real)
   
        #store errors for each subject
        error_all_subj = np.append(error_all_subj, round(np.mean(Y_real - Y_predicted), 3))
        mae_all_subj = np.append(mae_all_subj, round(np.mean(abs(Y_real - Y_predicted)), 3))
        mae_std_all_subj = np.append(mae_std_all_subj,  round(np.std(abs(Y_real - Y_predicted)), 3))
        std_all_subj = np.append(std_all_subj,  round(np.std(Y_real - Y_predicted), 3))
        r_all_subj = np.append(r_all_subj, round(concordance_correlation_coefficient(Y_real, Y_predicted), 3))
        rmse_all_subj = np.append(rmse_all_subj, round(np.sqrt(np.mean(np.square(Y_real - Y_predicted))), 3))
        Y_pred_store = np.append(Y_pred_store, Y_predicted)
        Y_real_store = np.append(Y_real_store, Y_real)
        mae_subj_store.append(abs(Y_real - Y_predicted))
        
        if bp_type ==1 and meas_type == 'BFi + PPG':    
            Y_real_DBP_both_in = np.append(Y_real_DBP_both_in, Y_real)
            Y_predicted_DBP_both_in = np.append(Y_predicted_DBP_both_in, Y_predicted)
            end_idx = np.append(end_idx, len(Y_real_DBP_both_in))
        elif bp_type ==0 and meas_type == 'BFi + PPG':
            Y_real_SBP_both_in = np.append(Y_real_SBP_both_in, Y_real)
            Y_predicted_SBP_both_in = np.append(Y_predicted_SBP_both_in, Y_predicted)    
        if bp_type ==1 and meas_type == 'PPG':    
            Y_real_DBP_ppg_in = np.append(Y_real_DBP_ppg_in, Y_real)
            Y_predicted_DBP_ppg_in = np.append(Y_predicted_DBP_ppg_in, Y_predicted)
            print('ppg stored')
        elif bp_type ==0 and meas_type == 'PPG':
            Y_real_SBP_ppg_in = np.append(Y_real_SBP_ppg_in, Y_real)
            Y_predicted_SBP_ppg_in = np.append(Y_predicted_SBP_ppg_in, Y_predicted)  
            print('ppg stored')
            
       
        
    #generate figures
    [table_data, me, mae, sd, rmse, ccc, index, diffs_store, num_pulses, rmse_store, mae_store, me_store, sd_store, mae_all_subj] = error_comp_manuscript(table_data, error_all_subj, mae_all_subj, std_all_subj, r_all_subj, rmse_all_subj, Y_real_store, Y_pred_store, num_pulses, mae, me, sd, rmse, rmse_store, mae_store, me_store, sd_store, ccc, index, diffs_store, z, bp_type, all_subjects, num_subjects, meas_types, meas_type, experiment_name, save_flag)
    #store data for subject time traces










#%% time trace plots

meas1_t = np.array(matData['t_all_subj'])
meas2_t = np.array(matData['t_r_all_subj'])

#select subject to plot. Note that you must run the above code for both SBP and DBP to generate the plot
subj = 11

t1 = meas1_t[0, subj-1]
t2 = meas2_t[0, subj-1]
t3 = np.zeros(np.shape(t2))
offset = 0
t_meas_type = []
t_meas_idx = []
for i in range(len(t2[0])-1):
    t3[0, i] = t2[0,i] + offset
    if (t2[0,i+1] - t2[0,i]) < 0:
        print(i)
        offset = offset + t2[0,i]
        print(offset)
        t_meas_type = np.append(t_meas_type, offset)
        t_meas_idx = np.append(t_meas_idx, i)
t3[0, i+1] = t2[0, i + 1] + offset
  

t1, t3 = BP_avg(num_avg, stride, t1[0], t3[0])


b_idx = [idx for idx, val in enumerate(t3) if (val < t_meas_type[0])]
e_idx = [idx for idx, val in enumerate(t3) if (t_meas_type[0] < val < t_meas_type[1])]

r_idx = [idx for idx, val in enumerate(t3) if (t_meas_type[1] < val < max(t3) )]



Y_real_SBP_both = Y_real_SBP_both_in[int(end_idx[subj-1]):int(end_idx[subj])]
Y_real_SBP_both = Y_real_SBP_both_in[int(end_idx[subj-1]):int(end_idx[subj])]
Y_predicted_SBP_both = Y_predicted_SBP_both_in[int(end_idx[subj-1]):int(end_idx[subj])]
Y_predicted_SBP_both = Y_predicted_SBP_both_in[int(end_idx[subj-1]):int(end_idx[subj])]
Y_real_DBP_both = Y_real_DBP_both_in[int(end_idx[subj-1]):int(end_idx[subj])]
Y_real_DBP_both = Y_real_DBP_both_in[int(end_idx[subj-1]):int(end_idx[subj])]
Y_predicted_DBP_both = Y_predicted_DBP_both_in[int(end_idx[subj-1]):int(end_idx[subj])]
Y_predicted_DBP_both = Y_predicted_DBP_both_in[int(end_idx[subj-1]):int(end_idx[subj])]


color_scatter = np.array([134, 157, 230])
color_scatter = color_scatter/max(color_scatter)
color_scatter = 'xkcd:deep lilac'

sbp_color = color_scatter
dbp_color = color_scatter



alpha_all = 0.9
lw = 2
lw2 = 3

fig = plt.figure(figsize= (7.6, 1.88), dpi = 600)
plt.plot(t3[b_idx], Y_real_SBP_both[b_idx], 'gray', linewidth = lw)
plt.plot(t3[e_idx], Y_real_SBP_both[e_idx], 'gray', linewidth = lw)
plt.plot(t3[r_idx], Y_real_SBP_both[r_idx], 'gray', linewidth = lw)

plt.plot(t3[b_idx], Y_predicted_SBP_both[b_idx], color = sbp_color, alpha = alpha_all, linewidth = lw2)
plt.plot(t3[e_idx], Y_predicted_SBP_both[e_idx],  color = sbp_color, alpha = alpha_all, linewidth = lw2)
plt.plot(t3[r_idx], Y_predicted_SBP_both[r_idx], color =  sbp_color, alpha = alpha_all, linewidth = lw2, label = 'Predicted DBP')

plt.plot(t3[b_idx], Y_real_DBP_both[b_idx], 'gray', linewidth = lw)
plt.plot(t3[e_idx], Y_real_DBP_both[e_idx], 'gray', linewidth = lw)
plt.plot(t3[r_idx], Y_real_DBP_both[r_idx], 'gray', linewidth = lw, label = 'Finapres BP')

plt.plot(t3[b_idx], Y_predicted_DBP_both[b_idx], color = dbp_color, alpha = alpha_all, linewidth = lw2)
plt.plot(t3[e_idx], Y_predicted_DBP_both[e_idx], color = dbp_color, alpha = alpha_all, linewidth = lw2)
plt.plot(t3[r_idx], Y_predicted_DBP_both[r_idx], color =  dbp_color, alpha = alpha_all, linewidth = lw2, label = 'Predicted DBP')

ax = fig.gca()
ax.fill_betweenx(np.arange(min(Y_real_DBP_both) - 10, max(Y_predicted_SBP_both) + 10, .01), t_meas_type[0], t_meas_type[1], facecolor = 'xkcd:cool grey', alpha = 0.3)
ax_label_size = 9
font_all = "Arial"

ax = fig.gca()

ax.tick_params(axis="x",direction="in", pad=-5)
ax.tick_params(axis='both', which='major', labelsize= ax_label_size)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_visible(False)
    
plt.yticks(fontname = font_all) 
plt.xticks(fontname = font_all) 
plt.tick_params(bottom = False)
plt.tick_params(left = False)




#%%statistical test
from scipy.stats import wilcoxon

rmse_store_reshape = np.reshape(mae_store, (2, int(len(mae_store)/2)))

stat, p = wilcoxon(np.transpose(rmse_store_reshape[0]- rmse_store_reshape[1]), alternative = 'greater')

print('Statistics=%.3f, p=%.3f' % (stat, p))