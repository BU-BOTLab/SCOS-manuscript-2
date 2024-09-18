# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:10:19 2024

@author: arianeg
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:52:07 2024

@author: arianeg
"""



import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import scipy.io as spio
from BP_avg import BP_avg
import xgboost as xg
from feat_selection_manuscript import feat_selection
from ccc import concordance_correlation_coefficient
from error_comp_manuscript import error_comp_manuscript


#user set parameters
meas_types = ['PPG', 'BFi + PPG'] #set measurement type. 'PPG' builds model with only PPG features. 'BFi + PPG' uses all features.
# 'PPG cold pressor' builds model with PPG features from cold pressor data. 'BFi + PPG cold pressor' builds model with all features from cold pressor data
# meas_types = ['PPG cold pressor', 'BFi + PPG cold pressor']
experiment_name = 'data' #save excel file name
save_flag = 1 # set as 1 to save file, other to not save file
feat_select_flag = 1 # set as 1 to use feature selection
max_features = 25 #set number of features to include in model
num_avg = 15 #set number of BP points to be averaged
stride = 5 #set stride for BP averaging 
feat_avg = 1 #set number of features to average prior to input (1 if no averaging)
bp_type = 1 #set as 0 to predict SBP, 1 to predict DBP
num_subjects = 30 #set number of subjects
cold = 0 #set to 1 if analyzing cold pressor data
shuffle_flag = True #set to true to shuffle data before splitting into folds

if bp_type == 0:
    bp_add = ''#set hyperparameter file name depending on SBP vs DBP
    #initialize arrays for storinng SBP- these are only used for time trace plots
    Y_real_SBP_both_in = np.array([])
    Y_predicted_SBP_both_in = np.array([])
    Y_real_SBP_ppg_in = np.array([])
    Y_predicted_SBP_ppg_in = np.array([])
elif bp_type == 1:
    bp_add = '_dbp'#set hyperparameter file name depending on SBP vs DBP
    #initialize arrays for storinng DBP- these are only used for time trace plots
    end_idx = np.array([0])
    Y_real_DBP_both_in = np.array([])
    Y_predicted_DBP_both_in = np.array([])
    Y_real_DBP_ppg_in = np.array([])
    Y_predicted_DBP_ppg_in = np.array([])

#initialize arrays to store errors
me = np.array([])
sd = np.array([])
mae = np.array([])
mae_sd = np.array([])
rmse = np.array([])
ccc = np.array([])
index = np.array([])
num_pulses = np.array([])
rmse_store = []
mae_store = []
table_data = []
diffs_store = np.array([])
me_store = []
sd_store = []


for z in range(len(meas_types)):
    
    
    meas_type = meas_types[z]

    #load data
    if meas_type == 'PPG':
        matData = spio.loadmat('all_subj_features_all_loc_incl_nan_PPG.mat')
        feature_names_data = spio.loadmat('feature_names_type.mat')
        best_params = np.load('best_params_repeat_new_hyperopt'+ '116'+ '_30_PPG' + bp_add + '.npy', allow_pickle= True)

    elif meas_type == 'BFi + PPG cold pressor':
        matData = spio.loadmat('all_subj_features_cold.mat')
        feature_names_data = spio.loadmat('feature_names_type.mat')
        best_params = np.load('best_params_repeat_new_hyperopt'+ '116'+ '_cold_30' + bp_add + '.npy', allow_pickle= True)

    elif meas_type == 'PPG cold pressor':
        matData = spio.loadmat('all_subj_features_cold_PPG.mat')
        feature_names_data = spio.loadmat('feature_names_type.mat')
        best_params = np.load('best_params_repeat_new_hyperopt'+ '116'+ '_cold_30_PPG' + bp_add + '.npy', allow_pickle= True)

    else:
        matData = spio.loadmat('all_subj_features_all_loc_incl_nan.mat')
        feature_names_data = spio.loadmat('feature_names.mat')
        best_params = np.load('best_params_repeat_new_hyperopt'+ '116'+ '_30' + bp_add + '.npy', allow_pickle= True)

    
    all_subjects = list(range(0, num_subjects))
    num_subjects = len(all_subjects)

    
    feature_names = (feature_names_data['names_all'])
    feature_names = feature_names[0]
    
    #initialize arrays to store error for specific measurement type
    mae_all_subj = np.array([])
    mae_std_all_subj = np.array([])
    error_all_subj = np.array([])
    std_all_subj = np.array([])
    r_all_subj = np.array([])
    rmse_all_subj = np.array([])
    Y_pred_store = np.array([])
    Y_real_store = np.array([])
    
    for s in all_subjects:
        test_subjects = [s]
        params = best_params[s] #select hyperparameters for subject s
        params['max_depth'] = int(params['max_depth'])

        #convert training and testing data to numpy array
        if cold == 1:
            X_all = np.array(matData['X_R'])
            Y_all = np.array(matData['Y_R']) 
            t_all_subj = np.array(matData['t_r_all_subj'])
        else:
            X_all = np.array(matData['X'])
            Y_all = np.array(matData['Y'])
            t_all_subj = np.array(matData['t_all_subj'])
        

        
        t_in = t_all_subj[0, s][0] #time data for subject s
        t = np.array([]) #initialize array to store time data before shuffling
        X = X_all[0, s] #feature data for subject s
        Y = Y_all[0, s][bp_type] #BP data for subject s
        X = np.transpose(X)
        Y = np.transpose(Y)
    

        #set 5 fold cross validation parameters
        kf = KFold(n_splits=5, shuffle = shuffle_flag, random_state= 0)
        Y_predicted, Y_real = np.array([]), np.array([])
        
        #set model parameters
        reg = xg.XGBRegressor(objective ='reg:squarederror', n_estimators = 500, seed = 123, **params)
        
        #5 fold cross validation
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index] #separate into training and testing data based on 5 fold CV parameters
            y_train, y_test = Y[train_index], Y[test_index]
            X_test, X_train = feat_selection(X_train, y_train, X_test, y_test, reg, feature_names, max_features) #feature selection
            t = np.append(t, t_in[test_index]) #store timing information for testing data
            
  
            reg.fit(X_train, y_train[:]) #train model using only best features
            Y_predicted = np.append(Y_predicted, reg.predict(X_test)) #predict BP using model
            Y_real = np.append(Y_real, y_test[:]) #store true BP
      
        #unshuffle output data
        arr1inds = t.argsort()
        t = t[arr1inds[::1]]
        Y_real  = Y_real[arr1inds[::1]]
        Y_predicted = Y_predicted[arr1inds[::1]]
        
        #average output BP
        Y_predicted, Y_real = BP_avg(num_avg, stride, Y_predicted, Y_real)

        #store errors for measurement type        
        Y_pred_store = np.append(Y_pred_store, Y_predicted)
        Y_real_store = np.append(Y_real_store, Y_real)      
        error_all_subj = np.append(error_all_subj, round(np.mean(Y_real - Y_predicted), 3))
        mae_all_subj = np.append(mae_all_subj, round(np.mean(abs(Y_real - Y_predicted)), 3))
        mae_std_all_subj = np.append(mae_std_all_subj,  round(np.std(abs(Y_real - Y_predicted)), 3))
        std_all_subj = np.append(std_all_subj,  round(np.std(Y_real - Y_predicted), 3))  
        r_all_subj = np.append(r_all_subj, round(concordance_correlation_coefficient(Y_real, Y_predicted), 3))
        rmse_all_subj = np.append(rmse_all_subj, round(np.sqrt(np.mean(np.square(Y_real - Y_predicted))), 3))
        Y_pred_store = np.append(Y_pred_store, Y_predicted)
        Y_real_store = np.append(Y_real_store, Y_real)
        
        #store data for time trace plots
        if bp_type ==1 and (meas_type == 'BFi + PPG' or meas_type == 'BFi + PPG cold pressor'):    
            Y_real_DBP_both_in = np.append(Y_real_DBP_both_in, Y_real)
            Y_predicted_DBP_both_in = np.append(Y_predicted_DBP_both_in, Y_predicted)
            end_idx = np.append(end_idx, len(Y_real_DBP_both_in))
        elif bp_type ==0 and (meas_type == 'BFi + PPG' or meas_type == 'BFi + PPG cold pressor'):
            Y_real_SBP_both_in = np.append(Y_real_SBP_both_in, Y_real)
            Y_predicted_SBP_both_in = np.append(Y_predicted_SBP_both_in, Y_predicted)    
        if bp_type ==1 and (meas_type == 'PPG' or meas_type ==  'PPG cold pressor'):    
            Y_real_DBP_ppg_in = np.append(Y_real_DBP_ppg_in, Y_real)
            Y_predicted_DBP_ppg_in = np.append(Y_predicted_DBP_ppg_in, Y_predicted)
            print('ppg stored')
        elif bp_type ==0 and (meas_type == 'PPG' or meas_type ==  'PPG cold pressor'):
            Y_real_SBP_ppg_in = np.append(Y_real_SBP_ppg_in, Y_real)
            Y_predicted_SBP_ppg_in = np.append(Y_predicted_SBP_ppg_in, Y_predicted)  
            print('ppg stored')
            
            

    #compile errors
    [table_data, me, mae, sd, rmse, ccc, index, diffs_store, num_pulses, rmse_store, mae_store, me_store, sd_store, mae_all_subj] = error_comp_manuscript(table_data, error_all_subj, mae_all_subj, std_all_subj, r_all_subj, rmse_all_subj, Y_real_store, Y_pred_store, num_pulses, mae, me, sd, rmse, rmse_store, mae_store, me_store, sd_store, ccc, index, diffs_store, z, bp_type, all_subjects, num_subjects, meas_types, meas_type, experiment_name, save_flag)
     





#%% time trace plots

meas1_t = np.array(matData['t_all_subj'])

if (meas_type == 'PPG cold pressor' or meas_type == 'BFi + PPG cold pressor'):
    meas2_t = np.array(matData['t_r_all_subj'])
else:
    meas2_t = np.array(matData['t_all_subj'])
        

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