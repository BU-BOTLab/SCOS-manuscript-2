# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:10:25 2024

@author: arianeg
"""

import numpy as np
import matplotlib.pyplot as plt
from ccc import concordance_correlation_coefficient
import pandas as pd

def error_comp_manuscript(table_data, error_all_subj, mae_all_subj, std_all_subj, r_all_subj, rmse_all_subj, Y_real_store, Y_pred_store, num_pulses, mae, me, sd, rmse, rmse_store, mae_store, me_store, sd_store, ccc, index, diffs_store, z, bp_type, all_subjects, num_subjects, meas_types, meas_type, experiment_name, save_flag):
    
    
    ax_label_size = 9
    dpi = 600
    x_square = 2.0
    y_square = 1.6
    x_mae = 8
    x_mae_avg = 2
    y_mae_avg = 1.4
    y_mae = 1.6
    x_me = 30
    y_me = 10
    scatter_size = 1.5
    lw1 = 1
    lw2 = 2
    lw3 = 2
    pad1 = 3.5
    pad2 = 2
    pad_mae_x = .5
    pad_mae_y = -2
    pad_mae_avg = 1
    
    
    color_scatter = 'darkgray'
    font_all = "Arial"
    ba_mean_line = 'black'
    
  
    #set colors for measurement types
    fig = plt.figure(10, figsize = (x_square, y_square), dpi = dpi)
    if meas_type == 'BFi + PPG' or meas_type == 'BFi + PPG cold pressor':
        color_scatter = 'xkcd:deep lilac'
        edgecolor = 'face'
        alpha = 1
    elif meas_type == 'PPG' or meas_type == 'PPG cold pressor':
        color_scatter = np.array([248, 180, 70])
        color_scatter = color_scatter/max(color_scatter)       
        edgecolor = 'face'
        alpha = 0.8
        
    #correlation between true and predicted BP, overlayed
    plt.scatter(Y_real_store, Y_pred_store, scatter_size, color = color_scatter, alpha = alpha, edgecolors= edgecolor)
    plt.axline((100, 100), slope = 1, color = "darkgray", linewidth = lw1, linestyle = 'dashed')
    a, b = np.polyfit(Y_real_store, Y_pred_store, 1)
    ax = fig.gca()
    ax.tick_params(axis='both', which='major', labelsize= ax_label_size)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_visible(False)      
    plt.yticks(fontname = font_all) 
    plt.xticks(fontname = font_all) 
    plt.tick_params(bottom = False)
    plt.tick_params(left = False)
    ax.tick_params(axis="y",direction="in", pad= pad1)
    ax.tick_params(axis="x",direction="in", pad= pad2)
    plt.savefig('corr1.png', dpi = 600)
  
    #bland altman plot data
    diffs = Y_real_store- Y_pred_store
    diffs_store = np.append(diffs_store, diffs)
    ba_lim = np.std(diffs)*1.96
    ba_mean = np.mean(diffs)
    BA_avg = (Y_real_store + Y_pred_store)/2
    
    #bland altman plot BFi + PPG overlayed on PPG
    fig = plt.figure(11, figsize = (x_square, y_square), dpi = dpi)
    plt.scatter(BA_avg, diffs, scatter_size, color_scatter, alpha = alpha)
    
    #add Bland altman lines based on BFi + PPG model data
    if meas_type == 'BFi + PPG':
        plt.axhline(y=ba_lim, color='darkgrey', linestyle='--', linewidth = lw2)
        plt.axhline(y=-1*ba_lim, color='darkgrey', linestyle='--', linewidth = lw2)
        plt.axhline(y= ba_mean, color=ba_mean_line, linestyle='-', linewidth = lw3)
        

    ax = fig.gca()
    ax.tick_params(axis='both', which='major', labelsize= ax_label_size)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_visible(False)
    
    plt.yticks(fontname = font_all) 
    plt.xticks(fontname = font_all)
    plt.tick_params(bottom = False)
    plt.tick_params(left = False)
    ax.tick_params(axis="y",direction="in", pad= pad1)
    ax.tick_params(axis="x",direction="in", pad= pad2)


    #correlation between true and predicted BP, not overlayed
    fig = plt.figure(z, figsize = (x_square, y_square), dpi = dpi)
    plt.scatter(Y_real_store, Y_pred_store, scatter_size, color = color_scatter, alpha = 0.5)
    plt.axline((100, 100), slope = 1, color = "darkgray", linewidth = lw1, linestyle = 'dashed')
    a, b = np.polyfit(Y_real_store, Y_pred_store, 1)
    ax = fig.gca()
    ax.tick_params(axis='both', which='major', labelsize= ax_label_size)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_visible(False)
        
    plt.yticks(fontname = font_all) 
    plt.xticks(fontname = font_all)
    plt.tick_params(bottom = False)
    plt.tick_params(left = False)
    ax.tick_params(axis="y",direction="in", pad= pad1)
    ax.tick_params(axis="x",direction="in", pad= pad1)


    
    #bland altman plot just BFi + PPG
    fig = plt.figure(z+2, figsize = (x_square, y_square), dpi = dpi)
    plt.scatter(BA_avg, diffs, scatter_size, color_scatter, alpha = alpha)
    plt.axhline(y=ba_lim, color='darkgrey', linestyle='--', linewidth = lw2)
    plt.axhline(y=-1*ba_lim, color='darkgrey', linestyle='--', linewidth = lw2)
    plt.axhline(y= ba_mean, color= ba_mean_line, linestyle='-', linewidth = lw3)


    ax = fig.gca()
    ax.tick_params(axis='both', which='major', labelsize= ax_label_size)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_visible(False)
    
    plt.yticks(fontname = font_all) 
    plt.xticks(fontname = font_all) 
    plt.tick_params(bottom = False)
    plt.tick_params(left = False)
    ax.tick_params(axis="y",direction="in", pad= pad1)
    ax.tick_params(axis="x",direction="in", pad= pad2)

    
    #compile errors into table
    me = np.append(me, [round(np.nanmean(error_all_subj), 2), round(ba_mean, 2)])
    mae = np.append(mae, [round(np.nanmean(mae_all_subj), 2), round(np.mean(abs(diffs)), 2)])
    sd = np.append(sd, [round(np.nanmean(std_all_subj), 2), round(np.std(diffs), 2)])
    rmse = np.append(rmse, [round(np.nanmean(rmse_all_subj), 2), round(np.sqrt(np.mean(np.square(Y_real_store - Y_pred_store))), 2)])
    rmse_store = np.append(rmse_store, rmse_all_subj)
    mae_store = np.append(mae_store, mae_all_subj)
    me_store = np.append(me_store, error_all_subj)
    sd_store = np.append(sd_store, std_all_subj)
    ccc = np.append(ccc, [round(np.nanmean(r_all_subj), 2), round(concordance_correlation_coefficient(Y_real_store, Y_pred_store), 2)])
    vals = {'ME (mmHg)': me, 'MAE (mmHg)': mae, 'SD (mmHg)': sd, 'RMSE (mmHg)': rmse, 'CCC': ccc}
    index = np.append(index, [meas_type, ' '])
    table_data = pd.DataFrame(data = vals, index = index)
    
    #store number of pulses included in analysis
    num_pulses = np.append(num_pulses, len(Y_pred_store))
    
    
    
    
    print(table_data)
    
    # subject error plots
    if len(meas_types) == 2:
        
        if meas_type == 'BFi + PPG'  or meas_type == 'BFi + PPG cold pressor':
            color_bar = color_scatter
        elif meas_type == 'PPG' or meas_type == 'PPG cold pressor':
            color_bar = color_scatter
        
        #subject mean error plot
        plt.rc('xtick', labelsize = ax_label_size)
        fig = plt.figure(5, figsize = (x_me, y_me), dpi = dpi)
        a = range(1, num_subjects+1)
        b = error_all_subj[0:num_subjects]
        plt.scatter(a, b, s = 450, label = meas_type, color = color_bar)         
        c = std_all_subj[0:num_subjects]         
        plt.errorbar(a, b, yerr=c, fmt="o", elinewidth= 8, color = color_bar)
        plt.axhline(y = 0,xmin = 1/(num_subjects*2), xmax = 1, c = 'k')
        plt.xticks([r+1 for r in range(len(a))], a)
        ax = fig.gca()
        ax.tick_params(axis='both', which='major', labelsize= ax_label_size)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_visible(False)

        plt.yticks(fontname = font_all) 
        plt.xticks(fontname = font_all) 
        ax.tick_params(axis="y",direction="in", pad=-2)
        plt.tick_params(bottom = False)
        plt.tick_params(left = False)

        
        #subject mean absolute error plot       
        fig = plt.figure(6, figsize = (x_mae, y_mae), dpi = dpi)
        barWidth = 0.25
        a = range(1, num_subjects + 1)
        b = mae_all_subj[0:num_subjects]

        
        br1 = np.arange(len(a))
        br2 = [x + barWidth for x in br1]
        xpos = [br1, br2]
        plt.bar(xpos[z], b, width = barWidth, label = meas_type, color = color_bar)  
        plt.xticks([r + barWidth for r in range(len(a))], a)

        ax = fig.gca()
        ax.tick_params(axis="y",direction="in", pad= pad_mae_y)
        ax.tick_params(axis="x",direction="in", pad= pad_mae_x)
        ax.tick_params(axis='both', which='major', labelsize= ax_label_size)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_visible(False)
        
        plt.yticks(fontname = font_all) 
        plt.xticks(fontname = font_all) 
        plt.tick_params(bottom = False)
        plt.tick_params(left = False)

        
        #plot average subject MAE
        fig = plt.figure(7, figsize = (x_mae_avg, y_mae_avg), dpi = dpi)
        barWidth = 0.01
        a = range(1, 2)
        b = np.mean(mae_all_subj[0:num_subjects])
        c = np.std(mae_all_subj[0:num_subjects])

        
        br1 = np.arange(len(a))
        br2 = [x + barWidth for x in br1]
        xpos = [br1, br2]
        plt.bar(xpos[z], b, width = barWidth, label = meas_type, color = color_bar, capsize=6)  
        plt.errorbar(xpos[z], b, yerr = c, color = "black", capsize=6)
        plt.xticks([r + barWidth for r in range(len(a))], a)
        

        ax = fig.gca()
        ax.tick_params(axis="y",direction="in", pad= pad_mae_avg)
        ax.tick_params(axis='both', which='major', labelsize= ax_label_size)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_visible(False)
        
        plt.yticks(fontname = font_all) 
        plt.xticks(fontname = font_all) 
        plt.tick_params(bottom = False)
        plt.tick_params(left = False)



    #save table of errors as excel 
    if save_flag == 1:
        table_data.to_excel(experiment_name + '.xlsx')

    
    return table_data, me, mae, sd, rmse, ccc, index, diffs_store, num_pulses, rmse_store, mae_store, me_store, sd_store, mae_all_subj
    
    

     