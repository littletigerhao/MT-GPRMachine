#!/usr/bin/python  
# -*- coding: utf-8 -*- 
#===============================================================================
# Date      : November/12/2020
# Author    : 
# Contact   : 
# Annotation: This file is for saving prediction results.
#===============================================================================
import numpy as np
#===============================================================================
def SaveResults(Y_train, Y_test, Y_pred, performance, result_dir, target, n_train, n_test, MU_list, mean_VAR):
    #===========================================================================
    YTr = list(Y_train)
    YTe = list(Y_test)
    YPr = list(Y_pred)
    mae = performance[0]
    rmse = performance[1]
    pcc = performance[2]
    result_name = result_dir+'/'+target+'_'+str(mae)+'_'+str(rmse)+'_'+str(pcc)
    result_txt = open(result_name + '.txt', 'w')
    result_var = open(result_name + '.var', 'w')
    for y in YTr:
        result_txt.write(str(y[0]) + '\t' + str(y[0]) + '\n')
    for y, ypr in zip(YTe, YPr):
        result_txt.write(str(y) + '\t' + str(ypr) + '\n')
    result_txt.close()
    for var in mean_VAR:
        result_var.write(str(np.sqrt(var)) + '\n')
    result_var.close()
#===============================================================================
#===============================================================================
