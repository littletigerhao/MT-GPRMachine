#!/usr/bin/python3
# -*- coding: utf-8 -*- 
#===============================================================================
# Date      : Dec/01/2020
# Author    : 
# Contact   : 
# Annotation: This file is the entrance of Lorentz.
#===============================================================================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import shutil
import numpy as np
import gpflow as gpf
import tensorflow as tf
from DataProcesser import DataProcesser
from MultiGPRModeller import TrainingPredicting
from ResultEvaluator import PerformanceEvaluation
from ResultSaver import SaveResults
#===============================================================================
#===============================================================================
if __name__ == '__main__':
    #===========================================================================
    # --------------------- Packages Version Information ----------------------#
    #===========================================================================
    print('==='*25)
    print('This program is running with the use of following packages:\n')
    print('      *** Tensorflow, version:', tf.__version__, '***')
    print('      *** GPFlow, version:', gpf.__version__, '***')
    print('      *** Numpy, version:', np.__version__, '***\n')
    #===========================================================================
    # ------------------------- Define parameters -----------------------------#
    #===========================================================================
    target = 'OceanT_1'
    n_start = 0
    n_train = 50
    n_test = 30
    noise_level = 0
    n_task = 5
    n_iter = 80
    k_low_list = [8.5, 8.9, 23.5, 20.5, 1.8, 0.8, 0.8, 1.1, 1.1, 25.5,
                  0.01, 0.001, 0.001, 0.05, 0.0004, 0.0005, 0.5, 0.5, 0.5, 0.8,
                  12.5, 4.8, 3.1, 0.5, 0.5, 1.5, 0.5, 0.5, 0.0006, 0.0005,]
    k_up_list = [9.85, 9.9, 25.5, 22.8, 2.8, 1.3, 1.3, 2.3, 2.3, 30.7,
                 0.03, 0.003, 0.003, 0.08, 0.0005, 0.0008, 0.1, 0.1, 0.1, 1.0,
                 15.85, 5.9, 4.2, 0.7, 0.7, 1.8, 0.9, 0.7, 0.0008, 0.0008]
    #===========================================================================
    # -------------------------- Initialization -------------------------------#
    #===========================================================================
    DP = DataProcesser(target)
    DP.load_data(n_start, n_train, n_test, noise_level)
    current_path = os.getcwd().split('/ProgramCodes')[0]
    result_dir = current_path + '/ResultFiles/' + target
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir)
    #===========================================================================
    # -------------------------Train and Prediction ---------------------------#
    #===========================================================================
    print('==='*25)
    print('Now running for %s, noise: %s ...'%(target, str(noise_level)))
    #---------------------------------------------------------------------------
    X_train = DP.X_train
    Y_train = DP.Y_train
    Y_test = DP.Y_test
    MU_list, VAR_list, mean_MU, mean_VAR = TrainingPredicting(X_train,
                                                              Y_train,
                                                              n_test,
                                                              n_task,
                                                              n_iter,
                                                              k_low_list,
                                                              k_up_list)
    #---------------------------------------------------------------------------
    scaler = DP.scaler
    t_indx = DP.t_indx
    scale_min = scaler.data_min_[t_indx]
    scale_max = scaler.data_max_[t_indx]
    Y_train = Y_train*(scale_max-scale_min)+scale_min
    Y_test = Y_test*(scale_max-scale_min)+scale_min
    Y_pred = mean_MU*(scale_max-scale_min)+scale_min
    #---------------------------------------------------------------------------
    print('>> Evaluation ...')
    performance = PerformanceEvaluation(Y_test.reshape(-1), Y_pred)
    #---------------------------------------------------------------------------
    print('>> Prediction done! Write to files ...')
    SaveResults(Y_train, Y_test, Y_pred, performance, result_dir, target, n_train, n_test, MU_list, mean_VAR)
    print('==='*25)
#===============================================================================
#===============================================================================
