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
    target = 'Plankton_1'
    n_start = 200
    n_train = 50
    n_test = 30
    noise_level = 0
    n_task = 5
    n_iter = 80
    k_low_list = [0.0008, 0.0008, 0.005, 0.05, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3,
                  0.3, 0.3, 0.8, 12.3, 13.5, 8.9, 3.0, 1.0, 0.5, 1.8,
                  40.0, 55.0, 80.9, 75.8, 50.8, 0.8, 0.7, 0.7, 0.7, 0.8]
    k_up_list = [0.0009, 0.0009, 0.008, 0.07, 0.55, 0.55, 0.55, 0.4, 0.4, 0.4,
                 0.4, 0.4, 0.9, 12.5, 13.6, 9.8, 3.5, 1.3, 0.9, 1.8,
                 45.0, 60.0, 88.9, 80.8, 55.8, 0.9, 0.9, 0.8, 0.9, 0.9]
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
