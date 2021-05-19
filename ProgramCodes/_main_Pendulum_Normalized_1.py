#!/usr/bin/python3
# -*- coding: utf-8 -*- 
#===============================================================================
# Date      : Dec/01/2020
# Author    : 
# Contact   : 
# Annotation: This file is the entrance of Pendulum.
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
    target = 'Pendulum_1'
    n_start = 0
    n_train = 50
    n_test = 30
    noise_level = 0.0
    n_task = 5
    n_iter = 80
    k_low_list = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    k_up_list = [k+0.1 for k in k_low_list]
    #===========================================================================
    # -------------------------- Initialization -------------------------------#
    #===========================================================================
    DP = DataProcesser(target)
    DP.load_data(n_start, n_train, n_test, noise_level)
    current_path = os.getcwd().split('/ProgramCodes')[0]
    result_dir = current_path + '/ResultFiles/' + target + '/Task_' + str(n_task)
    result_dir += '/NoiseLevel_' + str(noise_level)
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
    Y_pred = mean_MU
    #---------------------------------------------------------------------------
    print('>> Evaluation ...')
    performance = PerformanceEvaluation(Y_test.reshape(-1), Y_pred)
    #---------------------------------------------------------------------------
    print('>> Prediction done! Write to files ...')
    SaveResults(Y_train, Y_test, Y_pred, performance, result_dir, target, n_train, n_test, MU_list, mean_VAR)
    print('==='*25)
#===============================================================================
#===============================================================================
