#!/usr/bin/python3
# -*- coding: utf-8 -*- 
#===============================================================================
# Date      : Dec/01/2020
# Author    : 
# Contact   : 
# Annotation: This file is the core of MultiTask Gaussian Process Regression.
#===============================================================================
import numpy as np
import gpflow as gpf
from gpflow.ci_utils import ci_niter
#===============================================================================
#===============================================================================
class MultiGPR():
    #===========================================================================
    #===========================================================================
    def __init__(self, X_train, Y_train, n_iter, n_task):
        self.X_train = X_train                  # list of training X used in GPR
        self.Y_train = Y_train                  # list of training Y used in GPR
        self.n_iter = ci_niter(n_iter)
        self.n_task = n_task
    #===========================================================================
    #===========================================================================
    def set_init_kernel(self, lower, upper):
        np.random.seed()
        x_dim = np.shape(self.X_train[0])[1]
        lsv = np.random.uniform(lower, upper, x_dim)
        return x_dim, lsv
    #===========================================================================
    #===========================================================================
    def augment_xy(self):
        X_task = self.X_train[0]
        Y_task = self.Y_train[0]
        n_train = X_task.shape[0]
        mark = np.zeros((n_train, 1))
        X_augmented = np.hstack((X_task, mark))
        Y_augmented = np.hstack((Y_task, mark))
        for a in range(1, self.n_task):
            X_task = self.X_train[a]
            Y_task = self.Y_train[a]
            n_train = X_task.shape[0]
            mark = np.asarray([a] * n_train).reshape(n_train, 1)
            X_augmented = np.vstack((X_augmented, np.hstack((X_task, mark))))
            Y_augmented = np.vstack((Y_augmented, np.hstack((Y_task, mark))))
        return X_augmented, Y_augmented
    #===========================================================================
    #===========================================================================
    def build_model(self, k_low, k_up):
        output_dim = len(self.X_train)
        x_dim, lsv = self.set_init_kernel(k_low, k_up)
        dim_idx_list = [idx for idx in range(x_dim)]
        kernel_1 = gpf.kernels.Matern52(active_dims=dim_idx_list, lengthscales=lsv)
        kernel_2 = gpf.kernels.Matern32(active_dims=dim_idx_list, lengthscales=lsv)
        kernel_3 = gpf.kernels.RBF(active_dims=dim_idx_list, lengthscales=lsv)
        kernel_4 = gpf.kernels.White()
        base_kernel = kernel_1 + kernel_2 + kernel_3 + kernel_4
        coregion_kernel = gpf.kernels.Coregion(output_dim=output_dim,
                                               rank=output_dim,
                                               active_dims=[x_dim])
        kernel = base_kernel * coregion_kernel
        likelihood_list = [gpf.likelihoods.Gaussian() for _ in range(output_dim)]
        likelihood = gpf.likelihoods.SwitchedLikelihood(likelihood_list)
        X_augmented, Y_augmented = self.augment_xy()
        self.model = gpf.models.VGP((X_augmented, Y_augmented),
                                    kernel=kernel,
                                    likelihood=likelihood)
        self.coreg_kernel = coregion_kernel
        #print(self.model.trainable_parameters)
    #===========================================================================
    #===========================================================================
    def optimizing(self):
        gpf.optimizers.Scipy().minimize(self.model.training_loss,
                                        self.model.trainable_variables,
                                        options=dict(disp=False, maxiter=self.n_iter),
                                        method="L-BFGS-B")
    #===========================================================================
    #===========================================================================
    def predicting(self, X_test):
        self.mu = []
        self.var = []
        for t in range(self.n_task):
            x_test = X_test[t]
            n_test = x_test.shape[0]
            mark = np.asarray([t] * n_test).reshape((n_test, 1))
            X_augmented = np.hstack((x_test, mark))
            #=======================================================================
            mu, var = self.model.predict_f(X_augmented)
            self.mu.append(mu.numpy().reshape(-1))
            self.var.append(var.numpy().reshape(-1))
#===============================================================================
#===============================================================================
def TrainingPredicting(X_train, Y_train, n_test, n_task, n_iter, k_low_list, k_up_list):
    mu_list = []
    var_list = []
    n_train = X_train.shape[0]
    print('==='*25)
    for g in range(0, n_test):
        print('    >>> Training process is now running for group %d in total of %d ...'%(g+1, n_test))
        X_Train = []
        Y_Train = []
        X_Test = []
        for t in range(n_task):
            X_task = X_train[:n_train-g-t-1, :]
            X_Train.append(X_task)
            Y_task = Y_train[g+t+1:, :]
            Y_Train.append(Y_task)
            X_test = X_train[n_train-g-t-1:, :]
            X_Test.append(X_test)
        k_low = k_low_list[g]
        k_up = k_up_list[g]
        gpr_model = MultiGPR(X_Train, Y_Train, n_iter, n_task)
        gpr_model.build_model(k_low, k_up)
        gpr_model.optimizing()
        gpr_model.predicting(X_Test)
        for t in range(n_task):
            if len(gpr_model.mu[t]) <= n_test:
                mu_list.append(gpr_model.mu[t])
                var_list.append(gpr_model.var[t])
            else:
                mu_list.append(gpr_model.mu[t][:n_test])
                var_list.append(gpr_model.var[t][:n_test])
    mean_MU, mean_VAR = calculate_mean(n_test, mu_list, var_list)
    print('==='*25)
    return mu_list, var_list, mean_MU, mean_VAR
#===============================================================================
#===============================================================================
def calculate_mean(n_test, mu_list, var_list):
    MU = []
    VAR = []
    for i in range(0, len(mu_list)):
        current_mu = mu_list[i]
        current_var = var_list[i]
        n_mu_var = len(current_mu)
        if n_mu_var < n_test:
            expent_temp = np.asarray(['None' for _ in range(n_test-n_mu_var)])
            MU.append(np.hstack((current_mu, expent_temp)))
            VAR.append(np.hstack((current_var, expent_temp)))
        else:
            MU.append(current_mu[:n_test])
            VAR.append(current_var[:n_test])
    MU = np.asarray(MU)
    VAR = np.asarray(VAR)
    mean_MU = []
    mean_VAR = []
    for t in range(n_test):
        p_mu = list(MU[:, t])
        while 'None' in p_mu:
            p_mu.remove('None')
        p_var = list(VAR[:, t])
        while  'None' in p_var:
            p_var.remove('None')
        mean_MU.append(np.average(np.asarray(p_mu, dtype=float)))
        mean_VAR.append(np.average(np.asarray(p_var, dtype=float)))
    return np.asarray(mean_MU), np.asarray(mean_VAR)
#===============================================================================
#===============================================================================
