
import os
import numpy as np
from scipy.optimize import curve_fit
import math
from tqdm import tqdm
from scipy.optimize import OptimizeWarning, minimize
import warnings 
import rule_utils

class ParamLearner:
    """ Class to learn the parameters for each rule for the given dataset by fitting different types of curves and selecting the one with the smallest mse"""
    def __init__(self, dataset_name, rel_mapping, node_mapping, plot_flag=False, figure_path='figures', multi_flag=False, single_flag=True, learn_flag=False,
                 rule_unseen_neg=0, lmbda_reg=0, datapoint_threshold_multi=20, static_flag=False, max_time_window=50):
        self.dataset_name = dataset_name

        self.rel_mapping_dict = rel_mapping # the mapping of the relations to their string representation
        self.node_mapping_dict = node_mapping # the mapping of the nodes to their string representation

        self.learn_data = {} # dataset to use to fit the curves
        self.mse_dict = {}
        
        self.multi_flag = multi_flag
        self.single_flag = single_flag
        self.datapoint_threshold_multi = datapoint_threshold_multi

        self.method = rule_utils.score_single

        self.max_time_window = max_time_window # the maximum time window to take into account when computing the proporting of available timesteps
        self.lmbda_reg = lmbda_reg
        self.rule_unseen_neg = rule_unseen_neg # the p value for modifying the confidence, conf_new = conf/(conf+rule_unseen_neg)\

        self.learn_flag = learn_flag
        self.static_flag = static_flag

        self.fct = rule_utils.score_powtwo
        self.bounds=([0,0.001,0], [1000, 1, 3.]) # Constrain the optimization to the region of 0 <= lmbda <= 1, 0.001 <= alpha <= 1 and 0 <= phi <= 1:
        self.multi_bounds = np.asarray([(None,None),(None,None),(0,1.001)], dtype=np.float64)

        self.xtol = 0.1 # Relative error desired in the approximate solution. relative error between two consecutive iterates, 
        self.ftol = 0.1  #Relative error desired in the sum of squares.
        self.ftol_min = 0.1

        # the learned params. key: rule_id (e.g. body_rel, head_rel), value: entries for [lmbda, alpha, phi, rho, kappa, gamma, fuction_type]
        # for c-rules the key is (relh, ch, relb, cb) ... so its a different order with respect to head and body 
        self.params_learned = {} 
        
        self.plot_flag = plot_flag # whether to plot the fitted curves      
        self.figure_path = os.path.join(figure_path, dataset_name) # where to save the figures
        if not os.path.exists(self.figure_path):
            os.makedirs(self.figure_path)        
        
        self.no_rec =[]

    def data_prep(self, data_rul):
        """ prepare the data for the curve fitting; 
        create one X array, that contains one hot encoding of the time distances
        create one y array, that contains the kappa_hat values
        :param data_rul: np.array, the data for the rule to be fitted; can contain multiple lines for the same rule, and also multiple lines for the same time distances
        :return: np.array, np.array, the X values (shape: (number of samples for this rule,)), the y values (shape: (number of samples for this rule,))
        """
        weights = []
        sigmas = []
        unique_x = []
        unique_y = []
        counts = []
        positives = []
        num_occ_total  = 0
        original_betas = []
        original_dists = []
        original_weights = []
        p_weights =[]
        pfactors = [] # how high was the denominator befor pvalue and after pvalue. i.e. how does the p value for this dist influce the confidence.
        nonaggregated_package_dict, aggregated_vals_dict, p_factors_dict = {}, {}, {}
        for dist in data_rul:

            if self.multi_flag == False:
                beta_hat = data_rul[dist][0]/(data_rul[dist][1]+self.rule_unseen_neg)
                original_betas.append(data_rul[dist][0]/data_rul[dist][1]) #before laplace smooting or adding numbers
                original_weights.append(data_rul[dist][1])
                p_weights.append(data_rul[dist][1]+self.rule_unseen_neg)
                original_dists.append(dist)
                pfactor = data_rul[dist][1]/(data_rul[dist][1]+self.rule_unseen_neg)
                pfactors.append(pfactor)
            else:
                beta_hat = data_rul[dist][0]/(data_rul[dist][1])
                original_weights.append(data_rul[dist][1])
                pfactor = data_rul[dist][1]/(data_rul[dist][1]+self.rule_unseen_neg)
                pfactors.append(pfactor)

            num_occurences = data_rul[dist][1]
            num_occ_total += num_occurences

            sigmas.append(1/np.sqrt(num_occurences))  # needed for curve fitting
            weights.append(num_occurences)            
            unique_x.append(dist)
            unique_y.append(beta_hat)
            counts.append(num_occurences)
            positives.append(data_rul[dist][0])


        if self.multi_flag: # some preprocessing of the  multi learn_data is needed
            nonaggregated_package_dict, aggregated_vals_dict, p_factors_dict = rule_utils.make_packages(unique_x, unique_y, weights, positives, pfactors, self.max_time_window, self.rule_unseen_neg)
            ## 1) compute the single parameters based on the aggregated values. this leads to the same params as in single setting.
            dists_agg = np.array(list(aggregated_vals_dict.keys()))
            betas_agg = np.array(list(aggregated_vals_dict.values()))[:,0]
            weights_agg = np.array(list(aggregated_vals_dict.values()))[:,1]
            # sort dists_agg by increasting order and sort betas_agg (aggregated betas) and weights_agg (aggregated weights) in the same order
            # this is the same as the learn data for in signle setting
            sorted_indices = np.argsort(dists_agg)
            unique_x = dists_agg[sorted_indices]
            unique_y = betas_agg[sorted_indices]        
            weights = weights_agg[sorted_indices]

            sigmas = np.zeros(len(weights))
            for i in range(len(weights)):
                sigmas[i] = 1/np.sqrt(weights[i])

        return unique_x, np.array(unique_y),  np.array(sigmas), np.array(weights), np.array(original_betas), np.array(original_dists),  nonaggregated_package_dict, p_factors_dict
    
    def sanity_check(self, lmbda_init, phi_init, alpha_init, betas, weights, dists):
        """ compute static confidence based on weighted mean, and check whether the static confidence is better than the dynamic score
        better means lower mse. if that is the case, set the static params as the new params
        """
        # compute static confidence based on weighted mean.
        static_confidence = np.sum(betas*weights)/np.sum(weights)
        alpha_static = static_confidence
        lmbda_static = 0
        phi_static = 0

        y_pred_static = np.zeros(len(dists))
        y_pred_dyn = np.zeros(len(dists))
        i = 0
        for d in dists:
            y_pred_static[i] = alpha_static*self.fct(d, lmbda_static,phi_static)
            y_pred_dyn[i] = alpha_init*self.fct(d, lmbda_init,phi_init)
            i+=1

        mse_static = np.sum((y_pred_static - betas)**2 * weights) / np.sum(weights)
        mse_dyn = np.sum((y_pred_dyn - betas)**2 * weights) / np.sum(weights)

        if mse_static < mse_dyn:
            alpha_init = alpha_static
            lmbda_init = lmbda_static
            phi_init = phi_static
                                    
        return lmbda_init, alpha_init, phi_init
    
    def fit_and_compute_error(self, X, y, fct, sigmas, p0):
        """ fit the curve using the given function and compute the mean squared error
        :param X: np.array, the x values of the ground truth data, i.e. delta_ts
        :param y: np.array, the y values of the ground truth data, i.e. beta_hat
        :param fct: function, the function that should be used to fit the curve
        :param sigmas: the weights for the curve fitting
        :param p0: np.array, the initial values for the parameters of the curve, lmbda, alpha, phi
        :return: float, np.array, the mean squared error, the optimal parameters
        """
        try:              
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)  
                popt, _= curve_fit(fct, X, y, bounds=self.bounds, sigma=sigmas, ftol=self.ftol, xtol=self.xtol, p0=p0) # fit curve using the given function and take into account the param bounds

        except: # e.g. if  no params are found that fit the curve
            popt = p0 

        return popt
    
    def compute_params_powtwo_single(self, dists, betas, weights):
        """ compute the parameters for the powtwo function based on the given data
        :param dists: np.array, the distances of the data points
        :param betas: np.array, the beta values of the data points
        :param weights: np.array, the weights of the data points
        :return alpha_init, lmbda_init, phi_init: the parameters for the powtwo function
        """
        lmbda_bounds = [self.bounds[0][0], self.bounds[1][0]]
        alpha_bounds = [self.bounds[0][1], self.bounds[1][1]]
        phi_bounds = [self.bounds[0][2], self.bounds[1][2]]

        lmbda_init = 1
        alpha_init = np.max([alpha_bounds[0], betas[0]]) # alpha is the first beta value #  min value for alpha should be higher than lower bounds
        phi_init = 0
        last_vals = np.min([len(betas)-1, 10]) # the oldest 10 beta values or if less than 10, all beta values

        lmbdas = []
        weights_used = []
        if len(betas) > 1:
            offset = np.sum(betas[last_vals:]*weights[last_vals:])/np.sum(weights[last_vals:]) # the weighted mean across the last_vals oldest beta values

            phi_init = offset/((alpha_init-offset)+0.000000000000001) # to avoid division by zero
            phi_init = np.min([phi_bounds[1], phi_init]) #  max value for phi should not be higher than bounds
            phi_init = np.max([phi_bounds[0], phi_init]) #  min value for phi should be higher than lower bounds
            for index in range(1,np.min([len(betas),5])):    # compute lambda values for the first 3 beta values (delta_t >1) and take the mean to get an approximation
                y = betas[index]
                dist = dists[index]
                if dist == 1:
                    print(';')
                lmbda = - math.log2(np.max([(y*(1+phi_init)+0.000000000000001 )/(alpha_init+0.000000000000001) - phi_init,0.000000000000001])) / (dist-1)
                lmbda = np.min([lmbda_bounds[1], lmbda]) #  max value for lambda is 1
                lmbda = np.max([lmbda_bounds[0], lmbda]) # min value for lambda is 0
                weights_used.append(weights[index]) 
                lmbdas.append(weights[index]*lmbda)
            lmbda_init = np.sum(lmbdas)/np.sum(weights_used) # take the weighted mean of the lambda values. the weights are the number of occurences of the beta values
        
            if lmbda_init <0.3: # if the decay is rather flat: do another iteration of setting the params for finetuning, only taking into account the oldest third of timesteps
                if len(betas) > 2:

                    last_vals = int(len(betas)/3)  
                    offset = np.sum(betas[-last_vals:]*weights[-last_vals:])/np.sum(weights[-last_vals:])# the weighted mean across the last_vals oldest beta values

                    lmbdas = []
                    weights_used = []
                    if len(betas) > 1:
                        
                        for index in range(1,np.min([len(betas)-1,5])):    # compute lambda values for the first 3 beta values (delta_t >1) and take the mean to get an approximation
                            y = betas[index]
                            dist = dists[index]
                            lmbda = - math.log2(np.max([(y*(1+phi_init)+0.000000000000001 )/(alpha_init+0.000000000000001) - phi_init,0.000000000000001])) / (dist-1)
                            lmbda = np.min([lmbda_bounds[1], lmbda]) #  max value for lambda is 1
                            lmbda = np.max([lmbda_bounds[0], lmbda]) # min value for lambda is 0
                            weights_used.append(weights[index]) 
                            lmbdas.append(weights[index]*lmbda)
                        lmbda_init = np.sum(lmbdas)/np.sum(weights_used) # take the weighted mean of the lambda values. the weights are the number of occurences of the beta values
            if lmbda_init == 0:
                if np.mean(betas) < betas[0]:
                    lmbda_init = 0.000001

            lmbda_init, alpha_init, phi_init = self.sanity_check(lmbda_init, phi_init, alpha_init, betas, weights, dists)
        return lmbda_init, alpha_init, phi_init


    def weighted_ridge_loss(self, params, X, y, lmbda_reg, sample_weights, static=False):
        """Loss function for weighted Ridge Regression."""

        max_ampl = params[-1]

        if static:
            yhat = (X @ [params[0], 0])
        else:
            yhat = (X @ params[0:2])
        yhat = np.clip(yhat, -max_ampl, max_ampl)  # Clip predictions to avoid extreme values

        residuals = yhat - y # to be minimized: residulas =  (alpha* exp(-lmbda *x[0])) + rho*x[1] + beta) - y
        
        weighted_residuals = sample_weights * residuals**2  # Apply weights to residuals; weights are number of occurences
        penalty = lmbda_reg * np.sum(params**2)  # L2 regularization term
        return np.sum(weighted_residuals) + penalty


    def compute_initial_params(self, x, y, weights):
        """ compute initial params for the multi params; rho_init is the slope of the line between the first and last point, average across multiple "last points".
        This is used as input as initial value for the minimize fct.
        """
        rho_init = 0
        kappa_init = 0
        weights_total =0
        if len(x) > 1:
            for i in range(len(x)):
                if i > 0:
                    rho_init  += weights[i]*(y[i] - y[0])/(x[i] - x[0]) # slope of the line
                    weights_total += weights[i]
            rho_init = np.sum(rho_init)/weights_total # weighted mean of the slopes 
            # weighted mean of the x values
            kappa_init = -np.sum(np.array(x) * np.array(weights)) / np.sum(weights) 
        return rho_init, kappa_init

    def compute_params_powtwo_multi(self, lmbda, alpha, phi, nonaggregated_package_dict, p_factors_dict, dists_agg, weights_agg, static_flag_rule):
        """ compute the multi params for the given rule. the single params are already computed. the multi params are computed based on 
        the deviaton from prediction to ground truth for each proportion."""
        rho_init = 0
        kappa_init = 0
        gamma_init = 1   
        y = []
        x = []
        weights = []

        params = [lmbda, alpha, phi, rho_init, kappa_init, gamma_init]

        if min(dists_agg) <= self.max_time_window:
            ## compute the multi parameters
            # if the number of data points for this rule is higher than 20, compute multi params; else, set them to 0.
            if np.sum(weights_agg) > self.datapoint_threshold_multi:
                x = []
                y = []
                weights = []
                sigmas = []
                # prepare the data for computing multi params for this rule
                # construct the x and y values for the least squares fit
                # concatenate the values across all dists and across all proportions

                for dist in dists_agg:
                    # prediction = alpha*self.fct(dist, lmbda, phi) # the prediction from recency; it is the same for all proportions in with the same min distance 
                    pfactor = p_factors_dict[dist]# this is factor by which the p-value reduced the beta for fittting recency the same for all proportions with the same min distance                    
                    index = 0
                    prediction = alpha*self.fct(dist, lmbda, phi) # the prediction from recency; it is the same for all proportions in with the same min distance 
                    for proportion in nonaggregated_package_dict[dist]['proportion']: # append the values from learn data for each propotion

                        beta_dist_prop = nonaggregated_package_dict[dist]['beta'][index]*pfactor # confidence value for this distance and propotion, corrected by pfactor
                        weight=nonaggregated_package_dict[dist]['weight'][index] 

                        pred_diff = beta_dist_prop - prediction # deviation from predicted data point for this specific point
                        x.append([proportion, 1/float(dist)])
                        y.append(pred_diff)
                        weights.append(weight)
                        sigmas.append(1/np.sqrt(weight))   

                        index +=1


                # compute the multi params; least squares fit

                A = np.asarray(x, dtype=np.float64) # np.vstack([x, np.ones(len(x))]).T # rho*x + kappa  = [rho;kappa]*A
                # Regularization Parameter (λ)
                lambda_ = self.lmbda_reg

                # initial_params = np.zeros(A.shape[1])  #init param values to zero
                max_ampl = np.max([abs(np.min(y)), np.max(y)]) # not used right now

                # init_rho, init_kappa = self.compute_initial_params(x, y, weights)
                init_rho, init_kappa = 0,0

                if static_flag_rule:
                    initial_params = np.array([init_rho, 0, max_ampl]) 
                else:
                    initial_params = np.array([init_rho, init_kappa, max_ampl]) # then gamma should also be learned. this is significantly slower.
                    
                y = np.asarray(y, dtype=np.float64)
                weights = np.asarray(weights, dtype=np.float64)
               
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", OptimizeWarning)  
                    result = minimize(self.weighted_ridge_loss, initial_params, args=(A, y, lambda_, weights, static_flag_rule), tol= self.ftol_min) # bounds=self.multi_bounds)
                

                rho_init, kappa_init, gamma_init= result.x

                params = [lmbda, alpha, phi, rho_init, kappa_init, gamma_init]

        return params, x, y, weights



    def learn_params(self, progressbar_percentage=0.01):
        """ learn the params for each rule based on the learn data. compute the params with some heuristic. if learn_flag=True, do a curve fit to further improve the computed params
        section 4 of our paper
        :param progressbar_percentage: float, pertentage of the progress bar to update; default: 0.01
        :return:   
        self.params_learned: dict, the learned params for each rule; key: rule_key (e.g. (body_rel, head_rel)), value: entries for [lmbda, alpha, phi, rho, kappa, gamma, function_type]
        0: placeholder for mse
        """
        number_of_rules = len(self.learn_data)
        total_iterations = number_of_rules
        counter=0
        counter_x0  =0
        increment = int(total_iterations*progressbar_percentage) if int(total_iterations*progressbar_percentage) >=1 else 1
        remaining = total_iterations

        with tqdm(total=total_iterations) as pbar:            
            counter = 0
            for rul in self.learn_data: # todo: double check whether first entry is body and second entry is head
                # Update progress bar
                counter += 1
                if counter % increment == 0:
                    remaining -= increment
                    pbar.update(increment)
                if remaining < increment:
                    pbar.update(remaining)
                
                data_rul = dict(sorted(self.learn_data[rul].items()))

                # 1) data prep
                X, y, sigmas, weights, original_betas, original_dists, nonaggregated_package_dict, p_factors_dict  = self.data_prep(data_rul) 
                if y.mean() == 0:
                    continue

                # 2) learning the params for the given data
                kappa, rho, gamma = 0, 0, 1
                static_confidence = np.sum(y*weights)/np.sum(weights)

                if X[0] >1 : # if the smallest time distance for this rule is not 1
                    if self.multi_flag == True and self.single_flag == False:
                        single_flag_rule = False
                        multi_flag_rule = True
                        static_flag_rule = self.static_flag
                    else:
                        static_flag_rule = True
                        single_flag_rule =False
                        multi_flag_rule = False
                    counter_x0 +=1
                else:
                    static_flag_rule = self.static_flag
                    multi_flag_rule= self.multi_flag
                    single_flag_rule = self.single_flag

                ## 2.1) compute single params (f_r, equation 12, section 4.2)
                if static_flag_rule == True: # if static flag is set, compute static params
                    static_confidence = np.sum(y*weights)/np.sum(weights)
                    alpha = static_confidence
                    lmbda, phi = 0, 0                    
 
                else:
                    if single_flag_rule == True: # if single flag is set, compute single params
                        lmbda, alpha, phi = self.compute_params_powtwo_single(X, y, weights) # first compute single params based on heuristic # equation (14)

                        ## 2.2) if learn_flag is set, learn the params with a curve_fit approach
                        if self.learn_flag:
                            X = np.array(X).astype(np.float64)
                            sigmas = sigmas.astype(np.float64)
                            opt_params = self.fit_and_compute_error(X, y, self.method, sigmas,  p0=np.array([lmbda, alpha, phi]))
                            lmbda, alpha, phi = opt_params[0], opt_params[1], opt_params[2] # take the optimized params
                    else:
                        lmbda, alpha, phi = 0,0,0

                ## 2.3) compute multi params (g_r, equation 13, section 4.2)
                if multi_flag_rule: # if multi flag is set, compute multi params
                    params, X_multi, y_multi, weights_multi = self.compute_params_powtwo_multi(lmbda, alpha, phi, nonaggregated_package_dict, p_factors_dict, X, weights, static_flag_rule) # equation (15)
                    lmbda, alpha, phi, rho, kappa, gamma = params

     
                self.params_learned[rul] = [lmbda, alpha, phi, rho, kappa, gamma]  # lambda, alpha, phi, rho, kappa, gamma

                # 3) plot the fitted curve - this is a bit slow, and creates many figures, one for each rule; do only if needed
                # if np.sum(weights_multi) > 10000 and np.abs(rho)> 0.1:
                if self.plot_flag:
                    rule_utils.plot_curves(rul, X, y, (self.params_learned[rul]), self.method, weights, 0, self.params_learned[rul][-1],
                                self.node_mapping_dict, self.rel_mapping_dict, self.multi_flag, self.figure_path, learn=self.learn_flag, original_betas=original_betas, original_dists=original_dists)
                    if self.multi_flag == True:
                        # plot the fitted curve; two alternative plots
                        # rule_utils.plot_multi(self.params_learned[rul], nonaggregated_package_dict, p_factors_dict, rul, self.figure_path, self.node_mapping_dict, self.rel_mapping_dict, self.max_time_window)
                        # rule_utils.plot_multi2(self.params_learned[rul], nonaggregated_package_dict, p_factors_dict, rul, self.figure_path, self.node_mapping_dict, self.rel_mapping_dict, self.max_time_window)
                        rule_utils.plot_multi_freq(self.params_learned[rul], X_multi, y_multi, weights_multi, rul, self.figure_path, self.node_mapping_dict, self.rel_mapping_dict, self.max_time_window) #second plot
                        # rule_utils.plot_multi_freq(self.params_learned[rul], np.array(X_multi)[:,1], y_multi, weights, rul, self.figure_path, self.node_mapping_dict, self.rel_mapping_dict, self.max_time_window) #second plot
            
        # print('number of rules with x0 > 1: ', counter_x0)
        return self.params_learned, 0
    
 


    

