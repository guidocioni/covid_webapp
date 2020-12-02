import scipy
import statsmodels.api as sm
import warnings
import numpy as np
import pandas as pd

'''Module to estimate the value of Rt according to the method of
Arroyo Marioli, Francisco and Bullano, Francisco and Kucinskas,
Simas and Rondón-Moreno, Carlos,
Tracking R of COVID-19: A New Real-Time Estimation Using the Kalman
Filter (May 10, 2020). http://dx.doi.org/10.2139/ssrn.3581633
'''


def prepare_data(df, total_cases_var='total_cases', new_cases_var='new_cases',
                 time_var='date', location_var='location'):
    '''Add growth rate with different days delay to the dataset'''
    min_cases = 100
    # Values of (1 / gamma) used in constructing
    days_infectious_list = [5, 6, 7, 8, 9, 10]
    # time series of infected individual

    # Only consider days after a minimum
    # number of total cases has been reached
    mask = (df[total_cases_var] >= min_cases)
    df = df.loc[mask, ]

    # Sort by date
    df.sort_values(by=[location_var, time_var], ascending=True,
                   inplace=True)

    # Construct number of infected
    for days_infectious in days_infectious_list:
        df['infected_{}'.format(days_infectious)] = np.nan
        for country in df[location_var].unique():
            mask = df[location_var] == country
            df_country = df.loc[mask, ].copy().reset_index()
            T = df_country.shape[0]

            # Initialize number of infected
            infected = np.zeros(T) * np.nan
            infected[0] = df_country[total_cases_var][0]

            # Main loop
            for tt in range(1, T):
                gamma = 1 / float(days_infectious)

                # Calculate number of infected recursively;
                # In the JH CSSE dataset, there are some
                # data problems whereby new cases are occasionally
                # reported to be negative; in these case, take zero
                # when constructing time series for # of invected,
                # and then change values to NaN's later on
                infected[tt] = ((1 - gamma) * infected[tt - 1]
                                + np.maximum(df_country[new_cases_var][tt], 0.0))
            df.loc[mask, 'infected_{}'.format(days_infectious)] = infected

    # In the original JH CSSE dataset, there are
    # some inconsistencies in the data
    # Replace with NaN's in these cases
    mask = df[new_cases_var] < 0
    df.loc[mask, new_cases_var] = np.nan
    print('Inconsistent observations in new_cases in JH CSSE dataset: {:}'.format(
        mask.sum()))
    for days_infectious in days_infectious_list:
        df.loc[mask, 'infected_{}'.format(days_infectious)] = np.nan

    # Calculate growth rate of infected
    for days_infectious in days_infectious_list:
        df['gr_infected_{}'.format(days_infectious)] = ((df['infected_{}'.format(days_infectious)]
                                                         / df.groupby(location_var).shift(1)['infected_{}'.format(days_infectious)]) - 1)
        mask = df.groupby(location_var).shift(
            1)['infected_{}'.format(days_infectious)] == 0.0
        df.loc[mask, 'gr_infected_{}'.format(days_infectious)] = np.nan

    # Deal with potential consecutive zeros in the number of infected
    for days_infectious in days_infectious_list:
        mask = (df['infected_{}'.format(days_infectious)] == 0.0) & (
            df.groupby(location_var).shift(1)['infected_{}'.format(days_infectious)] == 0.0)
        df.loc[mask, 'gr_infected_{}'.format(
            days_infectious)] = - (1 / days_infectious)
        if mask.sum() > 0:
            print('     Number of observations with zero infected (with {} infectious days) over two consecutive days: {:}'.format(
                days_infectious, mask.sum()))

    # Set to NaN observations with very small
    # number of cases but very high growth rates
    # to avoid these observations acting as
    # large outliers
    for days_infectious in days_infectious_list:
        gamma = 1 / float(days_infectious)
        mask = (df[new_cases_var] <= 25) & (df['gr_infected_{}'.format(
            days_infectious)] >= gamma * (5 - 1))  # Implicit upper bound on R
        df.loc[mask, ['infected_{}'.format(days_infectious),
                      'gr_infected_{}'.format(days_infectious)]] = np.nan

    # Set to NaN observations implausibly
    # high growth rates that are likely
    # due to data issues
    for days_infectious in days_infectious_list:
        gamma = 1 / float(days_infectious)
        mask = (df['gr_infected_{}'.format(days_infectious)] >= gamma * (10 - 1))
        df.loc[mask, ['infected_{}'.format(days_infectious),
                      'gr_infected_{}'.format(days_infectious)]] = np.nan

    # Remove initial NaN values for growth rates
    for country in df[location_var].unique():
        mask = df[location_var] == country
        T = df.loc[mask, ].shape[0]
        df.loc[mask, 'days_since_min_cases'] = range(T)
    mask = df['days_since_min_cases'] >= 1
    df = df.loc[mask, ]
    del df['days_since_min_cases']

    return df


def estimate_R(y, gamma, n_start_values_grid = 0, maxiter = 200):
    """Estimate basic reproduction number using
    Kalman filtering techniques

    Args:
        y (np array): Time series of growth rate in infections
        gamma (double): Rate of recoveries (gamma)
        n_start_values_grid (int, optional): Number of starting values used in the optimization;
            the effective number of starting values is (n_start_values_grid ** 2)
        maxiter (int, optional): Maximum number of iterations

    Returns:
        dict: Dictionary containing the results
          R (np array): Estimated series for R
          se_R (np array): Estimated standard error for R
          flag (int): Optimization flag (0 if successful)
          sigma2_irregular (float): Estimated variance of the irregular component
          sigma2_level (float): Estimated variance of the level component
          gamma (float): Value of gamma used in the estimation

    """
    assert isinstance(n_start_values_grid, int), \
      "n_start_values_grid must be an integer"

    assert isinstance(maxiter, int), \
      "maxiter must be an integer"

    assert n_start_values_grid >= 0 and maxiter > 0, \
      "n_start_values_grid and max_iter must be positive"

    assert isinstance(y, np.ndarray), \
      "y must be a numpy array"

    assert y.ndim == 1, \
      "y must be a vector"

    # Setup model instance
    mod_ll = sm.tsa.UnobservedComponents(y, 'local level')

    # Estimate model
    if n_start_values_grid > 0:
        # If requested, use multiple starting
        # values for more robust optimization results
        start_vals_grid = np.linspace(0.01, 2.0, n_start_values_grid) * pd.Series(y).var()
        opt_res = []
        for start_val_1 in start_vals_grid:
            for start_val_2 in start_vals_grid:
                res_ll = mod_ll.fit(start_params=np.array([start_val_1, start_val_2]),
                                    disp=False, maxiter=maxiter)
                opt_res.append({'obj_value': res_ll.mle_retvals['fopt'],
                                'start_val_1': start_val_1,
                                'start_val_2': start_val_2,
                                'flag': res_ll.mle_retvals['warnflag']})
        # The optimizer minimizes the negative of
        # the likelihood, so find the minimum value
        opt_res = pd.DataFrame(opt_res)
        opt_res.sort_values(by='obj_value', ascending=True, inplace=True)
        res_ll = mod_ll.fit(start_params = np.array([opt_res['start_val_1'][0], 
                                                     opt_res['start_val_2'][0]]),
                            maxiter=maxiter, disp=False)
    else:
        res_ll = mod_ll.fit(maxiter=maxiter, disp=False)
    R = 1 + 1 / (gamma) * res_ll.smoothed_state[0]
    se_R = (1 / gamma * (res_ll.smoothed_state_cov[0] ** 0.5))[0]
    return {'R': R,
            'se_R': se_R,
            'flag': res_ll.mle_retvals['warnflag'],
            'sigma2_irregular': res_ll.params[0],
            'sigma2_level': res_ll.params[1],
            'signal_to_noise': res_ll.params[1] / res_ll.params[0],
            'gamma': gamma}


def compute_r(df, total_cases_var='total_cases', new_cases_var='new_cases',
              time_var='date', location_var='location', days_infectious=7,
              min_T=20, gamma=1 / 7.0, min_signal_to_noise=1e-3,
              max_signal_to_noise=1e2):

    # Impose minimum time-series observations
    df_temp = df.groupby(location_var).count()['gr_infected_{}'.format(days_infectious)].reset_index()
    df_temp.rename(columns = {'gr_infected_{}'.format(days_infectious): 'no_obs'},
                   inplace = True)
    df = pd.merge(df, df_temp, how = 'left')
    mask = df['no_obs'] >= min_T
    df = df.loc[mask, ]

    ################
    ## Estimate R ##
    ################

    df['R'] = np.nan
    df['se_R'] = np.nan

    df_optim_res = []

    with warnings.catch_warnings():
      # Ignore warnings from statsmodels
      # Instead, check later
      warnings.filterwarnings("ignore", message = "Maximum Likelihood optimization failed to converge. Check mle_retvals")
      for country in df[location_var].unique():
          mask = df[location_var] == country
          df_temp = df.loc[mask, ].copy()
          y = df_temp['gr_infected_{}'.format(days_infectious)].values
          res = estimate_R(y, gamma = gamma)
          df.loc[mask, 'R'] = res['R']
          df.loc[mask, 'se_R'] = res['se_R']
          df_optim_res.append({location_var: country,
                               'flag': res['flag'],
                               'sigma2_irregular': res['sigma2_irregular'],
                               'sigma2_level': res['sigma2_level'],
                               'signal_to_noise': res['signal_to_noise']})
    df_optim_res = pd.DataFrame(df_optim_res)

    # Merge in optimization results
    df = pd.merge(df, df_optim_res, how = 'left')

    ###################################
    ## Filter out unreliable results ##
    ###################################
    # Unsuccessful optimization
    mask = df['flag'] != 0
    df = df.loc[~mask, ]
    # Filter out implausible signal-to-noise ratios
    mask = (df['signal_to_noise'] <= min_signal_to_noise) | (df['signal_to_noise'] >= max_signal_to_noise)
    df = df.loc[~mask, ]

    ####################
    ## Export results ##
    ####################
    df = df[[location_var, time_var, 'R', 'se_R']].copy()
    df.reset_index(inplace = True)
    del df['index']
    df['days_infectious'] = 1 / gamma

    # Calculate confidence intervals
    alpha = [0.05, 0.35]
    names = ['95', '65']
    for aa, name in zip(alpha, names):
        t_crit = scipy.stats.norm.ppf(1 - aa / 2)
        df['ci_{}_u'.format(name)] = df['R'] + t_crit * df['se_R']
        df['ci_{}_l'.format(name)] = df['R'] - t_crit * df['se_R']

    return df


def process_compute_rt(df, total_cases_var='total_cases', new_cases_var='new_cases',
              time_var='date', location_var='location'):
    '''Process all data to compute Rt'''
    growth_rate = prepare_data(df, total_cases_var=total_cases_var, new_cases_var=new_cases_var,
              time_var=time_var, location_var=location_var)
    final = compute_r(growth_rate, total_cases_var=total_cases_var, new_cases_var=new_cases_var,
              time_var=time_var, location_var=location_var)

    out = pd.merge(df, final,
                   left_on=[time_var,location_var], 
                   right_on=[time_var,location_var])

    return out
