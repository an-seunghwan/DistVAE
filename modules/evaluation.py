#%%
import numpy as np
#%%
import statsmodels.api as sm
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier
)

from sklearn.metrics import f1_score

from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import wasserstein_distance

from sklearn import metrics
#%%
def merge_discrete(data, cont_dim):
    data = data[:, cont_dim:]
    cut_points = [0]
    for j in range(1, data.shape[1] + 1):
        if np.sum(data[:, cut_points[-1]:j]) == len(data):
            cut_points.append(j)
    return cut_points[1:]
#%%
def regression_eval(train, test, target):
    covariates = [x for x in train.columns if x not in [target]]
    
    result = []
    for name, regr in [
        ('linear', None), 
        # ('linear', LinearRegression(fit_intercept=False)), 
        ('RF', RandomForestRegressor(random_state=0)), 
        ('GradBoost', GradientBoostingRegressor(random_state=0))]:
        
        if name == 'linear':
            regr = sm.OLS(train[target], train[covariates]).fit()
        else:
            regr.fit(train[covariates], train[target])
        pred = regr.predict(test[covariates])
        
        mare = (test[target] - pred).abs()
        mare /= test[target].abs() + 1e-6
        mare = mare.mean()
        
        result.append((name, mare))
        print("[{}] MARE: {:.3f}".format(name, mare))
    return result
#%%
def classification_eval(train, test, target):
    covariates = [x for x in train.columns if not x.startswith(target)]
    target_ = [x for x in train.columns if x.startswith(target)]
    train_target = train[target_].idxmax(axis=1)
    test_target = test[target_].idxmax(axis=1).to_numpy()
    
    result = []
    for name, clf in [
        ('logistic', LogisticRegression(multi_class='ovr', fit_intercept=False, max_iter=1000)), 
        ('RF', RandomForestClassifier(random_state=0)), 
        ('GradBoost', GradientBoostingClassifier(random_state=0))]:
        
        clf.fit(train[covariates], train_target)
        pred = clf.predict(test[covariates])
        
        f1 = f1_score(test_target, pred, average='micro')
        
        result.append((name, f1))
        print("[{}] F1: {:.3f}".format(name, f1))
    return result
#%%
def goodness_of_fit(cont_dim, train, synthetic, cut_points1, cut_points2):
    Dn_list = []
    W1_list = []
    for j in range(cont_dim):
        xj = train[:, j]
        ecdf = ECDF(xj)
        ecdf_hat = ECDF(synthetic[:, j])

        Dn = np.abs(ecdf(xj) - ecdf_hat(xj)).max()
        W1 = wasserstein_distance(xj, synthetic[:, j])
        
        Dn_list.append(Dn)
        W1_list.append(W1)
    
    st1 = 0
    st2 = 0
    for j in range(len(cut_points1)):
        ed1 = cut_points1[j]
        ed2 = cut_points2[j]
        xj = train[:, cont_dim + st1 : cont_dim + ed1]
        ITS_xj = synthetic[:, cont_dim + st2 : cont_dim + ed2]
        ecdf = ECDF(xj.argmax(axis=1))
        ecdf_hat = ECDF(ITS_xj.argmax(axis=1))
        
        Dn = np.abs(ecdf(xj.argmax(axis=1)) - ecdf_hat(xj.argmax(axis=1))).max()
        W1 = wasserstein_distance(xj.argmax(axis=1), ITS_xj.argmax(axis=1))
        st1 = ed1
        st2 = ed2
        
        Dn_list.append(Dn)
        W1_list.append(W1)
    return Dn_list, W1_list
#%%
# def goodness_of_fit(config, train, synthetic):
#     Dn_list = []
#     W1_list = []
#     for j in range(config["CRPS_dim"]):
#         xj = train[:, j]
#         ecdf = ECDF(xj)
#         ecdf_hat = ECDF(synthetic[:, j])

#         Dn = np.abs(ecdf(xj) - ecdf_hat(xj)).max()
#         W1 = wasserstein_distance(xj, synthetic[:, j])
        
#         Dn_list.append(Dn)
#         W1_list.append(W1)
#     Dn = np.mean(Dn_list)
#     W1 = np.mean(W1_list)
#     return Dn, W1
#%%
def privacy_metrics(train, synthetic, data_percent=15):
    
    """
    Reference:
    [1] https://github.com/Team-TUD/CTAB-GAN/blob/main/model/eval/evaluation.py
    
    Returns privacy metrics
    
    Inputs:
    1) train -> real data
    2) synthetic -> corresponding synthetic data
    3) data_percent -> percentage of data to be sampled from real and synthetic datasets for computing privacy metrics
    Outputs:
    1) List containing the 5th percentile distance to closest record (DCR) between real and synthetic as well as within real and synthetic datasets
    along with 5th percentile of nearest neighbour distance ratio (NNDR) between real and synthetic as well as within real and synthetic datasets
    
    """
    
    # Sampling smaller sets of real and synthetic data to reduce the time complexity of the evaluation
    real_sampled = train.sample(n=int(len(train)*(.01*data_percent)), random_state=42).to_numpy()
    fake_sampled = synthetic.sample(n=int(len(synthetic)*(.01*data_percent)), random_state=42).to_numpy()

    # Computing pair-wise distances between real and synthetic 
    dist_rf = metrics.pairwise_distances(real_sampled, Y=fake_sampled, metric='minkowski', n_jobs=-1)
    # Computing pair-wise distances within real 
    dist_rr = metrics.pairwise_distances(real_sampled, Y=None, metric='minkowski', n_jobs=-1)
    # Computing pair-wise distances within synthetic
    dist_ff = metrics.pairwise_distances(fake_sampled, Y=None, metric='minkowski', n_jobs=-1) 
    
    # Removes distances of data points to themselves to avoid 0s within real and synthetic 
    rd_dist_rr = dist_rr[~np.eye(dist_rr.shape[0],dtype=bool)].reshape(dist_rr.shape[0],-1)
    rd_dist_ff = dist_ff[~np.eye(dist_ff.shape[0],dtype=bool)].reshape(dist_ff.shape[0],-1) 
    
    # Computing first and second smallest nearest neighbour distances between real and synthetic
    smallest_two_indexes_rf = [dist_rf[i].argsort()[:2] for i in range(len(dist_rf))]
    smallest_two_rf = [dist_rf[i][smallest_two_indexes_rf[i]] for i in range(len(dist_rf))]       
    # Computing first and second smallest nearest neighbour distances within real
    smallest_two_indexes_rr = [rd_dist_rr[i].argsort()[:2] for i in range(len(rd_dist_rr))]
    smallest_two_rr = [rd_dist_rr[i][smallest_two_indexes_rr[i]] for i in range(len(rd_dist_rr))]
    # Computing first and second smallest nearest neighbour distances within synthetic
    smallest_two_indexes_ff = [rd_dist_ff[i].argsort()[:2] for i in range(len(rd_dist_ff))]
    smallest_two_ff = [rd_dist_ff[i][smallest_two_indexes_ff[i]] for i in range(len(rd_dist_ff))]
    
    # Computing 5th percentiles for DCR and NNDR between and within real and synthetic datasets
    min_dist_rf = np.array([i[0] for i in smallest_two_rf])
    fifth_perc_rf = np.percentile(min_dist_rf,5)
    min_dist_rr = np.array([i[0] for i in smallest_two_rr])
    fifth_perc_rr = np.percentile(min_dist_rr,5)
    min_dist_ff = np.array([i[0] for i in smallest_two_ff])
    fifth_perc_ff = np.percentile(min_dist_ff,5)
    # nn_ratio_rf = np.array([i[0]/i[1] for i in smallest_two_rf])
    # nn_fifth_perc_rf = np.percentile(nn_ratio_rf,5)
    # nn_ratio_rr = np.array([i[0]/i[1] for i in smallest_two_rr])
    # nn_fifth_perc_rr = np.percentile(nn_ratio_rr,5)
    # nn_ratio_ff = np.array([i[0]/i[1] for i in smallest_two_ff])
    # nn_fifth_perc_ff = np.percentile(nn_ratio_ff,5)
    
    return [fifth_perc_rf,fifth_perc_rr,fifth_perc_ff]
    # return np.array([fifth_perc_rf,fifth_perc_rr,fifth_perc_ff,nn_fifth_perc_rf,nn_fifth_perc_rr,nn_fifth_perc_ff]).reshape(1,6) 
#%%