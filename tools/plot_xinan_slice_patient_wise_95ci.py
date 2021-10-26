import joblib
import numpy as np
from sklearn.metrics import roc_auc_score,accuracy_score, roc_curve, auc, confusion_matrix

cls4results = joblib.load('cls4_101.pkl')

def ci(y_t,y_p,best_thresh,senspe):
    y_pred = y_p
    y_true = y_t
    n_bootstraps = 100
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    if senspe =='sen':
        origin = get_sensitivity(y_pred, y_true,best_thresh)
    if senspe =='spe':
        origin = get_specificity(y_pred, y_true,best_thresh)
    if senspe =='acc':
        origin = get_accuracy(y_pred, y_true,best_thresh)
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue
        if senspe =='sen':
            score = get_sensitivity(y_pred[indices], y_true[indices],best_thresh)
        if senspe =='spe':
            score = get_specificity(y_pred[indices], y_true[indices],best_thresh)
        if senspe =='acc':
            score = get_accuracy(y_pred[indices], y_true[indices],best_thresh)
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_lower = (sorted_scores[int(0.05 * len(sorted_scores))])#3decimal
    confidence_upper = (sorted_scores[int(0.95 * len(sorted_scores))])
    return (origin,confidence_lower,confidence_upper)

def roc_ci(y_t,y_p):
    y_pred = y_p
    y_true = y_t
#     print("Original ROC area: {:0.3f}".format(roc_auc_score(y_true, y_pred)))
    n_bootstraps = 100
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    #     print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = (sorted_scores[int(0.05 * len(sorted_scores))])#3decimal
    confidence_upper = (sorted_scores[int(0.95 * len(sorted_scores))])
#     print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
#         confidence_lower, confidence_upper))
    return (confidence_lower,confidence_upper)

def get_sensitivity(SR,GT,threshold=0.1):
    # Sensitivity == Recall
    SR = SR > threshold
#     GT = GT == torch.max(GT)
    # TP : True Positive
    # FN : False Negative
    TP = (SR==1)&(GT==1)
    FN = (SR==0)&(GT==1)

    SE = float(TP.sum())/(float((TP+FN).sum()) + 1e-6)
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SR = SR > threshold
    TN = (SR==0)&(GT==0)
    FP = (SR==1)&(GT==0)
    SP = float(TN.sum())/(float((TN+FP).sum()) + 1e-6)
    return SP


def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > threshold
    TP = (SR==1)&(GT==1)
    TN = (SR==0)&(GT==0)
    P = (GT==1)
    N = (GT==0)
    AC = float((TN+TP).sum())/(float((P+N).sum()) + 1e-6)
    return AC

def get_each_metrics(preds,gts):
    thresh_array = np.arange(0,1,0.001)
    res = 0
    best_thresh = 0
    for thresh in thresh_array:
        specificity = get_specificity(preds, gts, thresh)
        sensitivity = get_sensitivity(preds, gts, thresh)
        tmp_metric = specificity + sensitivity
        if tmp_metric > res:
            res = tmp_metric
            best_thresh = thresh

    #cal roc
    roc = roc_auc_score(gts, preds)
    roc_lower,roc_upper = roc_ci(gts, preds)
    print('ROC')
    print(roc,roc_lower,roc_upper)
    # cal acc
    acc,acc_low,acc_high = ci(gts,preds,best_thresh,senspe='acc')
    print('accuracy')
    print(acc,acc_low,acc_high)
    #cal sen
    sen,sen_low,sen_high = ci(gts,preds,best_thresh,senspe='sen')
    print('sensitivity')
    print(sen,sen_low,sen_high)
    #cal spe
    spe,spe_low,spe_high = ci(gts,preds,best_thresh,senspe='spe')
    print('specificity')
    print(spe,spe_low,spe_high)




