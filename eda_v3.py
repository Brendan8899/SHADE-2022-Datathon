# load packages
import warnings                     # ignore warnings (e.g. from future, deprecation, etc.)

import quart as quart

warnings.filterwarnings('ignore')   # for layout reasons, after I read and acknowledged them all!

import pandas
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import scipy.stats                  # for pearson correlation
from scipy.stats import entropy
from scipy.stats import chi2_contingency

# read in data
data = pandas.read_csv('newcohort_team02_standardized_v3_action.csv', low_memory=False)

# subset the data and make a copy to avoid error messages later on
sub = data[['current_age',
 'current_baseexcess',
 'current_bun',
 'current_calcium',
 'current_creatinine',
 'current_heartrate',
 'current_lactate',
 'current_mbp',
 'current_output_total',
 'current_pH',
 'current_platelet',
 'current_pt',
 'current_ptt',
 'current_sbp',
 'current_shock_index',
 'current_sofa_24hours',
 'current_spo2',
 'current_urine_output',
 'current_wbc',
 'current_weight',
 'cluster_id',
 'mortality_hospital'
 ]]
sub_data = sub.copy()
num_clus = data['cluster_id'].nunique()

for i in range(num_clus):#for each cluster
    cluster_id = data['cluster_id'][i]
    confusion_region = sub_data[sub_data['cluster_id'] == cluster_id]
    continuous_name_list = ['current_age',
                            'current_baseexcess',
                            'current_bun',
                            'current_calcium',
                            'current_creatinine',
                            'current_heartrate',
                            'current_lactate',
                            'current_mbp',
                            'current_output_total',
                            'current_pH',
                            'current_platelet',
                            'current_pt',
                            'current_ptt',
                            'current_sbp',
                            'current_shock_index',
                            'current_sofa_24hours',
                            'current_spo2',
                            'current_urine_output',
                            'current_wbc',
                            'current_weight',
                           ]
    moderator_list = ['current_age', 'current_weight']
    cont_entropy = np.ones(len(continuous_name_list))

    discrete_name_list = ['current_gender']
    disc_entropy = np.ones(len(discrete_name_list))
    my_int = cluster_id
    result = f'The cluster id is {my_int}'
    num_interval = 4
    interval = 1/num_interval
    print(result)
    for j in range(len(continuous_name_list)):
        # continuous entropy(range and bin may vary between different variables)
        #given cluster and feature: what are the probabilities?
        my_int = continuous_name_list[j]
        result = f'The selected feature is {my_int}'
        print(result)
        print(f'Probability of {num_interval} bins as follows')
        p_list = []
        for n in range(num_interval):
            p = len(confusion_region[confusion_region[continuous_name_list[j]].between(n*interval,(n+1)*interval)])/len(confusion_region)
            if n == num_interval - 1:
                p = len(confusion_region[confusion_region[continuous_name_list[j]].between(n*interval, 1)])/len(confusion_region)
            p_list.append(p)
        cont_entropy[j] = entropy(p_list, base=2)

        result = p_list
        print(result)

    for j in range(len(discrete_name_list)):

        # discrete entropy(number of classes may vary)
        p1 = len(confusion_region[confusion_region[discrete_name_list[j]] == 0])/len(confusion_region)
        p2 = len(confusion_region[confusion_region[discrete_name_list[j]] == 1])/len(confusion_region)
        disc_entropy[j] = entropy([p1, p2], base=2)

        my_int = disc_entropy[j]
        result = f'The entropy of the  feature is {my_int}'
        print(result)

    # action entropy

    p1 = len(confusion_region[confusion_region['summary_action'] == 1])/len(confusion_region)
    p2 = len(confusion_region[confusion_region['summary_action'] == 2])/len(confusion_region)
    p3 = len(confusion_region[confusion_region['summary_action'] == 3])/len(confusion_region)
    p4 = len(confusion_region[confusion_region['summary_action'] == 4])/len(confusion_region)

    l = [p1, p2, p3, p4]
    l1 = sorted([p1, p2, p3, p4], reverse=True)
    l2 = sorted(range(len(l)), key=lambda k: l[k])
    e_action = entropy([p1, p2, p3, p4], base=2)

    unc_act_count=4
    action_probs=[p1, p2, p3, p4]
    unc_act_list=[0,1,2,3]
    for prob_id in range(len(action_probs)):
        if action_probs[prob_id]<0.02:
            unc_act_count=unc_act_count-1
            unc_act_list.remove(prob_id)
    if len(unc_act_list)>1:
        print(unc_act_list)

    # for j in range(len(continuous_name_list)):
    #     print(scipy.stats.chi2_contingency(sub_data[continuous_name_list[j+1]],
    #                                 sub_data[continuous_name_list[j]]))
    #     print('hello')
    # correlation = scipy.stats.chisquare(confusion_region[continuous_name_list[0]],sub_data['mortality_90'])
    # quartile split (use qcut function & ask for 4 groups)

    for j in range(len(continuous_name_list)):#correlation
        ans=pandas.crosstab(confusion_region[continuous_name_list[j]],confusion_region['mortality_hospital'])
        chi2, p, dof, ex = chi2_contingency(ans, correction=False)
        my_int = p
        my_int_2 = continuous_name_list[j]
        result = f'The p-value of the  feature {my_int_2} is {my_int}'
        print(result)
    for i in range(len(continuous_name_list)):#correlation
        confusion_region["mod_red_qua"]= pandas.qcut(confusion_region[continuous_name_list[i]], 2, labels=["50th", "100th"])
        mod_red_group=confusion_region.groupby("mod_red_qua")
        confusion_region["mod_red_qua"] = confusion_region.mod_red_qua.astype("category")
        updated_continuous_name_list = continuous_name_list.remove(continuous_name_list[i])
        for j in range(len(updated_continuous_name_list)):
            print(mod_red_group.get_group("50th")['mortality_hospital'])
            print(mod_red_group.get_group("50th")[updated_continuous_name_list[j]])
            ans = pandas.crosstab(mod_red_group.get_group("50th")['mortality_hospital'], mod_red_group.get_group("50th")['prev_MV'])
            chi2, p, dof, ex = chi2_contingency(ans, correction=False)
    print('hello')
    # for i in range(len(continuous_name_list)):
    #     for a in range(4):
    #         for j in range(len(continuous_name_list)):#causation
    #             ans=pandas.crosstab(confusion_region[continuous_name_list[j]],confusion_region['mortality_90'])
    #             chi2, p, dof, ex = chi2_contingency(ans, correction=False)
    #             my_int = p
    #             my_int_2 = continuous_name_list[j]
    #             result = f'The p-value of the  feature {my_int_2} is {my_int}'
    #             print(result)
    # for j in range(len(continuous_name_list)):
    #     feature_arr = np.ones((4, 2))
    #     for a in range(4):
    #         feature_1=confusion_region[confusion_region[continuous_name_list[j]].between(0.25 * (a), 0.25 * (a + 1))]
    #
    #         feature_2= len(feature_1[feature_1['mortality_90'] == 0])
    #         feature_3= len(feature_1[feature_1['mortality_90'] == 1])
    #         feature_arr[a,0]=feature_2
    #         feature_arr[a,1]=feature_3

    # sub_data["quart"] = pandas.qcut(feature, 4, labels=["25th", "50th", "75th","100th"],duplicates='drop')
    # sub_data["quart"] = sub_data.quart.astype("category")
    #
    # # split into groups
    # quart = sub_data.groupby("quart")
    #
    # print(scipy.stats.chisquare(quart.get_group("25th")['mortality_90'],
    #                            quart.get_group("25th")[continuous_name_list[0]]))
    #
    # print(scipy.stats.chisquare(quart.get_group("50th")['mortality_90'],
    #                            quart.get_group("50th")[continuous_name_list[0]]))
    #
    # print(scipy.stats.pearsonr(quart.get_group("75th")['mortality_90'],
    #                           quart.get_group("75th")[continuous_name_list[0]]))
    #
    # print(scipy.stats.chisquare(quart.get_group("100th")['mortality_90'],
    #                            quart.get_group("100th")[continuous_name_list[0]]))
    #
    # print(scipy.stats.pearsonr(confusion_region['mortality_90'], confusion_region[continuous_name_list[0]]))
