import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sn
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import decomposition
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler

# Designate reference path
ChkStartValue = 0.0
Target_Params = 'Setpoint'
Similarity_Mean_threshold = 85.0
Similarity_Peak_Pos_threshold = 0.9
Similarity_Peak_Neg_threshold = 0.65

# Define Reference Path
ref_path = "./Ref_Data"
# tst_path = "./Tst_Data"
# tst_Path_1 = "./Tst_Data"
tst_Path = "./Tst_Data"
# Define Reference Path
# _RefPath = "./Ref_Data"
# _TstPath = "./Tst_Data"

result_Data_path = './Results/'
pcaData_result_path = './Results/RF_Diagnosys_Results/pca_data/'
pcaLoading_result_path = './Results/RF_Diagnosys_Results/pca_loading/'
heatMap_result_path = './Results/RF_Diagnosys_Results/Similarity_Heatmap/'

result_Params = ['Setpoint', 'Forward', 'Reverse', 'Dissipated', 'PA01 Current', 'PA Voltage']
Self_result_Params_SelfTest = ['Forward', 'Reverse', 'Dissipated', 'Rail Setpoint', 'Drive Setpoint', 'PA01 Current', 'HK Bias Voltage', 'Soft Start Volts', 'Setpoint', 'Gamma Magnitude', 'Gamma Phase',  'PA02 Current', 'PA03 Current', 'PA Voltage', 'Fan Current', 'Heatsink Temp']
Self_result_Params_FNG = ['Forward', 'Reverse', 'Dissipated', 'Rail Setpoint', 'Drive Setpoint', 'PA01 Current    ', 'HK Bias Voltage ', 'Soft Start Volts']    # 'Setpoint', 'Gamma Magnitude', 'Gamma Phase', ,

Self_target_Params1 = ['Forward']
Similarity_Target1_Peak_Pos_threshold = 0.9
Self_target_Params2 = ['Dissipated']
Similarity_Target2_Peak_Pos_threshold = 0.9
Self_target_Params3 = ['PA01 Current']
Similarity_Target3_Peak_Pos_threshold = 0.9
Self_target_Params4 = ['PA Voltage']
Similarity_Target4_Peak_Pos_threshold = 0.9

# Generation the Min-Max Scaler
sc = MinMaxScaler()
RefData = pd.DataFrame([])  # Reference DataFrame for RF Diagnosis
TstData = pd.DataFrame([])  # Test DataFrame for RF Diagnosis

def my_kernel(X, Y):
    return np.dot(X, Y.T)

def euclidean_distance(inst1, inst2):
    # Vector space
    return norm(inst1 - inst2)

def cosine_sim(inst1, inst2):
    return np.dot(inst1, inst2.T)/(norm(inst1)*norm(inst2))

def Rf_Diagnosis_Module(_RefPath, _TstPath, ref_file_list, tst_file_list, _ref_param_list, _tst_param_list):

    try:
#=========================RF Diagnosys Start========================================
        # Load DataSet
        df_Ref = pd.read_csv(_RefPath + "/" + ref_file_list)
        df_Tst = pd.read_csv(_TstPath + "/" + tst_file_list)

        print("Test_Data" + " : " + tst_file_list)
        print("Test_Data_Columns : {0}".format(list(df_Tst.columns)))

        # df_Ref_ = df_Ref.fillna(0)
        # df_Tst_ = df_Tst.fillna(0)

        # Slicing DataFrame
        RefData = df_Ref.loc[(df_Ref['Setpoint'] >= ChkStartValue), _ref_param_list]
        TstData = df_Tst.loc[(df_Tst['Setpoint'] >= ChkStartValue), _tst_param_list]

        # Sorting DataFrame by SetPoint
        # RefData_ = RefData.sort_values(by=['Setpoint'], ascending=[True])
        # TstData_ = TstData.sort_values(by=['Setpoint'], ascending=[True])

        # Change DataFrame's Index to SetPoint
        # RefData__ = RefData_.set_index(['Setpoint'])
        # TstData__ = TstData_.set_index(['Setpoint'])

        # Cosine Similarity
        Similarity_val = cosine_similarity(RefData, TstData)

        if Rf_Diagnosis_Module.oneTimeFlag is True:
            # apply PCA
            X = scale(RefData)
            pca_X = decomposition.PCA(n_components=2)
            pca_X.fit_transform(X)

            loadings_X = pd.DataFrame(pca_X.components_.T, columns=['PC1', 'PC2'], index=RefData.columns)
            print('Ref_loadings : {0}'.format(loadings_X))
            loadings_X.to_csv(pcaLoading_result_path + 'pca_Loading_' + ref_file_list, mode='w')
            Rf_Diagnosis_Module.oneTimeFlag = False

        # apply PCA
        Y = scale(TstData)
        pca_Y = decomposition.PCA(n_components=2)
        pca_Y.fit_transform(Y)

        loadings_Y = pd.DataFrame(pca_Y.components_.T, columns=['PC1', 'PC2'], index=TstData.columns)
        print('Tst_loadings : {0}'.format(loadings_Y))
        loadings_Y.to_csv(pcaLoading_result_path + 'pca_Loading_' + tst_file_list, mode='w')

        # Pearson Correlation Coefficient
        corr = np.corrcoef(RefData, TstData)
        print('{1} Pearson Correlation : {0}'.format(round((np.ndarray.mean(corr)) * 100., 4), tst_file_list))

        # Calculate Average
        print('{1} RawData_Similarity_average : {0}'.format(round((np.ndarray.mean(Similarity_val)) * 100., 4), tst_file_list))
        # print('{1} corr_average : {0}'.format(round((np.ndarray.mean(corr)) * 100., 4), tst_file_list))

#================================== RF Diagnosys End =====================================================
        # Data Scaling
        sc_ref = sc.fit_transform(RefData)
        sc_tst = sc.fit_transform(TstData)

        # Covariance matrix
        features_ref = sc_ref.T
        features_tst = sc_tst.T
        cov_matrix_ref = np.cov(features_ref)
        cov_matrix_tst = np.cov(features_tst)

        # Eigendecomposition
        values_ref, vectors_ref = np.linalg.eig(cov_matrix_ref)
        values_tst, vectors_tst = np.linalg.eig(cov_matrix_tst)

        projected_Ref_1 = np.fabs(sc_ref.dot(vectors_ref.T[0]))
        projected_Ref_2 = np.fabs(sc_ref.dot(vectors_ref.T[1]))

        projected_tst_1 = np.fabs(sc_tst.dot(vectors_tst.T[0]))
        projected_tst_2 = np.fabs(sc_tst.dot(vectors_tst.T[1]))

        df_PcaRef = pd.DataFrame(projected_Ref_1, columns=['PC1'])
        df_PcaRef['PC2'] = projected_Ref_2

        df_PcaTst = pd.DataFrame(projected_tst_1, columns=['PC1'])
        df_PcaTst['PC2'] = projected_tst_2

        df_PcaRef.to_csv(pcaData_result_path + 'pca_' + ref_file_list, mode='w')
        df_PcaTst.to_csv(pcaData_result_path + 'pca_' + tst_file_list, mode='w')

        # Calculate Euclidean Distance
        # Euclidean_val = euclidean_distance(df_PcaRef, df_PcaTst)

        # Pearson Correlation Coefficient
        # corr = np.corrcoef(df_PcaRef, df_PcaTst)

        # Calculate Similarity Value
        Similarity_val = cosine_similarity(df_PcaRef, df_PcaTst)

        Rf_Diagnosis_Module.count += 1
        plt.figure(Rf_Diagnosis_Module.count)
        sn.heatmap(Similarity_val)
        plt.savefig(heatMap_result_path + 'heatmap_' + tst_file_list + '.png', format='png')
        plt.title('heatmap_' + tst_file_list)
        # plt.show()

        Similarity_average = round((np.ndarray.mean(Similarity_val)) * 100., 4)
        print('{1} PCA_Similarity_average : {0}'.format(Similarity_average, tst_file_list))

        Similarity_Peak_sum = 0.0

        for idx_i in range(0, len(Similarity_val[:, 0])):
            for idx_j in range(0, len(Similarity_val[0, :])):
                if (Similarity_val[idx_i, idx_j] > Similarity_Peak_Pos_threshold):
                    Similarity_Peak_sum += 1.0
                # elif (Similarity_val[idx_i, idx_j] < Similarity_Peak_Neg_threshold):
                    # Similarity_Peak_sum -= 1.0
                # else:
                    # Similarity_Peak_sum += 0.5

        total_sample = float((len(Similarity_val[:, 0]) * len(Similarity_val[0, :])))
        Similarity_Peak_result = round((Similarity_Peak_sum / total_sample)*100., 4)

        # explained_variances = []
        # for i in range(len(values)):
        #     explained_variances.append(values[i] / np.sum(values))
        #
        # # first value is just the sum of explained variances - and must be equal to 1.
        # # The second value is an array, representing the explained variance percentage per principal component
        # print(np.round(np.sum(explained_variances), 4), '\n', np.round(explained_variances, 4))

        # print('{1} PCA_Similarity_average : {0} %'.format(Similarity_average, tst_file_list))
        # print('{1} PCA_Similarity_Peak_Results : {0} %'.format(Similarity_Peak_result, tst_file_list))

        # if (Similarity_average > Similarity_Mean_threshold) :
        #     Similarity_average_result = 'Normal'
        # else:
        #     Similarity_average_result = 'Abnormal'
        #
        # if (Similarity_Peak_result > Similarity_Mean_threshold) :
        #     Similarity_Peak_result = 'Normal'
        # else:
        #     Similarity_Peak_result = 'Abnormal'
        #
        # Result = Similarity_average_result

    except Exception as e:
         print(e)

Rf_Diagnosis_Module.oneTimeFlag = True
Rf_Diagnosis_Module.count = 0

def main():
    tst_file_list_1 = []
    tst_file_list_2 = []
    ref_file_list = []

    ref_file_list.clear()
    # read the reference csv file name
    for ref_file_name in os.listdir(ref_path):
        if ref_file_name.endswith('.csv'):
            ref_file_list.append(ref_file_name)
    ref_file_list.sort(reverse=True)

    df_Ref = pd.read_csv(ref_path + "/" + ref_file_list[0])

    print("Reference_Data_Columns : {0}".format(list(df_Ref.columns)))
    print("Reference_Data" + " : " + ref_file_list[0])

    tst_file_list_1.clear()
    # read the test csv file name
    for tst_file_name in os.listdir(tst_Path):
        if tst_file_name.endswith('.csv'):
            tst_file_list_1.append(tst_file_name)
        tst_file_list_1.sort(reverse=True)

    for idx in range(0, len(tst_file_list_1)):      # Self-Test Mode
        Rf_Diagnosis_Module(ref_path, tst_Path, ref_file_list[0], tst_file_list_1[idx], Self_result_Params_SelfTest, Self_result_Params_SelfTest)

    # read the test csv file name
    for tst_file_name in os.listdir(tst_Path):
        if tst_file_name.endswith('.csv'):
            tst_file_list_2.append(tst_file_name)
        tst_file_list_2.sort(reverse=True)

    # for idx in range(0, len(tst_file_list_2)):    # FNG Mode
    #     Rf_Diagnosis_Module(ref_path, tst_Path, ref_file_list[0], tst_file_list_2[idx], Self_result_Params_SelfTest, Self_result_Params_FNG)

if __name__=="__main__":
    main()
