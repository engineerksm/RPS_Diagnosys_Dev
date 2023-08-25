import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Designate reference path
ChkStartValue = 0.0
Target_Params = 'Setpoint'
Similarity_threshold = 85.0

# Define Reference Path
ref_path = "./Ref_Data"
tst_path = "./Tst_Data"
# Define Reference Path
_RefPath = "./Ref_Data"
_TstPath = "./Tst_Data"
result_Data_path = './Results_Revised/'
result_Params = ['Setpoint', 'Forward', 'Reverse', 'Dissipated', 'PA01 Current', 'PA Voltage']
Self_result_Params3 = ['Setpoint', 'Forward', 'Reverse', 'Dissipated', 'PA01 Current', 'PA Voltage', 'Rail Setpoint', 'Drive Setpoint', 'Soft Start Volts', 'HK Bias Voltage']

# Generation the Min-Max Scaler
sc = MinMaxScaler()
RefData = pd.DataFrame([])  # Reference DataFrame for RF Diagnosis
TstData = pd.DataFrame([])  # Test DataFrame for RF Diagnosis

def Rf_Diagnosis_Module(_RefPath, _TstPath, _TstFile):
    ref_file_list = []

    try:
        # read the reference csv file name
        for ref_file_name in os.listdir(_RefPath):
            if ref_file_name.endswith('.csv'):
                ref_file_list.append(ref_file_name)
        ref_file_list.sort(reverse=True)

        df_Ref = pd.read_csv(_RefPath + "/" + ref_file_list[0])
        df_Tst = pd.read_csv(_TstPath + "/" + _TstFile)

        # # Print the ref path
        # print(_RefPath + "/" + ref_file_list[0])
        # # read parameters
        # print(list(df_Ref.columns))
        # one_time_flag = False

        # Print the tst path
        print("Test_Data: " + _TstFile)

        # Slicing DataFrame
        RefData = df_Ref.loc[(df_Ref['Setpoint'] >= ChkStartValue), Self_result_Params3]
        TstData = df_Tst.loc[(df_Tst['Setpoint'] >= ChkStartValue), Self_result_Params3]

        # Sorting DataFrame by SetPoint
        RefData_ = RefData.sort_values(by=['Setpoint'], ascending=[True])
        TstData_ = TstData.sort_values(by=['Setpoint'], ascending=[True])

        # Change DataFrame's Index to SetPoint
        RefData__ = RefData_.set_index(['Setpoint'])
        TstData__ = TstData_.set_index(['Setpoint'])

        # Define PCA object
        pca = PCA(n_components=3)
        # Generation the Min-Max Scaler
        sc = MinMaxScaler()
        #==============Reference Data==========================#
        # Data Scaling
        sc_df_ref = sc.fit_transform(RefData__)
        # Fit Data-set to Model
        pc_ref = pca.fit_transform(sc_df_ref)
        # Save PCA Data to csv format
        df_PcaRef = pd.DataFrame(data=pc_ref[:, 0], columns=['PC1'])
        # df_PcaRef['PC2'] = pc_ref[:, 1]
        # df_PcaRef['PC3'] = pc_ref[:, 2]
        df_PcaRef.index = RefData__.index
        df_PcaRef.to_csv(result_Data_path + 'PCA_' + ref_file_list[0])

        #==============Test Data==========================#
        # Data Scaling
        sc_df_tst = sc.fit_transform(TstData__)
        # Fit Data-set to Model
        pc_tst = pca.fit_transform(sc_df_tst)
        # Save PCA Data to csv format
        df_PcaTst = pd.DataFrame(data=pc_tst[:, 0], columns=['PC1'])
        # df_PcaTst['PC2'] = pc_tst[:, 1]
        # df_PcaTst['PC3'] = pc_tst[:, 2]
        # result_tst_Df['PCA_component_1'] = principalComponents
        df_PcaTst.index = TstData__.index
        df_PcaTst.to_csv(result_Data_path + 'PCA_' + _TstFile)

        # Calculate Similarity Value
        Similarity_val = cosine_similarity(df_PcaRef, df_PcaTst)    #, dense_output=True

        sn.heatmap(Similarity_val)

        Similarity_average = round((np.ndarray.mean(Similarity_val)) * 100., 4)

        # print(Similarity_average)
        print('Similarity_average : {0} %'.format(Similarity_average))

        if (Similarity_average > Similarity_threshold):
            Similarity_result = 'Normal'
        else:
            Similarity_result = 'Abnormal'

        Result = Similarity_val

        # print(Similarity_result)
        print('Similarity Result : {0} '.format(Result))

        return Result

    except Exception as e:
        print(e)


def main():
    tst_file_list = []
    ref_file_list = []

    # read the reference csv file name
    for ref_file_name in os.listdir(_RefPath):
        if ref_file_name.endswith('.csv'):
            ref_file_list.append(ref_file_name)
    ref_file_list.sort(reverse=True)

    df_Ref = pd.read_csv(_RefPath + "/" + ref_file_list[0])

    print(list(df_Ref.columns))
    # Print the ref path
    print("Reference_Data" + " : " + ref_file_list[0])
    # read parameters
    print(list(Self_result_Params3))

    # read the test csv file name
    for tst_file_name in os.listdir(_TstPath):
        if tst_file_name.endswith('.csv'):
            tst_file_list.append(tst_file_name)
            Similarity_Result = Rf_Diagnosis_Module(ref_path, tst_path, tst_file_name)
            sn.heatmap(Similarity_Result)
        tst_file_list.sort(reverse=True)

if __name__=="__main__":
    main()
