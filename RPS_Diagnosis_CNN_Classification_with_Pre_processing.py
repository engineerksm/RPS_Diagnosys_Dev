import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

path_List = [
    "./CST_SELF/E122NA-01/1. SRF"
  ]
len(path_List)

npy_path_path = './npy_data/E122NA-01/'
print(npy_path_path)

file_list_csv = []
for idx in range(0, len(path_List)):
  file_list = os.listdir(path_List[idx])
  file_list = [file for file in file_list if file.endswith(".csv")]
  file_list_csv.append(file_list)
  print("{0} file_list : {1}".format(path_List[idx], file_list))

save_file = []
sample_len = 277
filepath_divide_idx = 37
idx = 0
Train = 5
Test = 1
Total = Train + Test
y_val = 1

idx_col = ['Impedance']
num_of_feature = len(idx_col)

for featureIdx in range(0, num_of_feature):

    plt.figure(1)
    plt.suptitle("Normal Data")
    ax1 = plt.subplot(3, 4, featureIdx+1)
    plt.title(idx_col[featureIdx])
    # for idx in range(0, len(path_List)):
    idx = 0  # Normal Data
    for fileName in file_list_csv[idx]:
        file = path_List[idx] + '/' + fileName
        loadData = pd.read_csv(file)
        normalData = loadData[idx_col[featureIdx]]
        if fileName == '1. ReferenceData.csv':
            plt.plot(normalData, label='normal', color='green', markersize=3, linewidth=2, alpha=0.7)
        else:
            plt.plot(normalData, label='Normal')

    plt.figure(2)
    plt.suptitle("Abnormal Case1. Agile_Rail_Threshold Err.")
    ax2 = plt.subplot(3, 4, featureIdx+1)
    plt.title(idx_col[featureIdx])
    # for idx in range(1, len(path_List)):
    idx = 1  # Abnormal Case1. Agile_Rail_Threshold Err.
    for fileName in file_list_csv[idx]:
        file = path_List[idx] + '/' + fileName
        loadData = pd.read_csv(file)
        abnormalData1 = loadData[idx_col[featureIdx]]
        if fileName == '1. ReferenceData.csv':
            plt.plot(abnormalData1, label='normal1', color='green', markersize=3, linewidth=2, alpha=0.7)
        else:
            plt.plot(abnormalData1, label='Abnormal1')

    plt.figure(3)
    plt.suptitle("Abnormal Case2. Drive_Upper_Limit Err.")
    ax3 = plt.subplot(3, 4, featureIdx+1)
    plt.title(idx_col[featureIdx])
    # for idx in range(1, len(path_List)):
    idx = 2  # Abnormal Case2. Drive_Upper_Limit Err.
    for fileName in file_list_csv[idx]:
        file = path_List[idx] + '/' + fileName
        loadData = pd.read_csv(file)
        abnormalData2 = loadData[idx_col[featureIdx]]
        if fileName == '1. ReferenceData.csv':
            plt.plot(abnormalData2, label='normal2', color='green', markersize=3, linewidth=2, alpha=0.7)
        else:
            plt.plot(abnormalData2, label='Abnormal2')

    plt.figure(4)
    plt.suptitle("Abnormal Case3. CIC_Filter_Decimation Err.")
    ax4 = plt.subplot(3, 4, featureIdx+1)
    plt.title(idx_col[featureIdx])
    # for idx in range(1, len(path_List)):
    idx = 3  # Abnormal Case3. CIC_Filter_Decimation Err.
    for fileName in file_list_csv[idx]:
        file = path_List[idx] + '/' + fileName
        loadData = pd.read_csv(file)
        abnormalData3 = loadData[idx_col[featureIdx]]
        if fileName == '1. ReferenceData.csv':
            plt.plot(abnormalData3, label='normal3', color='green', markersize=3, linewidth=2, alpha=0.7)
        else:
            plt.plot(abnormalData3, label='Abnormal3')

    plt.figure(5)
    plt.suptitle("Abnormal Case4. Ambient_Air_Temp Err.")
    ax5 = plt.subplot(3, 4, featureIdx+1)
    plt.title(idx_col[featureIdx])
    # for idx in range(1, len(path_List)):
    idx = 4
    for fileName in file_list_csv[idx]:
        file = path_List[idx] + '/' + fileName
        loadData = pd.read_csv(file)
        abnormalData4 = loadData[idx_col[featureIdx]]
        if fileName == '1. ReferenceData.csv':
            plt.plot(abnormalData4, label='normal4', color='green', markersize=3, linewidth=2, alpha=0.7)
        else:
            plt.plot(abnormalData4, label='Abnormal4')

    plt.figure(6)
    plt.suptitle("Abnormal Case5. Heatsink_Temp Err.")
    ax6 = plt.subplot(3, 4, featureIdx+1)
    plt.title(idx_col[featureIdx])
    # for idx in range(1, len(path_List)):
    idx = 5
    for fileName in file_list_csv[idx]:
        file = path_List[idx] + '/' + fileName
        loadData = pd.read_csv(file)
        abnormalData5 = loadData[idx_col[featureIdx]]
        if fileName == '1. ReferenceData.csv':
            plt.plot(abnormalData5, label='normal5', color='green', markersize=3, linewidth=2, alpha=0.7)
        else:
            plt.plot(abnormalData5, label='Abnormal5')

    plt.figure(7)
    plt.suptitle("Exploratory Data Analysis on the raw data")
    ax7 = plt.subplot(3, 4, featureIdx+1)
    plt.title(idx_col[featureIdx])
    for idx in range(1, len(path_List)):
        for fileName in file_list_csv[idx]:
            file = path_List[idx] + '/' + fileName
            loadData = pd.read_csv(file)
            AllData = loadData[idx_col[featureIdx]]
            if fileName == '1. ReferenceData.csv':
                plt.plot(AllData, label='normalAllData', color='green', markersize=3, linewidth=2, alpha=0.7)
            else:
                plt.plot(AllData, label='AbnormalAllData')
            # plt.plot(AllData)
plt.show()


# for idx in range(0, len(path_List)):
#     arr = np.empty((sample_len, len(idx_col), 1))
#     for n in file_list_csv[idx]:
#         file = path_List[idx] + '/' + n
#         raw = pd.read_csv(file)
#         tmp = raw[idx_col]
#         npraw = np.array(tmp)
#         indexed = npraw[0:sample_len, 0:len(idx_col)]
#         reshpaed = np.reshape(indexed, (sample_len, len(idx_col), 1))
#         arr = np.append(arr, reshpaed, axis=-1)
#     arr = np.delete(arr, [0, 0, 0], axis=-1)
#     filename = path_List[idx][filepath_divide_idx:]
#     save_file = npy_path_path + filename
#     print("file_name : {0}".format(filename))
#
#     if filename == '0. Normal_Case':
#         x_train = arr[:, :, :round(np.shape(arr)[2] / Total * Train)]
#         x_test = arr[:, :, round(np.shape(arr)[2] / Total * Train):]
#
#         y_train = np.zeros(np.shape(x_train)[2])
#         y_test = np.zeros(np.shape(x_test)[2])
#
#         np.save(save_file + '_x_train', x_train)
#         np.save(save_file + '_y_train', y_train)
#
#         np.save(save_file + '_x_test', x_test)
#         np.save(save_file + '_y_test', y_test)
#
#     elif filename == '1. Agile Rail Threshold':
#         x_train = arr[:, :, :round(np.shape(arr)[2] / Total * Train)]
#         x_test = arr[:, :, round(np.shape(arr)[2] / Total * Train):]
#
#         y_train = np.ones(np.shape(x_train)[2]) * y_val
#         y_test = np.ones(np.shape(x_test)[2]) * y_val
#
#         np.save(save_file + '_x_train', x_train)
#         np.save(save_file + '_y_train', y_train)
#
#         np.save(save_file + '_x_test', x_test)
#         np.save(save_file + '_y_test', y_test)
#
#         y_val = y_val + 1
#
#     elif filename == '2. Drive Upper Limit':
#         x_train = arr[:, :, :round(np.shape(arr)[2] / Total * Train)]
#         x_test = arr[:, :, round(np.shape(arr)[2] / Total * Train):]
#
#         y_train = np.ones(np.shape(x_train)[2]) * y_val
#         y_test = np.ones(np.shape(x_test)[2]) * y_val
#
#         np.save(save_file + '_x_train', x_train)
#         np.save(save_file + '_y_train', y_train)
#
#         np.save(save_file + '_x_test', x_test)
#         np.save(save_file + '_y_test', y_test)
#
#         y_val = y_val + 1
#
#     elif filename == '3. CIC Filter Decimation':
#         x_train = arr[:, :, :round(np.shape(arr)[2] / Total * Train)]
#         x_test = arr[:, :, round(np.shape(arr)[2] / Total * Train):]
#
#         y_train = np.ones(np.shape(x_train)[2]) * y_val
#         y_test = np.ones(np.shape(x_test)[2]) * y_val
#
#         np.save(save_file + '_x_train', x_train)
#         np.save(save_file + '_y_train', y_train)
#
#         np.save(save_file + '_x_test', x_test)
#         np.save(save_file + '_y_test', y_test)
#
#         y_val = y_val + 1
#
#     elif filename == '4. Ambient Air Temp':
#         x_train = arr[:, :, :round(np.shape(arr)[2] / Total * Train)]
#         x_test = arr[:, :, round(np.shape(arr)[2] / Total * Train):]
#
#         y_train = np.ones(np.shape(x_train)[2]) * y_val
#         y_test = np.ones(np.shape(x_test)[2]) * y_val
#
#         np.save(save_file + '_x_train', x_train)
#         np.save(save_file + '_y_train', y_train)
#
#         np.save(save_file + '_x_test', x_test)
#         np.save(save_file + '_y_test', y_test)
#
#         y_val = y_val + 1
#
#     elif filename == '5. Heatsink Temp':
#         x_train = arr[:, :, :round(np.shape(arr)[2] / Total * Train)]
#         x_test = arr[:, :, round(np.shape(arr)[2] / Total * Train):]
#
#         y_train = np.ones(np.shape(x_train)[2]) * y_val
#         y_test = np.ones(np.shape(x_test)[2]) * y_val
#
#         np.save(save_file + '_x_train', x_train)
#         np.save(save_file + '_y_train', y_train)
#
#         np.save(save_file + '_x_test', x_test)
#         np.save(save_file + '_y_test', y_test)
#
#         y_val = y_val + 1
#
#     elif filename == 'IIR_Filter_Coefficient':
#         x_test = arr[:, :, :]
#         y_test = np.ones(np.shape(x_test)[2]) * y_val
#
#         np.save(save_file + '_x_test', x_test)
#         np.save(save_file + '_y_test', y_test)
#
#     else:
#         print('No dir')
#
#     print(x_train.shape)
#     print(x_test.shape)
#     print(y_train.shape)
#     print(y_test.shape)
#
# idx_col = ['Setpoint','Forward','Reverse','Dissipated','PA Voltage','Drive Setpoint','PA01 Current','Rail Setpoint','Soft Start Volts','VSWR','Heatsink Temp','Ambient Air Temp']
# num_of_feature = len(idx_col)
#
# plt.figure(1)
# for featureIdx in range(0, num_of_feature):
#     plt.subplot(3, 4, featureIdx+1)
#     plt.title(idx_col[featureIdx])
#     for rowIdx in range(0, x_train.shape[0]):
#         # plt.xlim(0, 277)
#         plt.plot(x_train[rowIdx, :, featureIdx])
#
# plt.show()

