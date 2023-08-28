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

# Load the dataset
file_idx = 0
file_path = path_List[idx] + '/' + file_list[file_idx]
df = pd.read_csv(file_path)
print(file_idx)
print(df.shape)
print(df.columns)

idx_col = df.columns
df.index = df[idx_col[0]]
df_sample = df.loc[:, idx_col[1]:]

# Draw EDA Plot
ROW = 1
COL = 1

for col_idx in range(1, len(idx_col)):
    plt.figure(1, figsize=(20, 22))
    supTitle = "Exploratory Data Analysis for RPS SRF data "
    plt.suptitle(supTitle)

    axe = plt.subplot(ROW, COL, 1)
    plt.title("Source Resonance Frequency")
    print(df_sample.head())

    plotData = df_sample[idx_col[col_idx]]

    if idx_col[col_idx] == 'Current_Fault':
        color_name = 'red'
    else:
        color_name = 'green'

    plt.plot(plotData, label=idx_col[col_idx], color=color_name, markersize=3, linewidth=2, alpha=0.7)
    plt.legend()
    plt.xlabel('Frequency')
    plt.ylabel('Impedance')

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

