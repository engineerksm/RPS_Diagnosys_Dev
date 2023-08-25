import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition
from sklearn.cross_decomposition import PLSRegression
import plotly.express as px

path_List = [
    r".\CST_SELF\E122NA-01\0. Ignition window"
  ]
len(path_List)

# Generation the Min-Max Scaler
sc = MinMaxScaler()

# Generation PCA model
pca_3d = PCA(n_components=3)
pca_2d = PCA(n_components=2)
pca_1d = PCA(n_components=1)

# Generation PLS model
pls = PLSRegression(n_components=1)

file_list_csv = []
for idx in range(0, len(path_List)):
  file_list = os.listdir(path_List[idx])
  file_list = [file for file in file_list if file.endswith(".csv")]
  file_list_csv.append(file_list)
  print("{0} file_list : {1}".format(path_List[idx], file_list))

idx_col = []
ROW = 2
COL = 3

idx = 0  # Ignition Window
# for file_idx in range(len(file_list)):
file_idx = 0

# Load the dataset
file_path = path_List[idx] + '/' + file_list[file_idx]
df = pd.read_csv(file_path)
print(file_idx)
print(df.shape)
print(df.columns)

idx_col = df.columns
num_of_feature = len(idx_col)
ROW = int(np.floor(num_of_feature/COL))+1

# Change DataFrame's Index to TimeStamp
dateTime = pd.to_datetime(df[idx_col[0]])
timeStamp = dateTime.values.astype(np.int64) // 10 ** 9
df.index = timeStamp

# Slicing dataset
ChkStartValue = 0.0
Extracted_Params = df.columns[1:]
Sliced_Data = df.loc[(df['SetPower'] >= ChkStartValue), Extracted_Params]
print("Extracted Params : {0}".format(Extracted_Params))

# Sorting DataFrame
# Sorted_Data = Sliced_Data.sort_values(by=['SetPower'], ascending=[True])

# Data Scaling
sc_df = sc.fit_transform(Sliced_Data)
Params_Result = list()

# Generation the PCA Module. At this, We need to decide the number of the principal component.
for idx in range(1, len(Extracted_Params)):
    pca = PCA(n_components=idx)
    principal_components = pca.fit_transform(sc_df)
    print('PC {0}, {1}, {2}'.format(idx, pca.explained_variance_ratio_[idx-1] * 100., pca.explained_variance_ratio_.sum() * 100.))
    # print('Explained Variance : {0}'.format(pca.explained_variance_))      # 설명 가능한 분산량.
    # print('Explained Variance in percent : {0}%'.format(pca.explained_variance_ratio_ * 100.))  # 앞서 설정한 주성분의 개수(n_components)로 전체 데이터의 분산을 얼마만큼 설명 가능한지 표시.

# Fit Data-set to Model
pca_3d_df = pca_3d.fit_transform(sc_df)
pca_2d_df = pca_2d.fit_transform(sc_df)
pca_1d_df = pca_1d.fit_transform(sc_df)

# Save PCA Data to csv format
df_PcaRes_3d = pd.DataFrame(data=pca_3d_df,
                            columns=['principal_component1', 'principal_component2', 'principal_component3'])
df_PcaRes_3d.index = Sliced_Data.index
result_Data_path = r".\Results\RPS_Diagnosys_Results"
df_PcaRes_3d.to_csv(result_Data_path + '_Pca3d_' + file_list[file_idx])

# Calculate the total variance
total_var_3d = pca_3d.explained_variance_ratio_.sum() * 100
print("PCA-3D can be explained up to {0} for the variance of dataset".format(total_var_3d))

# Draw the 3D plot
fig = px.scatter_3d(pca_3d_df, x=0, y=1, z=2, color=Sliced_Data['SetPower'],
                    color_continuous_scale=px.colors.sequential.Rainbow,
                    title=f'Through PCA_3D, Total Explained Variance: {total_var_3d:.2f}% , RPS test data : {file_list[file_idx]}',
                    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'})
# fig.add_annotation(x=0.5, y=0.5, text="SetPower")
fig.show()
# scatter_file_save_path = r"C:\Users\smkim\PycharmProjects\RF_Diagnosys\Results\scatter_"
# fig.write_image(scatter_file_save_path + "PCA_3d.png")

# PCA_3D Loading effects
loadings_3d_df = pd.DataFrame(pca_3d.components_.T, columns=['PC1', 'PC2', 'PC3'], index=Extracted_Params)
print("PCA-3D loading effect\n", loadings_3d_df)
# Draw the 3D plot
fig = px.scatter(loadings_3d_df, title=f'PCA_3D Loading Effects , RPS test data : {file_list[file_idx]}')
fig.show()

# Save PCA_2D Data to csv format
df_PcaRes_2d = pd.DataFrame(data=pca_2d_df,
                            columns=['principal_component1', 'principal_component2'])
df_PcaRes_2d.index = Sliced_Data.index
result_Data_path = r".\Results\RPS_Diagnosys_Results"
df_PcaRes_2d.to_csv(result_Data_path + '_Pca2d_' + file_list[file_idx])

# Calculate the total variance
total_var_2d = pca_2d.explained_variance_ratio_.sum() * 100
print("total_var_PCA2d : ", total_var_2d)

# Draw the 2D plot
fig = px.scatter(pca_2d_df, x=0, y=1, color=Sliced_Data['SetPower'],
                 color_continuous_scale=px.colors.sequential.Rainbow,
                 title=f'Through PCA_2D, Total Explained Variance: {total_var_2d:.2f}% , RPS test data : {file_list[file_idx]}',
                 labels={'0': 'PC 1', '1': 'PC 2'})
fig.show()

# PCA_2D Loading effects
loadings_2d_df = pd.DataFrame(pca_2d.components_.T, columns=['PC1', 'PC2'], index=Extracted_Params)
print("PCA-2D loading effect\n", loadings_2d_df)
# Draw the 2D plot
fig = px.scatter(loadings_2d_df, title=f'PCA_2D Loading Effects , RPS test data : {file_list[file_idx]}')
fig.show()

# Save PCA_1D Data to csv format
df_PcaRes_1d = pd.DataFrame(data=pca_1d_df,
                            columns=['principal_component1'])
df_PcaRes_1d.index = Sliced_Data.index
result_Data_path = r".\Results\RPS_Diagnosys_Results"
df_PcaRes_1d.to_csv(result_Data_path + '_Pca1d_' + file_list[file_idx])

# Calculate the total variance
total_var_1d = pca_1d.explained_variance_ratio_.sum() * 100
print("total_var_PCA1d : ", total_var_1d)

# Draw the 1D plot
fig = px.scatter(pca_1d_df, color=Sliced_Data['SetPower'],
                 color_continuous_scale=px.colors.sequential.Rainbow,
                 title=f'Through PCA_1D, Total Explained Variance: {total_var_1d:.2f}% , RPS test data : {file_list[file_idx]}',
                 labels={'0': 'PC 1'})
fig.show()

# PCA_1D Loading effects
loadings_1d_df = pd.DataFrame(pca_1d.components_.T, columns=['PC1'], index=Extracted_Params)
print("PCA-2D loading effect\n", loadings_1d_df)
# Draw the 2D plot
fig = px.scatter(loadings_1d_df, title=f'PCA_1D Loading Effects , RPS test data : {file_list[file_idx]}')
fig.show()

for featureIdx in range(1, num_of_feature):
    plt.figure(1, figsize=(20, 22))
    subTitle = "Exploratory Data Analysis for RPS test data : " + file_list[file_idx]
    plt.suptitle(subTitle)
    axe = plt.subplot(ROW, COL, featureIdx)
    plt.title(idx_col[featureIdx])

    #격자 여백 설정
    plt.subplots_adjust(wspace=0.3, hspace=1.0)

    # for idx in range(0, len(path_List)):
    normalData = df[idx_col[featureIdx]]

    plt.plot(timeStamp, normalData, label='Operation Data', color='green', markersize=3, linewidth=2, alpha=0.7)
    plt.legend()
    plt.xlabel('time')

result_file_save_path = r".\Results\plot_"
img_save_path = result_file_save_path + subTitle + '.png'
plt.savefig(img_save_path)
print(img_save_path)
plt.show()
