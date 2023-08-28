import numpy as np
import matplotlib.pyplot as plt

"""
x, y를 각각의 평균과 분산에 따라서 생성해주는 함수를 만들었습니다. 
np.vstack은 세로로 쌓아줍니다.
(1, nrow)
(1, nrow)
"""
def normal_sampling(mu1, v1, mu2, v2, nrow):
    x = np.random.normal(mu1, v1, nrow)
    y = np.random.normal(mu2, v2, nrow)
    return np.vstack([x,y])
"""
다양한 평균과 분산에 대해서 샘플링하여 쌓아줍니다. 
np.hstack은 가로로 붙여줍니다. 
그래서 마지막에 Transpose 했습니다. 
"""
sample_size = 500
cluster_num = 3
X = np.hstack([
    normal_sampling(0, 1, 0, 1, sample_size),
    normal_sampling(2, 1, 2, 1, sample_size),
    normal_sampling(3, 1, 7, 1, sample_size),
    normal_sampling(8, 1, 4, 1, sample_size),
    normal_sampling(6, 1, 5, 1, sample_size),
    normal_sampling(6, 3, 0, 1, sample_size),
    normal_sampling(0, 3, 6, 2, sample_size)
]).T
Y = []
for i in range(0, X.shape[0]//sample_size):
    Y+=[i for j in range(0, sample_size)]
Y = np.array(Y)

plt.figure(figsize=(15, 6))
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.rainbow, alpha=0.2)

"""
대충 학습을 시키고요...
"""
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=[50, 50, 10], activation='relu')
clf.fit(X, Y)
#print(clf.score(X, Y))

"""
grid_size를 대충 잡고, 인공적으로 값들을 만들어줍니다. 
이를 활용해서 np.meshgrid를 만들고, 이 값들별로 class를 에측하고, 컨투어를 그려줍니다. 
"""
grid_size = 500
A, B = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), grid_size),
                   np.linspace(X[:, 1].min(), X[:, 1].max(), grid_size))
C = clf.predict( np.hstack([A.reshape(-1, 1), B.reshape(-1, 1)]) ).reshape(grid_size, grid_size)
plt.contourf(A, B, C, alpha=0.3, cmap=plt.cm.rainbow)
plt.axis('off')
plt.savefig("./decision_boundary_180529_1807.svg")
plt.show()

import plotly.graph_objects as go

# Valid color strings are CSS colors, rgb or hex strings
colorscale = [[0, 'gold'], [0.5, 'mediumturquoise'], [1, 'lightsalmon']]

fig = go.Figure(data =
    go.Contour(
        z=[[10, 10.625, 12.5, 15.625, 20],
           [5.625, 6.25, 8.125, 11.25, 15.625],
           [2.5, 3.125, 5., 8.125, 12.5],
           [0.625, 1.25, 3.125, 6.25, 10.625],
           [0, 0.625, 2.5, 5.625, 10]],
        colorscale=colorscale)
)

fig.show()

import plotly.express as px
import pandas as pd

result_Data_path = r".\Results\RPS_Diagnosys_Results"
file_path = result_Data_path + '_Pca2d_20220527_181445_Ignition window 1.csv'
pca_2d_df = pd.read_csv(file_path)[1:2]
Extracted_Params = pca_2d_df.columns[1:]
pca_2d_df_ = pca_2d_df.iloc[:, [1, 2]]

# Draw the 2D plot
fig = px.scatter(pca_2d_df_,
                 color_continuous_scale=px.colors.sequential.Rainbow,
                 labels={'0': 'PC 1', '1': 'PC 2'})
fig.show()

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

z = [[2, 4, 7, 12, 13, 14, 15, 16],
       [3, 1, 6, 11, 12, 13, 16, 17],
       [4, 2, 7, 7, 11, 14, 17, 18],
       [5, 3, 8, 8, 13, 15, 18, 19],
       [7, 4, 10, 9, 16, 18, 20, 19],
       [9, 10, 5, 27, 23, 21, 21, 21],
       [11, 14, 17, 26, 25, 24, 23, 22]]

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('Without Smoothing', 'With Smoothing'))

fig.add_trace(go.Contour(z=z, line_smoothing=0), 1, 1)
fig.add_trace(go.Contour(z=z, line_smoothing=0.85), 1, 2)

fig.show()

x = pca_2d_df[:, 0]
Xmin = x.min()
Xmax = x.max()

y = pca_2d_df[:, 1]
Ymin = y.min()
Ymax = y.max()

Xrange = np.linspace(Xmin, Xmax, 100)
Yrange = np.linspace(Ymin, Ymax, 100)

Xmesh, Ymesh = np.meshgrid(Xrange, Yrange)

def f(x,t):
    return x**2 + y**2

g = np.vectorize(f)
Zmesh = g(Xmesh, Ymesh)

plt.contour(Xmesh, Ymesh, Zmesh)

fig.add_trace(go.Contour(z=Zmesh, line_smoothing=0.0))

fig.show()