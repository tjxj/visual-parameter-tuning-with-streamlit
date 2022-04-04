from definitions import *

"""
：控制了叶节点的数目，控制树模型复杂度的主要参数。
, default=-1, type=int，树的最大深度限制，防止过拟合
, default=, type=int, 叶子节点最小样本数，防止过拟合
feature_fraction, default=1.0, type=double, 0.0 < feature_fraction < 1.0,随机选择特征比例，加速训练及防止过拟合
feature_fraction_seed, default=2, type=int，随机种子数，保证每次能够随机选择样本的一致性
bagging_fraction, default=1.0, type=double, 类似随机森林，每次不重采样选取数据
lambda_l1, default=0, type=double, L1正则
lambda_l2, default=0, type=double, L2正则
min_split_gain, default=0, type=double, 最小切分的信息增益值
top_rate, default=0.2, type=double，大梯度树的保留比例
other_rate, default=0.1, type=int，小梯度树的保留比例
min_data_per_group, default=100, type=int，每个分类组的最小数据量
max_cat_threshold, default=32, type=int，分类特征的最大阈值
针对更快的训练速度：

通过设置 bagging_fraction 和 bagging_freq 参数来使用 bagging 方法
通过设置 feature_fraction 参数来使用特征的子抽样
使用较小的 max_bin
使用 save_binary 在未来的学习过程对数据加载进行加速
获取更好的准确率：

使用较大的 max_bin （学习速度可能变慢）
使用较小的 learning_rate 和较大的 num_iterations
使用较大的 num_leaves （可能导致过拟合）
使用更大的训练数据
尝试 dart
缓解过拟合：

使用较小的 max_bin
使用较小的 num_leaves
使用 min_data_in_leaf 和 min_sum_hessian_in_leaf
通过设置 bagging_fraction 和 bagging_freq 来使用 bagging
通过设置 feature_fraction 来使用特征子抽样
使用更大的训练数据
使用 lambda_l1, lambda_l2 和 min_gain_to_split 来使用正则
"""

num_leaves = st.sidebar.slider(label = 'num_leaves', min_value = 4.0,
                          max_value = 16.0 ,
                          value = 10.0,
                          step = 0.1)

max_depth = st.sidebar.slider(label = 'max_depth', min_value = 0.00,
                          max_value = 2.00 ,
                          value = -1.00,
                          step = 0.01)
                          
min_data_in_leaf = st.sidebar.slider(label = 'min_data_in_leaf', min_value = 0.00,
                          max_value = 1.00 ,
                          value = 20,
                          step = 0.01)                          

residual_sugar = st.sidebar.slider(label = 'Residual Sugar', min_value = 0.0,
                          max_value = 16.0 ,
                          value = 8.0,
                          step = 0.1)

chlorides = st.sidebar.slider(label = 'Chlorides', min_value = 0.000,
                          max_value = 1.000 ,
                          value = 0.500,
                          step = 0.001)
   
f_sulf_diox = st.sidebar.slider(label = 'Free Sulfur Dioxide', min_value = 1,
                          max_value = 72,
                          value = 36,
                          step = 1)

t_sulf_diox = st.sidebar.slider(label = 'Total Sulfur Dioxide', min_value = 6,
                          max_value = 289 ,
                          value = 144,
                          step = 1)

density = st.sidebar.slider(label = 'Density', min_value = 0.0000,
                          max_value = 2.0000 ,
                          value = 0.9900,
                          step = 0.0001)

ph = st.sidebar.slider(label = 'pH', min_value = 2.00,
                          max_value = 5.00 ,
                          value = 3.00,
                          step = 0.01)
                          
sulphates = st.sidebar.slider(label = 'Sulphates', min_value = 0.00,
                          max_value = 2.00,
                          value = 0.50,
                          step = 0.01)

alcohol = st.sidebar.slider(label = 'Alcohol', min_value = 8.0,
                          max_value = 15.0,
                          value = 10.5,
                          step = 0.1)