from definitions import *

st.set_option('deprecation.showPyplotGlobalUse', False)
st.sidebar.subheader("请选择模型参数:sunglasses:")

num_leaves = st.sidebar.slider(label = 'num_leaves', min_value = 4.0,
                          max_value = 16.0 ,
                          value = 10.0,
                          step = 0.1)

max_depth = st.sidebar.slider(label = 'max_depth',  min_value = 8,
                          max_value = 15,
                          value = 10,
                          step = 1)
                          
min_data_in_leaf = st.sidebar.slider(label = 'min_data_in_leaf',  min_value = 8,
                          max_value = 15,
                          value = 10,
                          step = 1)

feature_fraction = st.sidebar.slider(label = 'feature_fraction', min_value = 0.0,
                          max_value = 1.0 ,
                          value = 0.3,
                          step = 0.1)

lambda_l1 = st.sidebar.slider(label = 'lambda_l1', min_value = 0.000,
                          max_value = 1.000 ,
                          value = 0.500,
                          step = 0.001)
   
lambda_l2 = st.sidebar.slider(label = 'lambda_l2', min_value = 1,
                          max_value = 72,
                          value = 36,
                          step = 1)

min_split_gain = st.sidebar.slider(label = 'min_split_gain', min_value = 6,
                          max_value = 289 ,
                          value = 144,
                          step = 1)

top_rate = st.sidebar.slider(label = 'top_rate', min_value = 0.0,
                          max_value = 1.0 ,
                          value = 0.3,
                          step = 0.1)

other_rate = st.sidebar.slider(label = 'other_rate', min_value = 0.0,
                          max_value = 1.0 ,
                          value = 0.3,
                          step = 0.1)
                          
min_data_per_group = st.sidebar.slider(label = 'min_data_per_group', min_value = 6,
                          max_value = 289 ,
                          value = 32,
                          step = 1)

max_cat_threshold = st.sidebar.slider(label = 'max_cat_threshold', min_value = 6,
                          max_value = 289 ,
                          value = 32,
                          step = 1)
                          
learning_rate = st.sidebar.slider(label = 'learning_rate', min_value = 8.0,
                          max_value = 15.0,
                          value = 10.5,
                          step = 0.1)

num_leaves = st.sidebar.slider(label = 'num_leaves',  min_value = 6,
                          max_value = 289 ,
                          value = 31,
                          step = 1)
                          
min_gain_to_split  = st.sidebar.slider(label = 'min_gain_to_split', min_value = 0.0,
                          max_value = 15.0,
                          value = 0.0,
                          step = 0.1)


max_bin = st.sidebar.slider(label = 'max_bin', min_value = 6,
                          max_value = 289 ,
                          value = 255,
                          step = 1)

num_iterations = st.sidebar.slider(label = 'num_iterations', min_value = 8,
                          max_value = 15,
                          value = 10,
                          step = 1)
                                 
st.title('parameter-tuning-with-streamlit')


# 加载数据
breast_cancer = load_breast_cancer()
data = breast_cancer.data
target = breast_cancer.target

# 划分训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

# 转换为Dataset数据格式
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# 模型训练
params = {'num_leaves': num_leaves, 'max_depth': max_depth,
            'min_data_in_leaf': min_data_in_leaf, 
            'feature_fraction': feature_fraction,
            'lambda_l1': lambda_l1, 'lambda_l2': lambda_l2,
            'min_split_gain': min_split_gain, 'top_rate': top_rate,
            'other_rate': other_rate, 'min_data_per_group': min_data_per_group, 
            'max_cat_threshold': max_cat_threshold,
            'learning_rate':learning_rate,'num_leaves':num_leaves,'min_gain_to_split':min_gain_to_split,
            'max_bin':max_bin,'num_iterations':num_iterations
            }

gbm = lgb.train(params, lgb_train, num_boost_round=2000, valid_sets=lgb_eval, early_stopping_rounds=500)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)  
probs = gbm.predict(X_test, num_iteration=gbm.best_iteration)  # 输出的是概率结果  

fpr, tpr, thresholds = roc_curve(y_test, probs)

st.write('------------------------------------')
report = classification_report(y_test, np.where(probs > 0.5, 1, 0))
report_matrix = pd.DataFrame(report)
st.write(report_matrix)

st.write('------------------------------------')

st.write('auc: ', (roc_auc_score(y_test, probs)))
st.write("ks: ", max(tpr - fpr))
st.write('------------------------------------')

st.write('confusion matrix:')

st.write(confusion_matrix(y_test, np.where(probs > 0.5, 1, 0)))

st.write('------------------------------------')

plot_roc(fpr, tpr)

# col1, col2= st.columns(2)

# with col1:
#     lgb.plot_importance(gbm)
#     st.pyplot()
# with col2:
#     plot_auc(fpr, tpr)