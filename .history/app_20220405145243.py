from definitions import *

st.set_option('deprecation.showPyplotGlobalUse', False)
st.sidebar.subheader("请选择模型参数:sunglasses:")

num_leaves = st.sidebar.slider(label = 'num_leaves', min_value = 4,
                          max_value = 200 ,
                          value = 31,
                          step = 1)

max_depth = st.sidebar.slider(label = 'max_depth',  min_value = -1,
                          max_value = 15,
                          value = -1,
                          step = 1)
                          
min_data_in_leaf = st.sidebar.slider(label = 'min_data_in_leaf',  min_value = 8,
                          max_value = 55,
                          value = 20,
                          step = 1)

feature_fraction = st.sidebar.slider(label = 'feature_fraction', min_value = 0.0,
                          max_value = 1.0 ,
                          value = 0.8,
                          step = 0.1)
                          
min_data_per_group = st.sidebar.slider(label = 'min_data_per_group', min_value = 6,
                          max_value = 289 ,
                          value = 100,
                          step = 1)

max_cat_threshold = st.sidebar.slider(label = 'max_cat_threshold', min_value = 6,
                          max_value = 289 ,
                          value = 32,
                          step = 1)
                          
learning_rate = st.sidebar.slider(label = 'learning_rate', min_value = 0.0,
                          max_value = 1.00,
                          value = 0.05,
                          step = 0.01)

num_leaves = st.sidebar.slider(label = 'num_leaves',  min_value = 6,
                          max_value = 289 ,
                          value = 31,
                          step = 1)

max_bin = st.sidebar.slider(label = 'max_bin', min_value = 6,
                          max_value = 289 ,
                          value = 255,
                          step = 1)

num_iterations = st.sidebar.slider(label = 'num_iterations', min_value = 8,
                          max_value = 289,
                          value = 100,
                          step = 1)
                                 
st.title('LightGBM-parameter-tuning-with-streamlit')


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
            'min_data_per_group': min_data_per_group, 
            'max_cat_threshold': max_cat_threshold,
            'learning_rate':learning_rate,'num_leaves':num_leaves,
            'max_bin':max_bin,'num_iterations':num_iterations
            }

gbm = lgb.train(params, lgb_train, num_boost_round=2000, valid_sets=lgb_eval, early_stopping_rounds=500)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)  
probs = gbm.predict(X_test, num_iteration=gbm.best_iteration)  # 输出的是概率结果  

fpr, tpr, thresholds = roc_curve(y_test, probs)
st.write('------------------------------------')
st.write('Confusion Matrix:')
st.write(confusion_matrix(y_test, np.where(probs > 0.5, 1, 0)))

st.write('------------------------------------')
st.write('Classification Report:')
report = classification_report(y_test, np.where(probs > 0.5, 1, 0), output_dict=True)
report_matrix = pd.DataFrame(report).transpose()
st.dataframe(report_matrix)

st.write('------------------------------------')
st.write('ROC:')

plot_roc(fpr, tpr)