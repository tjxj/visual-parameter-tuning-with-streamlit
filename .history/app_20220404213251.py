from definitions import *

st.set_option('deprecation.showPyplotGlobalUse', False)
st.write('Parameters')

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


# ????????????
breast_cancer = load_breast_cancer()
data = breast_cancer.data
target = breast_cancer.target

# ?????????????????????????????????
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

# ?????????Dataset????????????
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# ????????????
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
probs = gbm.predict(X_test, num_iteration=gbm.best_iteration)  # ????????????????????????  

st.write('Start training...')  

lgb.plot_importance(gbm)
st.pyplot()

fpr, tpr, thresholds = roc_curve(y_test, probs)

st.write('auc: ', (roc_auc_score(y_test, probs)))

def plot_auc(fpr, tpr, label=None):
    # ??????auc??????
    
    # ????????????
    
    plt.stackplot(fpr, tpr, color='steelblue', alpha=0.5, edgecolor='black')
    
    # ????????????
    
    plt.plot(fpr, tpr, color='black', lw=1)
    
    # ???????????????
    
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    
    # ??????????????????
    
    plt.text(0.5, 0.3, 'ROC curve (area=%0.2f)' % roc_auc_score(y_test, probs))
    
    # ??????x??????y???
    
    plt.xlabel('1-Specificity')
    
    plt.ylabel('Sensitivity')
    
    # ????????????
    
    plt.show()
    st.pyplot()


# ??????ks??????????????????0.2

st.write("ks: ", max(tpr - fpr))

st.write('------------------------------------')

st.write(classification_report(y_test, np.where(probs > 0.5, 1, 0)))

st.write('-----------------????????????-----------------------')

st.write(confusion_matrix(y_test, np.where(probs > 0.5, 1, 0)))


def plot_roc(fpr, tpr, label=None):
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    st.pyplot()
    
plot_auc(fpr, tpr)
plot_roc(fpr, tpr)