from definitions import *

st.write('Parameters')

num_leaves = st.sidebar.slider(label = 'num_leaves', min_value = 4.0,
                          max_value = 16.0 ,
                          value = 10.0,
                          step = 0.1)

max_depth = st.sidebar.slider(label = 'max_depth', min_value = -10.00,
                          max_value = 2.00 ,
                          value = -1.00,
                          step = 0.01)
                          
min_data_in_leaf = st.sidebar.slider(label = 'min_data_in_leaf', min_value = 0.00,
                          max_value = 21.00 ,
                          value = 20.0,
                          step = 0.01)                          

feature_fraction = st.sidebar.slider(label = 'feature_fraction', min_value = 0.0,
                          max_value = 16.0 ,
                          value = 1.0,
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

top_rate = st.sidebar.slider(label = 'top_rate', min_value = 0.0000,
                          max_value = 2.0000 ,
                          value = 0.9900,
                          step = 0.0001)

other_rate = st.sidebar.slider(label = 'other_rate', min_value = 2.00,
                          max_value = 5.00 ,
                          value = 3.00,
                          step = 0.01)
                          
min_data_per_group = st.sidebar.slider(label = 'min_data_per_group', min_value = 0.00,
                          max_value = 2.00,
                          value = 0.50,
                          step = 0.01)

max_cat_threshold = st.sidebar.slider(label = 'max_cat_threshold', min_value = 8.0,
                          max_value = 15.0,
                          value = 10.5,
                          step = 0.1)
                          
learning_rate = st.sidebar.slider(label = 'learning_rate', min_value = 8.0,
                          max_value = 15.0,
                          value = 10.5,
                          step = 0.1)

num_leaves = st.sidebar.slider(label = 'num_leaves', min_value = 8.0,
                          max_value = 15.0,
                          value = 10.5,
                          step = 0.1)
                          
min_gain_to_split  = st.sidebar.slider(label = 'min_gain_to_split', min_value = 8.0,
                          max_value = 15.0,
                          value = 10.5,
                          step = 0.1)


max_bin = st.sidebar.slider(label = 'max_bin', min_value = 8.0,
                          max_value = 15.0,
                          value = 10.5,
                          step = 0.1)

num_iterations = st.sidebar.slider(label = 'num_iterations', min_value = 8.0,
                          max_value = 15.0,
                          value = 10.5,
                          step = 0.1)
                          
feature_fraction  = st.sidebar.slider(label = 'feature_fraction', min_value = 8.0,
                          max_value = 15.0,
                          value = 10.5,
                          step = 0.1)

params = {'num_leaves': num_leaves, 'max_depth': max_depth,
            'min_data_in_leaf': min_data_in_leaf, 'feature_fraction': feature_fraction,
            'lambda_l1': lambda_l1, 'lambda_l2': lambda_l2,
            'min_split_gain': min_split_gain, 'top_rate': top_rate,
            'other_rate': other_rate, 'min_data_per_group': min_data_per_group, 
            'max_cat_threshold': max_cat_threshold,
            'learning_rate':learning_rate,'num_leaves':num_leaves,'min_gain_to_split':min_gain_to_split,
            'max_bin':max_bin,'num_iterations':num_iterations,'feature_fraction':feature_fraction
            }
 
def evaluate(model, plotROC=False):
    """
    1. Print AUC and accuracy on the test set
    2. Plot ROC
    """
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.4f}')
    
    # Find optimal threshold
    rocDf = pd.DataFrame({'fpr': fpr, 'tpr':tpr, 'threshold':threshold})
    rocDf['tpr - fpr'] = rocDf.tpr - rocDf.fpr
    optimalThreshold = rocDf.threshold[rocDf['tpr - fpr'].idxmax()]
    
    # Get accuracy over the test set
    y_pred = np.where(preds >= optimalThreshold, 1, 0)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')
    
    # Plot ROC AUC
    if plotROC:
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        
        
st.title('Wine Quality Classifier Web App')


# 加载数据
iris = load_iris()
data = iris.data
target = iris.target

# 划分训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

# 模型训练

model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

probs = model.predict(X_test)
preds = probs
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
print(f'AUC: {roc_auc:.4f}')

