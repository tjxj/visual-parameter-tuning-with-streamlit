from definitions import *

st.write.sidebar('Parameters')

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

