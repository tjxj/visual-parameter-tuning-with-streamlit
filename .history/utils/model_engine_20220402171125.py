from definitions import *


# Build LightGBM engine
def LightGBM_model(X, y):
    """
    Build LightGBM engine
    :param X: X
    :param y: y
    :return: scores , model
    """
    my_bar = st.progress(0)
    my_bar.progress(1)

    # Perform Grid-Search
    gsc = GridSearchCV(
        estimator=lgb.LGBMRegressor(),
        param_grid={
            'n_estimtaer': [1, 10, 20, 40, 100],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'boosting_type': ['gbdt'],
            'max_depth': [1, 10, 20, 30, 40, 60, 200],
            'random_state': [42]
            },
        cv=5,
        verbose=0,
        n_jobs=-1
    )
    my_bar.progress(50)

    grid_result = gsc.fit(X, y)
    my_bar.progress(80)
    best_params = grid_result.best_params_

    model = lgb.LGBMRegressor(boosting_type=best_params["boosting_type"], learning_rate=best_params["learning_rate"],
                              max_depth=best_params["max_depth"], random_state=best_params["random_state"],
                              n_estimtaer=best_params["n_estimtaer"])
    my_bar.progress(99)
    # Perform K-Fold CV
    scores = cross_val_score(model, X, y, cv=5)
    my_bar.progress(100)
    return scores, model


def model_metrics(xx, yy, model_name):
    """
    custom function prints model evaluation parameters FIX ME
    :param xx: xx
    :param yy: yy
    :param model_name: model_name
    :return: metrics
    """
    st.write('\nEvaluate results ' + model_name)
    st.write('Mean Absolute Error:', metrics.mean_absolute_error(xx, yy))
    st.write('Mean Squared Error:', metrics.mean_squared_error(xx, yy))
    st.write('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(xx, yy)))
    st.write('r2_score:', metrics.r2_score(xx, yy))
    return metrics.r2_score(xx, yy)
