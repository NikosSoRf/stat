from sklearn.model_selection import GridSearchCV

# Определение гиперпараметров для Grid Search базовых моделей
base_param_grids = {
    "SVM": {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    "KNN": {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
}
# Определение гиперпараметров для метамодели
meta_param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [100, 200]
}

def grid_search_base(base_models, X_train_scaled, X_train, y_train):
    optimized_base_models = {}
    best_params = {}
    for name, model in base_models.items():
        print(f"\n--- Optimizing {name} ---")
        
        if name in base_param_grids:
            if name in ["SVM", "KNN"]:
                X_train_used = X_train_scaled
            else:
                X_train_used = X_train
                
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=base_param_grids[name],
                cv=4,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train_used, y_train)
            
            # Сохранение лучших параметров и модели
            best_params[f"Base_{name}"] = grid_search.best_params_
            optimized_base_models[name] = grid_search.best_estimator_
            
            print(f"Best parameters for {name}: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
        else:
            # Для моделей без Grid Search
            if name in ["SVM", "KNN"]:
                model.fit(X_train_scaled, y_train)
            else:
                model.fit(X_train, y_train)
                
            optimized_base_models[name] = model
            best_params[f"Base_{name}"] = "No hyperparameters to tune"
            print(f"No hyperparameter tuning for {name}")
    
    return optimized_base_models, best_params

def grid_search_meta(meta_model, X_meta_train, y_train, best_params):
    # Оптимизация гиперпараметров метамодели
    meta_grid_search = GridSearchCV(
        estimator=meta_model,
        param_grid=meta_param_grid,
        cv=4,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    meta_grid_search.fit(X_meta_train, y_train)
    best_meta_model = meta_grid_search.best_estimator_
    best_params["Meta_Model"] = meta_grid_search.best_params_

    print(f"Best parameters for Meta Model: {meta_grid_search.best_params_}")
    print(f"Best cross-validation score for Meta Model: {meta_grid_search.best_score_:.4f}")
    return best_meta_model, best_params["Meta_Model"]