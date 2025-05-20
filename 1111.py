# ... (код до этого момента) ...
active_feature_set_for_others = features_long if features_long else feature_cols_initial
if not active_feature_set_for_others:
    logging.error(f"Нет признаков для регрессоров/TP-hit для {log_context_str}.")
    # Инициализируем, чтобы не было ошибок при сохранении или выводе метрик
    reg_delta_model, reg_vol_model, clf_tp_hit_model = None, None, None
else:
    X_cv_pool_others_sel = X_cv_pool[active_feature_set_for_others]
    X_test_others_sel = X_test_full[active_feature_set_for_others]

    # --- Обучение/Загрузка Регрессоров ---
    if cli_args and cli_args.skip_regressors:
        logging.info(f"Пропуск обучения регрессоров для {log_context_str}.")
        if os.path.exists(f"models/{file_prefix}_reg_delta.pkl"):
            reg_delta_model = joblib.load(f"models/{file_prefix}_reg_delta.pkl")
        if os.path.exists(f"models/{file_prefix}_reg_vol.pkl"):
            reg_vol_model = joblib.load(f"models/{file_prefix}_reg_vol.pkl")
        if reg_delta_model and reg_vol_model:
            logging.info("Загружены существующие модели регрессоров.")
        else:
            logging.warning("Пропуск регрессоров, но существующие модели не найдены.")
    else:  # Обучаем регрессоры
        # Функция objective_regressor должна быть определена здесь или глобально
        def objective_regressor(trial, X_tr, y_tr, X_v, y_v, name, trial_num, total_trials):
            # ... (полный код objective_regressor, как в вашем файле)
            logging.info(f"Optuna HPO для {name}: Попытка {trial_num}/{total_trials}")
            params = {'iterations': trial.suggest_int('iterations', 200, 1000, step=100),
                      'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                      'depth': trial.suggest_int('depth', 3, 8),
                      'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                      'loss_function': 'RMSE', 'eval_metric': 'MAE', 'random_seed': 42,
                      'task_type': 'GPU', 'devices': '0', 'verbose': 0, 'early_stopping_rounds': 30}
            model_r = CatBoostRegressor(**params)
            try:
                model_r.fit(X_tr, y_tr, eval_set=(X_v, y_v), plot=False)
                preds_r = model_r.predict(X_v)
                mae_r = mean_absolute_error(y_v, preds_r)
                logging.info(f"  Optuna Попытка {trial_num} для {name}: MAE={mae_r:.6f}, Параметры: {trial.params}")
                return mae_r
            except Exception as e_cv_r:
                logging.error(
                    f"  Ошибка в Optuna Попытке {trial_num} для {name}: {e_cv_r}\n  Параметры: {trial.params}")
                return float('inf')


        val_size_reg = int(len(X_cv_pool_others_sel) * 0.20)
        if val_size_reg < N_CV_SPLITS * 2: val_size_reg = min(len(X_cv_pool_others_sel) // 2, N_CV_SPLITS * 5)

        # Убедимся, что после выделения val_size_reg остается достаточно для обучения в N_CV_SPLITS фолдах Optuna
        min_optuna_train_size_reg = (len(X_cv_pool_others_sel) - val_size_reg) * (
                    1 - 1 / (N_CV_SPLITS + 1))  # Примерно размер одного трейн фолда
        if len(X_cv_pool_others_sel) - val_size_reg >= N_CV_SPLITS * 5 and min_optuna_train_size_reg > N_CV_SPLITS:  # Достаточно данных для HPO
            split_idx_r = len(X_cv_pool_others_sel) - val_size_reg
            X_tr_r_opt, X_v_r_opt = X_cv_pool_others_sel.iloc[:split_idx_r], X_cv_pool_others_sel.iloc[split_idx_r:]
            y_d_tr_r_opt, y_d_v_r_opt = y_delta_cv_pool.iloc[:split_idx_r], y_delta_cv_pool.iloc[split_idx_r:]
            y_v_tr_r_opt, y_v_v_r_opt = y_vol_cv_pool.iloc[:split_idx_r], y_vol_cv_pool.iloc[split_idx_r:]

            # Проверяем, что в валидационных выборках для Optuna есть данные
            if not X_v_r_opt.empty and not y_d_v_r_opt.empty and not y_v_v_r_opt.empty:
                logging.info(f"\n--- Optuna HPO для reg_delta ({optuna_trials_reg_to_use} попыток) ---")
                study_d = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
                study_d.optimize(
                    lambda t: objective_regressor(t, X_tr_r_opt, y_d_tr_r_opt, X_v_r_opt, y_d_v_r_opt, "reg_delta",
                                                  t.number + 1, optuna_trials_reg_to_use),
                    n_trials=optuna_trials_reg_to_use, n_jobs=1, callbacks=[EarlyStoppingCallback(patience=7,
                                                                                                  min_delta=1e-5)])  # Используем EarlyStoppingCallback для регрессоров
                bp_d = study_d.best_params
                reg_delta_model = CatBoostRegressor(**bp_d, loss_function='RMSE', eval_metric='MAE', random_seed=42,
                                                    task_type="GPU", devices='0', verbose=100, early_stopping_rounds=30)
                # Финальное обучение регрессора на всем CV пуле (X_cv_pool_others_sel)
                reg_delta_model.fit(X_cv_pool_others_sel, y_delta_cv_pool, eval_set=(X_test_others_sel,
                                                                                     y_test_delta) if not X_test_others_sel.empty and not y_test_delta.empty else None)
                logging.info(f"Лучшие параметры reg_delta: {bp_d}, MAE_val_optuna: {study_d.best_value:.6f}")

                logging.info(f"\n--- Optuna HPO для reg_vol ({optuna_trials_reg_to_use} попыток) ---")
                study_v = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
                # Инициализация best_value для минимизации в EarlyStoppingCallback
                early_stopping_cb_vol = EarlyStoppingCallback(patience=7, min_delta=1e-5)
                early_stopping_cb_vol.best_value = float('inf')
                study_v.optimize(
                    lambda t: objective_regressor(t, X_tr_r_opt, y_v_tr_r_opt, X_v_r_opt, y_v_v_r_opt, "reg_vol",
                                                  t.number + 1, optuna_trials_reg_to_use),
                    n_trials=optuna_trials_reg_to_use, n_jobs=1, callbacks=[early_stopping_cb_vol])
                bp_v = study_v.best_params
                reg_vol_model = CatBoostRegressor(**bp_v, loss_function='RMSE', eval_metric='MAE', random_seed=42,
                                                  task_type="GPU", devices='0', verbose=100, early_stopping_rounds=30)
                reg_vol_model.fit(X_cv_pool_others_sel, y_vol_cv_pool, eval_set=(X_test_others_sel,
                                                                                 y_test_vol) if not X_test_others_sel.empty and not y_test_vol.empty else None)
                logging.info(f"Лучшие параметры reg_vol: {bp_v}, MAE_val_optuna: {study_v.best_value:.6f}")
            else:
                logging.warning("Валидационные данные для HPO регрессоров пусты. Обучение с параметрами по умолчанию.")
                # ... (блок дефолтного обучения регрессоров)
                reg_delta_model = CatBoostRegressor(iterations=500, learning_rate=0.03, depth=6, loss_function='RMSE',
                                                    eval_metric='MAE', task_type="GPU", devices='0', random_seed=42,
                                                    verbose=100, early_stopping_rounds=50)
                if not X_cv_pool_others_sel.empty and not y_delta_cv_pool.empty: reg_delta_model.fit(
                    X_cv_pool_others_sel, y_delta_cv_pool, eval_set=(X_test_others_sel,
                                                                     y_test_delta) if not X_test_others_sel.empty and not y_test_delta.empty else None)
                reg_vol_model = CatBoostRegressor(iterations=500, learning_rate=0.03, depth=6, loss_function='RMSE',
                                                  eval_metric='MAE', task_type="GPU", devices='0', random_seed=42,
                                                  verbose=100, early_stopping_rounds=50)
                if not X_cv_pool_others_sel.empty and not y_vol_cv_pool.empty: reg_vol_model.fit(X_cv_pool_others_sel,
                                                                                                 y_vol_cv_pool,
                                                                                                 eval_set=(
                                                                                                     X_test_others_sel,
                                                                                                     y_test_vol) if not X_test_others_sel.empty and not y_test_vol.empty else None)
        else:  # Дефолтные для регрессоров, если данных для HPO мало
            logging.warning("Мало данных для Optuna HPO регрессоров. Обучение с параметрами по умолчанию.")
            reg_delta_model = CatBoostRegressor(iterations=500, learning_rate=0.03, depth=6, loss_function='RMSE',
                                                eval_metric='MAE', task_type="GPU", devices='0', random_seed=42,
                                                verbose=100, early_stopping_rounds=50)
            if not X_cv_pool_others_sel.empty and not y_delta_cv_pool.empty: reg_delta_model.fit(X_cv_pool_others_sel,
                                                                                                 y_delta_cv_pool,
                                                                                                 eval_set=(
                                                                                                     X_test_others_sel,
                                                                                                     y_test_delta) if not X_test_others_sel.empty and not y_test_delta.empty else None)
            reg_vol_model = CatBoostRegressor(iterations=500, learning_rate=0.03, depth=6, loss_function='RMSE',
                                              eval_metric='MAE', task_type="GPU", devices='0', random_seed=42,
                                              verbose=100, early_stopping_rounds=50)
            if not X_cv_pool_others_sel.empty and not y_vol_cv_pool.empty: reg_vol_model.fit(X_cv_pool_others_sel,
                                                                                             y_vol_cv_pool, eval_set=(
                    X_test_others_sel, y_test_vol) if not X_test_others_sel.empty and not y_test_vol.empty else None)

    # --- Блок для clf_tp_hit ---
    # Этот блок должен быть на том же уровне отступа, что и `if cli_args and cli_args.skip_regressors:`
    # т.е. ВНУТРИ `else:` от `if not active_feature_set_for_others:`

    clf_tp_hit_model, _, opt_thresh_tp_hit = (None, None, 0.5)  # Инициализация по умолчанию
    if not (cli_args and cli_args.skip_tp_hit):  # Если НЕ пропускаем обучение TP-Hit
        if train_tp_hit_model_flag and y_tp_hit_cv_pool is not None:
            if y_tp_hit_cv_pool.nunique() >= 2 and y_tp_hit_cv_pool.value_counts().min() >= N_CV_SPLITS:
                clf_tp_hit_model, _, opt_thresh_tp_hit = _train_binary_classifier(
                    X_cv_pool_full=X_cv_pool_others_sel, y_cv_pool_target=y_tp_hit_cv_pool,
                    # Используем X_cv_pool_others_sel
                    X_test_full=X_test_others_sel,  # и X_test_others_sel для TP-Hit
                    y_test_target=y_test_tp_hit,
                    target_name="tp_hit",
                    log_context_str=log_context_str,
                    file_prefix=file_prefix,
                    n_top_features=n_top_features_to_use,  # Передаем параметры
                    n_optuna_trials=optuna_trials_clf_to_use,
                    minority_weight_boost_factor=minority_boost_to_use
                )
                if clf_tp_hit_model: optimal_thresholds["tp_hit"] = opt_thresh_tp_hit
            else:
                logging.warning(
                    f"Мало данных/классов для обучения clf_tp_hit ({y_tp_hit_cv_pool.value_counts().to_dict() if y_tp_hit_cv_pool is not None else 'N/A'}). Пропуск.")
        # else: # Если train_tp_hit_model_flag is False или y_tp_hit_cv_pool is None
        # logging.info(f"Обучение clf_tp_hit не требуется или невозможно для {log_context_str}.")
    else:  # Пропускаем обучение TP-Hit, пытаемся загрузить
        logging.info(f"Пропуск обучения clf_tp_hit для {log_context_str}.")
        model_path_tp = f"models/{file_prefix}_clf_tp_hit.pkl"
        threshold_path_tp = f"models/{file_prefix}_optimal_thresholds.json"  # тот же файл
        if os.path.exists(model_path_tp):
            try:
                clf_tp_hit_model = joblib.load(model_path_tp)
                if os.path.exists(threshold_path_tp):
                    with open(threshold_path_tp, "r") as f_thr:
                        opt_thresh_tp_hit = json.load(f_thr).get("tp_hit", 0.5)
                logging.info(f"Загружена существующая модель clf_tp_hit. Порог: {opt_thresh_tp_hit}")
                if clf_tp_hit_model: optimal_thresholds["tp_hit"] = opt_thresh_tp_hit
            except Exception as e:
                logging.error(f"Ошибка загрузки clf_tp_hit: {e}.")
                clf_tp_hit_model, opt_thresh_tp_hit = None, 0.5
        else:
            logging.warning("Пропуск clf_tp_hit, но существующая модель не найдена.")

# Сохранение оптимальных порогов (этот блок должен быть на один уровень левее, вне else для active_feature_set_for_others)
thresholds_path = f"models/{file_prefix}_optimal_thresholds.json"
# ... (остальной код train_all_models)