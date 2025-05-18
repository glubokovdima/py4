from catboost import CatBoostClassifier
model = CatBoostClassifier(task_type='GPU', devices='0')
print(model.get_param('task_type'))