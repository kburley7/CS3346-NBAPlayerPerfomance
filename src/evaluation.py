def main():
    X_test, y_test = load_test_data()

    # Must match the name used in models.py
    model_names = [
        "random_forest_target_pts_over",
        # If you later change TARGET_COL and retrain, update names here too:
        # "random_forest_target_trb_over",
        # "random_forest_target_ast_over",
    ]

    for name in model_names:
        model = load_model(name)
        evaluate_model(model, X_test, y_test, model_name=name)
