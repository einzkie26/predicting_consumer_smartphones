# Purchase Intent Prediction

A Streamlit app with dropdown and chat input, shared prediction results, and profile snapshots. Python code uses classes organized under `purchase_intent`; `app.py` starts the application.

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Enter the customer's actual age, gender, product category, brand, price, previous purchase count, and recorded satisfaction rating. Both modes use the same validated profile and model pipeline. Chat offers randomized quick replies for the current question. Satisfaction is required and never filled with an assumed value.

## Supported inputs

| Input | Supported values |
| --- | --- |
| Age | Whole numbers 18–69 |
| Gender | Female or Male, encoded as 0 or 1 as documented in the supplied notebook |
| Category | Headphones, Laptops, Smart Watches, Smartphones, Tablets |
| Brand | Apple, HP, Other Brands, Samsung, Sony |
| Price | $100.38–$2,999.85 for prices entered in cents |
| Previous purchases | Whole numbers 1–19 |
| Recorded satisfaction | Whole-number rating 1–5 |

These bounds come from the supplied CSV. Age 100 is outside that evidence, so the app withholds the prediction and requests a supported age. It does not clamp age or manufacture a zero probability. Invalid chat edits clear the displayed result until corrected.

## Model and evaluation

The active artifact is `artifacts/purchase_intent_model.joblib`, with preprocessing and prediction in one pipeline. `artifacts/model_report.json` contains its checksum, dataset checksum, versions, supported ranges, validation comparisons, test IDs, calibration data, and metrics. The previous `.pkl` files remain available but are not loaded by the app.

The model was trained from the 9,000 original labeled rows without changing `PurchaseIntent` or balancing the dataset by discarding rows. A stratified split with seed 42 reserves 1,800 rows for evaluation. Within the remaining 7,200 rows, 5,400 rows fit candidates and 1,800 select the model and calibration by validation log loss. The candidates include logistic regression, three XGBoost configurations, and two histogram gradient boosting configurations. Sigmoid and isotonic calibration are compared for the best base model using five-fold predictions from the training partition. The selected depth-3 XGBoost model with sigmoid calibration is refitted on the 7,200 development rows; calibration uses out-of-fold predictions. The evaluation rows are excluded from fitting, calibration, threshold tuning, and hyperparameter selection.

| Holdout metric | Value |
| --- | --- |
| Accuracy at the 50% threshold | 94.00% |
| Precision | 93.76% |
| Recall | 95.78% |
| F1 | 94.76% |
| ROC AUC | 0.9340 |
| Log loss | 0.2322 |
| Brier score | 0.0569 |
| Majority-class accuracy baseline | 56.67% |
| Incorrect predictions | 108 of 1,800 |

A preliminary diagnostic omitting gender was evaluated on this same holdout before the feature schema was corrected to include all seven recorded inputs. This is internal holdout validation; a separate dataset is needed for fully independent confirmation. Collection quality, population representativeness, and future customer performance have not been independently established. Dataset associations involving gender may not generalize. A 94% classification accuracy does not mean every probability is correct or every individual outcome can be predicted.

[Scikit-learn's calibration documentation](https://scikit-learn.org/stable/modules/calibration.html) explains the distinction between classifier scores and calibrated probabilities and the use of data separate from base-model fitting for calibration.

## Changes from the supplied notebook

The [training notebook](https://colab.research.google.com/drive/1QtOEp2IUo0lqB4Betd0uoWs_SOSl63Ke) contains cells that manually replace purchase-intent labels for expensive products. Its final grid search fits the full feature table before a later evaluation on an earlier test subset. That evaluation is not an independent test of that fitted model. Some outputs also depend on variables or cell states not reproducible in notebook order.

The app previously ignored the collected gender, silently substituted satisfaction `3`, and ran an unrelated scaler whose transformed output was discarded. The replacement uses all seven actual recorded values and a single fitted preprocessing pipeline. An older person's probability cannot be judged from age alone; outside the supported range, no estimate is provided.

## Retraining and checks

```powershell
python -B -m purchase_intent.training.train_model
python -B -m unittest discover -s tests -v
```

The trainer preserves source labels, records the split and metrics, and verifies prediction consistency after reloading the artifact. The app checks the artifact checksum, input schema, category options, and supported bounds against the report before loading it. It clears old session results when the model changes.

The serialized model was produced with Python 3.15.0, scikit-learn 1.9.1, XGBoost 3.4.1, pandas 3.1.0, NumPy 2.5.4, and joblib 1.6.0. Use the recorded model dependency versions, or retrain both artifacts in a different environment before running the app.


Made for Business Analytics Project
-Stein And Fhiona
