# Customer Churn Prediction

End-to-end ML pipeline predicting telecom customer churn, with a deployable
inference app.

**Dataset:** [IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) — 7,043 customers, 20 features.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.737 | 0.503 | 0.783 | 0.613 | 0.841 |
| Random Forest | 0.759 | 0.532 | 0.767 | 0.628 | **0.842** |

Random Forest selected as best model on ROC-AUC. `class_weight="balanced"`
used on both models to handle the ~27% churn class imbalance; recall is
prioritized over raw accuracy since missing a churner is costlier than a
false alarm.

## Project structure

```
churn-prediction/
├── data/telco_churn.csv       # raw dataset
├── src/
│   ├── data_prep.py           # loading, cleaning, preprocessing pipeline
│   └── train.py                # trains + compares models, saves best one
├── models/
│   ├── best_model.joblib       # trained sklearn pipeline (preprocessing + model)
│   └── metrics.json            # full metrics for both models
├── app.py                      # Streamlit inference UI
└── requirements.txt
```

## Run it

```bash
pip install -r requirements.txt

# retrain from scratch
python src/train.py

# launch the inference app
streamlit run app.py
```

## Notes

- Preprocessing (scaling + one-hot encoding) is baked into the saved
  pipeline, so `app.py` only needs raw feature values, not the training code.
- `TotalCharges` has a few blank strings in the raw data (new customers with
  0 tenure); these are coerced to numeric and imputed with the median.
