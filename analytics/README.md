# Module 2 — Analytics Pipeline

Run in order:

```powershell
python analytics\01_eda.py
python analytics\02_modeling.py
```

The EDA script uses the committed `titanic.csv` offline fallback, measures missingness before cleaning, applies documented threshold-based decisions, calculates IQR outliers and descriptive statistics, compares survival rates, limits correlation analysis to the six required numeric fields, and saves nine charts.

The modeling script uses a stratified 80/20 split. All imputers, encoders, and scalers are fitted inside pipelines using training data only. It compares logistic regression, a decision tree, and a random forest; evaluates baseline, class-weighted, and SMOTE imbalance strategies; tunes a random forest with five-fold cross-validation; and saves the complete fitted pipeline.

The regression side-task predicts fare and reports MAE, RMSE, R², and adjusted R². Exact results and interpretations generated on this machine are recorded below after execution.

<!-- RESULTS_START -->
## Measured findings

Missing-value handling:

- `deck`: 77.2166% missing; very high missingness; dropped because direct imputation would be unreliable.
- `age`: 19.8653% missing; 5%–30%; median-imputed to retain useful records while limiting sensitivity to outliers.
- `embarked`: 0.2245% missing; under 5%; the two affected rows were dropped.
- `embark_town`: 0.2245% missing; dropped because it duplicates `embarked`.

The IQR method identified 65 age outliers and 114 fare outliers. Fare has mean 32.0967, median 14.4542, and mode 8.05, so it is strongly right-skewed.

Female survival was 74.04%, versus 18.89% for males. Survival fell from 62.62% in first class to 47.28% in second and 24.24% in third. The strongest required-field correlations were `pclass`/`fare` (-0.5482), reflecting higher fares in better classes, and `sibsp`/`parch` (0.4145), reflecting family travel.

The charts tell a consistent story: sex was the largest visible survival divider, and higher passenger class also improved survival. Their interaction was pronounced—first-class female survival was 96.74%, while third-class male survival was 13.54%. Survivor and non-survivor age distributions overlapped substantially. Fare varied strongly by class, with a long upper tail and greater spread among first-class passengers.

Standardized age and fare had means effectively equal to 0 (`2.72e-16` and `1.40e-16`) and population standard deviations equal to 1.

## Classification and imbalance handling

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.8156 | 0.8000 | 0.6957 | 0.7442 | 0.8300 |
| Logistic Regression | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8437 |
| Decision Tree | 0.7933 | 0.8636 | 0.5507 | 0.6726 | 0.8292 |

SMOTE produced the best F1 trade-off (precision 0.7397, recall 0.7826, F1 0.7606), narrowly ahead of balanced class weights (F1 0.7552) and ahead of baseline logistic regression (F1 0.7244). SMOTE is applied only inside the training pipeline after the split.

The best random forest used `max_depth=5`, `max_features='sqrt'`, and `n_estimators=100`. Its five-fold cross-validation F1 was 0.7459 and its out-of-bag score was 0.8272.

## Regression and recommendation

Fare regression produced MAE 20.8977, RMSE 30.5328, R² 0.3975, and adjusted R² 0.3617. The residual plot shows increasing spread at larger predictions and several large positive residuals, indicating heteroscedasticity and the difficulty of modeling the right-skewed fare tail with a linear model.

The random forest is the recommended classifier because it achieved the strongest measured test accuracy (0.8156) and F1 (0.7442), with precision 0.8000, recall 0.6957, and AUC 0.8300. Logistic regression had a slightly higher AUC, but the forest provided the best overall thresholded classification balance. The saved fitted pipeline can accept raw passenger rows directly.
<!-- RESULTS_END -->
