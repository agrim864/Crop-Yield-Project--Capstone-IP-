# Crop Yield, Aerosols & Forecasting (Leak-safe ML pipeline)

This project builds an end-to-end, time-aware ML pipeline that links aerosol/meteorology signals to crop yield, with two prediction tasks:

1. regression: predict yield (continuous)
2. classification: predict yield tier (low / mid / high) as 3 classes

It also generates leak-free backtests, multi-scenario forecasts (baseline vs clean air vs polluted), and model interpretability outputs (permutation importance).

The pipeline is designed to be “leak-safe” for time-series/panel data: training never sees future years when evaluating.

---

## Quick start

Create venv + install requirements (your repo may already have this set up):

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Run the full pipeline:

```bash
python app.py run
```

Run dashboard:

```bash
python app.py dashboard
```

Run both:

```bash
python app.py all
```

---

## What you get (outputs)

After `python app.py run`, you get:

Data + metrics

* outputs/panel_dataset_cleaned.csv
* outputs/metrics_summary.csv

Backtest outputs (walk-forward, leak-free)

* outputs/backtest_forecasts.csv
* outputs/backtest_metrics_by_year.csv

Forecasting outputs

* outputs/yield_forecast_10_years.csv
* outputs/yield_forecast_scenarios.csv
* outputs/yield_forecast_past_only.csv
* outputs/yield_forecast_scenarios_past_only.csv

Interpretability + runtime

* outputs/permutation_importance.csv
* outputs/gb_feature_importance.csv
* outputs/timing_summary.csv

Model artifacts (cached)

* outputs/models/best_regressor.joblib
* outputs/models/best_regressor_info.json
* outputs/models/best_classifier.joblib
* outputs/models/best_classifier_info.json

---

## Current best results (from your latest run)

Dataset

* rows: 252
* regions: 4
* crops: 3
* years: 2000 → 2020

Baselines

* classifier macro-F1: 0.167
* regressor RMSE: 0.738
* regressor R²: 0.000

Best classifier

* model: Reg2Tertile(HistGB)
* OOF macro-F1: 0.594
* OOF accuracy: 0.708
* train macro-F1: 0.694

Best regressor

* model: SVR_RBF
* OOF RMSE: 0.464
* OOF R²: 0.735

Leak-free walk-forward backtest

* RMSE: 0.413
* R²: 0.802
* (also reports MAE and MAPE)

Year-level backtest stability (last few years)

* 2018: RMSE 0.593, R² 0.670
* 2019: RMSE 0.332, R² 0.875
* 2020: RMSE 0.225, R² 0.921

---

## Project structure (important files)

* app.py

  * CLI entrypoint: run pipeline, run dashboard
* run_pipeline.py

  * orchestrates: load data → build features → tune models → backtest → forecasts → reports
* data_loading.py

  * reads Excel sheets and builds one “panel” dataset
* features.py

  * feature engineering, lag creation, yield_class creation
* year_split.py

  * YearForwardSplit: time-aware CV splitting
* modeling.py

  * model training, tuning, OOF evaluation, permutation importance
* forecasting.py

  * leak-free backtest logic + recursive forecasting + scenario generation
* report_generation.py

  * writes HTML/text summary reports

---

## Data modeling: what “panel” means here

Your input is a panel dataset:

* each row is a (region, crop, year) observation
* yield is the target
* aerosol + meteorology variables are predictors
* lagged yield features are added (yield_lag1, yield_lag2) per region/crop over time

This is why time-aware evaluation matters: we want “predict future years from past years”, not random splits.

---

## Leakage prevention (core design)

This is the main reason your results are trustworthy:

1. Time-aware CV (YearForwardSplit)

* every fold trains on earlier years and tests on later years
* no mixing future data into training folds

2. Out-of-fold (OOF) scoring as the reported “best_score”

* tuning uses CV mean scores to search hyperparameters
* final reported score is OOF (leak-free) computed by refitting per fold and predicting only on that fold’s test slice
* you store OOF macro-F1 for classifier and OOF R² for regressor as final “best_score”

3. Target-mean encoding (te_*) is learned inside each fold (leak-free)

* te_crop, te_region, te_region_crop are computed only from training fold yields
* smoothing prevents overfitting on rare groups:
  smoothed = (count*mean + m*global_mean) / (count + m)
  where m = TARGET_MEAN_SMOOTHING (default 10)

---

## Feature engineering (what the models actually see)

There are 3 main types of features:

1. Raw numeric predictors (aerosol + meteo)
   Examples (from your best models):

* aod
* pm2.5 ugm-3
* precipitation percentiles
* methane / CO / surface skin temp, etc.

2. Lag yield features (direct lags)

* yield_lag1
* yield_lag2
  These capture the “inertia” of yield over time and are extremely predictive in panel forecasting.

3. Lag-derived statistics (computed from available lag columns)
   The pipeline creates stable lag summary features (even when some lags are missing), such as:

* yield_lag_last
* yield_lag_mean_2 / mean_3 / mean_5
* yield_lag_std_3 / std_5
* min/max over windows
* slope features (trend)
* momentum features (differences between last and earlier lags)
* ratio features (last / mean3)

4. Target-mean features + interactions (if enabled)

* te_crop, te_region, te_region_crop
* interactions between lag_last and te_* (differences and ratios)

Why this helps

* the lag block captures temporal dynamics
* the te_* block captures stable structural differences (some crops/regions have consistently higher/lower yields)
* interactions let the model learn “is this year above/below what’s normal for this crop/region?”

---

## Model selection strategy (what gets tried)

You train and select two separate “best models”:

1. best regressor (predict yield)
2. best classifier (predict yield_class)

Both are tuned using RandomizedSearchCV under year-forward CV.

You control how wide the search is using:

* CROP_TUNE_MODE=fast | metrics | full

You also cache results by data signature + code version unless forced.

---

## Regression task: models tried

Depending on tune mode and installed libraries, the regression candidates include:

A) HistGradientBoostingRegressor (HistGB)

* usually strong on tabular data
* handles non-linearities
* relatively robust to missingness patterns after imputation

B) MLPRegressor (neural network)

* a feed-forward neural net (scikit-learn)
* uses scaling, early stopping, and tuned architecture/hyperparams
* candidates include:

  * hidden_layer_sizes: (64,), (128,), (64,32), (128,64), (256,128)
  * activation: relu/tanh
  * alpha (L2): 1e-6 → 1e-2
  * learning_rate_init: 1e-4 → 3e-3
  * batch_size: 32/64/128

C) XGBoost regressor (optional, if installed)

* strong boosted-tree baseline with broader tuning
* included in metrics/full mode only

D) ExtraTreesRegressor (full mode only)

* strong but can be high-variance on small datasets if not constrained well

E) SVR with RBF kernel (SVR_RBF, full mode only)

* classic non-linear kernel method
* requires scaling (the pipeline enables scaling for this)
* tuned over C, gamma, epsilon

Your winner (current)

* SVR_RBF with C=10.0, gamma=auto, epsilon=0.03
* OOF R² = 0.735, OOF RMSE = 0.464

Why SVR_RBF likely won here

* dataset is small (252 rows), so high-capacity tree ensembles can overfit or become unstable fold-to-fold
* RBF SVR is a strong “smooth non-linear function approximator” and often performs well on small, dense tabular signals after scaling
* your tuned epsilon=0.03 suggests a tight fit to capture structure without chasing every noise point
* you also have strong lag/TE features, which make the underlying mapping more learnable (SVR thrives when features already contain structured signal)

---

## Classification task: models tried (two approaches)

The classifier selection is more interesting: you try two different “philosophies” and keep whichever wins macro-F1.

Approach 1: direct 3-class classifier
Candidate models include:

* HistGradientBoostingClassifier (HistGB)
* ExtraTreesClassifier (ExtraTrees)
* LinearSVC (metrics/full)
* MLPClassifier (neural network)
* XGBoost classifier (optional if installed, metrics/full)

Neural net classifier details (MLPClassifier)

* max_iter ~1200, early stopping enabled
* tuned over:

  * hidden_layer_sizes: (64,), (128,), (64,32), (128,64)
  * activation: relu/tanh
  * alpha (L2): 1e-5 → 1e-2
  * learning_rate_init: 1e-4 → 3e-3
  * batch_size: 32/64/128

Important detail for direct classifiers

* for classification, the pipeline uses a “yield-aware” wrapper so target-mean encoding te_* is learned from yield (continuous) while the classifier learns from yield_class
* this is done using y_combo = [yield, yield_class] to keep te_* leak-free inside folds

Approach 2: Regression → Tertile classification (Reg2Tertile)
This approach:

1. trains a regressor on yield
2. predicts yield_hat
3. converts yield_hat into class 0/1/2 via crop-specific tertile thresholds (or overall fallback)
4. evaluates macro-F1 on the resulting 3-class output

Your winner (current)

* Reg2Tertile(HistGB)
* OOF macro-F1 = 0.594, OOF accuracy = 0.708

Why Reg2Tertile(HistGB) likely won

* yield is fundamentally continuous; learning “ordering” is easier than learning 3 buckets directly
* the thresholding step makes the final output less sensitive to small numeric noise (you only need to land in the right tertile)
* HistGB with shallow depth + regularization is a good fit for small data and structured signals:

  * max_depth=2
  * max_leaf_nodes=15
  * min_samples_leaf=10
  * learning_rate=0.05
  * l2_regularization=0.1
    This combination is very “anti-overfit”: it forces the model to learn only strong patterns, which typically improves generalization on small panel data.

---

## So what about the neural networks?

Neural networks are included as first-class candidates for both tasks:

* MLPRegressor in regression candidates
* MLPClassifier in classification candidates

Why they might not win (common reasons in your specific setting)

1. data size is small

* MLPs often shine when you have lots of examples; 252 rows is small for a NN
* even with early stopping, the model can be sensitive to fold splits and random variation

2. tabular data + mixed signal types

* boosted trees and kernel methods often outperform MLPs on classical tabular ML unless you do careful feature scaling, architecture tuning, and often more data

3. lag + target-mean features already “linearize” the problem

* once you add lag statistics and te_* baselines, a lot of complexity is already captured
* tree models or SVR can exploit that structured feature space very efficiently

When MLPs can win in this project

* if you expand data (more years, more regions, more crops)
* if you add richer temporal context (more lag steps, rolling windows)
* if you standardize feature sets and reduce noise/missingness
* if you tune MLP more aggressively (wider architecture search, longer training, multiple random seeds)

How to force-try MLP more seriously

* set CROP_TUNE_MODE=full
* consider disabling overly-wide feature sets if missingness is high (or force FEATURE_SET=core/mid)
* optionally run multiple seeds and average performance (not implemented by default, but easy to add later)

---

## Feature-set auto-selection (why regression used more features than classification)

You’ll notice:

* classifier selected a stricter missing threshold (0.25) and ended up with fewer “raw” numeric features
* regressor selected a more permissive missing threshold (0.6) and used more “raw” numeric features

Why this happens

* macro-F1 for 3-class classification can degrade when you add noisy/missing-heavy features
* regression can still benefit from additional weak predictors because the model can average signal across continuous outputs
* the auto-selection step explicitly tries multiple missingness thresholds and keeps the one that performs best in CV

---

## Forecasting + scenarios (what the pipeline does after training)

After training the best regressor, the pipeline also runs forecasting:

1. Leak-free walk-forward backtest

* simulates “train up to year t, predict year t+1”
* reports RMSE/R²/MAE/MAPE
* produces year-by-year metrics for recent years to show stability

2. Recursive multi-year forecast (10 years)

* if lag features exist, forecasting is recursive:
  predict next year → feed prediction as lag input → predict next year → repeat

3. Scenario forecasting

* generates alternate futures by modifying aerosol conditions:

  * baseline
  * clean_air
  * polluted
    This helps answer: “if air improves/worsens, how does yield change?”

---

## Interpretability: permutation importance

You compute permutation importance on the fitted pipeline, so the importance reflects the entire preprocessing + model together.

For regression:

* scoring defaults to model.score (R²)

For classification:

* scoring uses macro-F1

This outputs:

* feature, importance_mean, importance_std
  in outputs/permutation_importance.csv

Use this to explain results:

* “lag_last and lag_mean_3 were most important, meaning yield is strongly autocorrelated”
* “pm2.5/aod features matter in some folds, indicating aerosols add predictive power beyond lags”

---

## Caching + reproducibility controls (important knobs)

Environment variables you can set:

Tuning breadth

* CROP_TUNE_MODE=fast | metrics | full

Caching

* CROP_USE_CACHE=1 (default) to reuse saved best models
* CROP_FORCE_RETRAIN=1 to ignore cache and retrain

Feature set selection

* CROP_FEATURE_SET=auto | core | mid | full

Tertile behavior (classification wrapper)

* CROP_TERTILES_MODE=fixed (default) uses thresholds computed once
* CROP_TERTILES_MODE=strict recomputes thresholds per fit

Target-mean encoding

* CROP_USE_TARGET_MEAN=1 (default)
* CROP_TARGET_MEAN_SMOOTHING=10.0 (default)

Debug

* CROP_DEBUG_SANITY=1 prints data sanity checks

---

## How to explain “why these winners” in 5 lines (interview-style)

Regression winner (SVR_RBF)

* small dataset + strong engineered features
* scaling + RBF kernel captures smooth non-linear relationships well
* SVR can generalize better than high-variance ensembles on limited data
* OOF R² 0.735 shows stable year-forward generalization
* backtest R² 0.802 confirms real forward-time predictability

Classification winner (Reg2Tertile(HistGB))

* yield is continuous; regressing then binning often beats direct 3-class learning
* HistGB with shallow depth + L2 is highly regularized (good for small data)
* macro-F1 0.594 OOF is much stronger than baseline 0.167
* approach is naturally aligned with “yield tiers” defined by tertiles

Neural networks (MLP)

* included and tuned
* likely not best because 252 rows is small for NN reliability
* trees/SVR tend to dominate on small tabular data with engineered features

---

## Troubleshooting

If you think something “isn’t updating”

* set `CROP_FORCE_RETRAIN=1` to ignore cached models

If training is too slow

* set `CROP_TUNE_MODE=metrics` or `fast`
* reduce outer jobs: `CROP_OUTER_JOBS=2`

If you see missing-data issues

* force a stricter feature set: `CROP_FEATURE_SET=core`
* or keep auto but reduce missing threshold candidates (optional code change)

---

If you want, paste your `outputs/permutation_importance.csv` top 20 rows here and I’ll add a “Key drivers of yield” section that is specific to your run (not generic), including what the importance implies and how you’d present it in the report.
