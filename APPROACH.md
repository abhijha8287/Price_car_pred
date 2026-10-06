# Approach

This note explains how we went from the raw CSVs to the predictions in `submission.csv`, and why we made the choices we did.

## The problem

We have about 3,200 used car listings in `train.csv`, each with a price, and 802 listings in `test.csv` without one. The job is to predict the price for the test cars.

Each listing has the brand, model name, model year, mileage, fuel type, exterior and interior colour, accident history, clean title, engine size, number of cylinders and transmission type.

## What we noticed in the data

A few things stood out when we first looked at the data, and they shaped most of the decisions below.

**Prices are very skewed.** The median car sells for about $31,000, but a handful go well past $500,000 and one is listed at almost $3 million. If we train on raw prices, those few cars dominate the error and the model ends up chasing them. So we train on the log of the price and convert back to dollars at the end. That way, being 10% off on a $10k car counts about the same as being 10% off on a $200k car.

**Some columns are text that should be numbers.** Mileage comes in as `"50,992 mi."` and price as `"$28,000"`. Both needed cleaning before anything else.

**Missing values sometimes carry meaning.** When both fuel type and engine size are blank, the car is almost always electric (lots of Teslas, for example). Rather than filling these in with a guess, we added an "is electric" flag.

**Many test cars are models we never saw in training.** About 30% of the model names in the test set don't appear in the training set at all, because model names are very specific (`"Q50 3.0T Premium"`, `"SL-Class SL 550"`). A model that only learns from the full name has nothing to go on for those cars. To help it, we also pulled out the first word of the model name (`Q50`, `SL-Class`) and a brand-plus-model group, so an unseen trim can still borrow from similar cars.

**Colours are messy.** There are 275 different exterior colour names, such as "Burnished Bronze Metallic" and "Glacier White". We kept the original names but also mapped each one to a basic colour (bronze/brown, white, and so on).

## Features we built

- Mileage as a number
- Age of the car (2025 minus model year)
- Miles driven per year, which tells us whether a car has been driven hard
- Is-electric flag
- Accident and clean title turned into simple yes/no flags
- Short model name, and brand + short model name
- Simplified exterior and interior colour
- Flags for trim words in the model name that usually signal a pricier car: AMG, Raptor, Hellcat, Denali, Platinum, Turbo, quattro, xDrive and a few others

## Choosing a model

We compared two gradient boosting libraries, LightGBM and CatBoost, using 5-fold cross-validation. The data was split into five parts; each part was held out once while the model trained on the other four, so every car got a prediction from a model that had never seen it.

| Model | RMSLE (lower is better) |
|---|---|
| LightGBM | 0.49 |
| CatBoost | 0.31 |
| Mix of both | worse than CatBoost alone |

CatBoost won by a wide margin. That isn't too surprising, because this dataset has a lot of text categories with many possible values (brands, models, colours). CatBoost handles those natively and is careful not to leak the target while doing it. Our quick LightGBM setup probably overfit on those columns. Mixing the two models only pulled the score down, so we dropped LightGBM.

## The final model

- CatBoost, 2,000 trees, learning rate 0.03, depth 6
- Trained on log(price)
- 5 folds × 3 random seeds = 15 models, and the test prediction is the average of all 15

Averaging over seeds and folds smooths out some of the randomness and usually gives slightly more stable predictions than a single model.

## How well it works

These numbers come from cross-validation, so they are an honest estimate of how the model does on cars it hasn't seen.

| Metric | Value |
|---|---|
| R² (on log price) | 0.868 |
| RMSLE | 0.310 |
| Mean absolute error | $11,205 |
| Mean absolute % error | 21.3% |
| RMSE | $70,549 |

In plain terms, a typical prediction lands within about 20% of the real price. The RMSE looks scary, but it is driven almost entirely by the few very expensive cars. The mean absolute error is a better picture of how far off a normal prediction is.

The model leans most on mileage, age, engine size and brand, which matches how people actually price used cars.

## What could be improved

- **Outliers.** The ~$3M listing and a few others are hard to predict and may even be data entry mistakes. Looking at these by hand could help.
- **Better parsing of the model name.** Our keyword list is short and hand-picked. A more systematic way to pull out trim levels would probably help with unseen models.
- **More data.** 3,200 cars across 57 brands and 1,600+ models is thin. Many models appear only once.
- **Tuning.** We used sensible default settings and didn't run a proper hyperparameter search.
