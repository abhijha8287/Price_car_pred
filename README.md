# Used Car Price Prediction

A regression model that predicts used car prices. It's trained on `train.csv` and makes predictions for the cars in `test.csv`.

The model is CatBoost trained on log(price). In cross-validation it gets an R² of about 0.87, and a typical prediction is within roughly 21% of the actual price.

For the full reasoning behind the features and model choice, see [APPROACH.md](APPROACH.md).

## What's in here

| File | What it is |
|---|---|
| `train.csv` | 3,207 cars with known prices |
| `test.csv` | 802 cars we need to price |
| `notebook.ipynb` | The full pipeline: cleaning, features, cross-validation, predictions |
| `submission.csv` | Predicted price for every car in `test.csv` (`id`, `price`) |
| `APPROACH.md` | How and why we built it this way |

## Running it

You need Python 3.10 or newer.

```bash
# set up an environment
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn catboost jupyter

# run the notebook top to bottom
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

Or just open `notebook.ipynb` in Jupyter or VS Code, pick the `.venv` kernel and run all cells.

A full run takes a few minutes on a laptop, since it trains 15 models (5 folds × 3 seeds). When it finishes, `submission.csv` is rewritten with fresh predictions.

## Results

From 5-fold cross-validation on the training data:

| Metric | Value |
|---|---|
| R² (log price) | 0.869 |
| RMSLE | 0.309 |
| Mean absolute error | $11,115 |
| Mean absolute % error | 21.2% |

The most important features turned out to be mileage, age, engine size and brand.

## Notes

- Car age is calculated as `2026 - model_year`. The reference year also feeds into miles driven per year, so changing it shifts predictions a little.
- Predicted prices are rounded to the nearest dollar.
