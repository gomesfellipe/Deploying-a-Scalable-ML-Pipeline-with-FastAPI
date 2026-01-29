# Model Card

For additional information see the Model Card paper: <https://arxiv.org/pdf/1810.03993.pdf>


## Model Details

This model is a binary classifier trained to predict whether an individual's income is ">50K" or "<=50K" based on the UCI Census Income dataset. The pipeline uses a `StandardScaler` followed by a `LogisticRegression` classifier (scikit-learn). The model is trained in `train_model.py` and serialized with pickle.

Algorithm: Logistic Regression (lbfgs solver)
Features: One-hot encoded categorical features + continuous numeric features
Output: Binary prediction mapped to ">50K" or "<=50K"


## Intended Use

Primary intended use is as an instructional example for a scalable ML pipeline with FastAPI. It can be used for batch or online inference in a controlled environment. This model should not be used for high-stakes, real-world decisions without additional validation, monitoring, and fairness assessment.


## Training Data

Training data comes from `data/census.csv` (UCI Census Income dataset). The dataset includes demographic and employment-related attributes such as workclass, education, occupation, relationship, race, sex, and native-country. The target label is `salary` (binary: ">50K" or "<=50K").

Train/Test split: 80/20 stratified split by `salary`.


## Evaluation Data

Evaluation is performed on the 20% holdout test split from the same dataset (`census.csv`) using stratified sampling to preserve the label distribution.


## Metrics

The following metrics are computed on the holdout test set:

- Precision: 0.7365
- Recall: 0.6186
- F1: 0.6724

Metrics are computed using scikit-learn with `zero_division=1` for stability.


## Ethical Considerations

The dataset includes sensitive attributes (e.g., race and sex). Using these features can encode or amplify existing biases. Any deployment should include fairness analysis, bias mitigation strategies, and clear governance on how predictions are used.


## Caveats and Recommendations

- The model was trained on a historical dataset and may not generalize to current labor market conditions.
- The model is intended for learning and demonstration, not for production decision-making.
- Consider feature scaling, hyperparameter tuning, and model monitoring if used beyond the course context.
