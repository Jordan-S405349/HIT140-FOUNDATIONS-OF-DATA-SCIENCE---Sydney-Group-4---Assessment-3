# HIT140 Foundations of Data Science — Sydney Group 4 — Assessment 3

# FIFA World Cup 2026 — Linear Regression Project

Project Overview
This project investigates FIFA World Cup 2026 match data using two related but distinct linear regression tasks.
The analysis uses data sourced from:
- FIFA Official Website
  https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/statistics
- The Stats Don't Lie
  https://www.thestatsdontlie.com/football/world-cup-2026/
- FBref
  https://fbref.com/en/
All explanatory variables are based on information available before the match being predicted to avoid target leakage.

## Team Member

| Student ID | Full Name |
|---|---|
| S403360 | Syed Suleman Uddin Farhad |
| S405349 | Jordan Then Ryan |
| S403295 | Ujjwol Neupane |
| S403036 | Saugat Shrestha |

## Objective 2.1 — Goal Difference Prediction
The aim of Objective 2.1 is to predict the goal difference between two opposing teams in FIFA World Cup 2026 matches.
Dataset Requirements
- 104 rows
- One row represents one World Cup 2026 match
- Target variable: goal_difference
- Exactly 8 explanatory variables
- All predictors must be known before kickoff

### Explanatory Variables
1. X1_ppg_diff
2. X2_avg_goals_for_diff
3. X3_avg_goals_against_diff
4. X4_win_rate_diff
5. X5_clean_sheet_rate_diff
6. X6_scored_2plus_rate_diff
7. X7_previous_goal_margin_diff
8. X8_host_advantage_diff

### Models Evaluated
- Full OLS
- Reduced OLS
- Ridge Regression
- Lasso Regression

### Evaluation Metrics
- R²
- Adjusted R² where applicable
- RMSE
- MAE
- Cross-validation performance

## Objective 2.2 — Goals Scored Prediction
The aim of Objective 2.2 is to predict the number of goals scored by one team in a single World Cup 2026 match.
## Dataset Requirements
- 208 rows
- Each match appears twice, once for each team
- Target variable: goals_scored
- Exactly 8 explanatory variables
- All predictors must be known before kickoff
## Explanatory Variables
1. X1_team_ppg
2. X2_team_avg_goals_for
3. X3_opponent_avg_goals_against
4. X4_host_nation_flag
5. X5_team_draw_rate
6. X6_team_failed_to_score_rate
7. X7_opponent_loss_rate
8. X8_opponent_clean_sheet_rate
## Models Evaluated
- Full OLS
- Reduced OLS
- Ridge Regression
- Lasso Regression
## Evaluation Metrics
- R²
- Adjusted R² where applicable
- RMSE
- MAE
- Cross-validation performance

## Analysis Workflow
Both objectives follow a similar modelling workflow:
1. Data loading and validation
2. Descriptive statistics
3. Histograms and boxplots
4. Inferential statistics
5. Correlation analysis
6. VIF multicollinearity check
7. Full OLS regression
8. Reduced OLS regression
9. Ridge regression
10. Lasso regression
11. Cross-validation
12. Model comparison
13. Residual diagnostics
14. Final model recommendation
15. Limitations and discussion

## File Description
- Objective_2_1_Dataset.csv — Dataset used for Linear Regression Objective 2.1.
- Objective_2_2_Dataset.csv — Dataset used for Linear Regression Objective 2.2.
- linearreg_objective_2_1.ipynb — Jupyter Notebook containing the Objective 2.1 analysis.
- linearreg_objective_2_2.ipynb — Jupyter Notebook containing the Objective 2.2 analysis.
- linearreg_objective_2_2.py — Python script version of the Objective 2.2 analysis.
- README.md — Project documentation and instructions.
- LICENSE — Repository licence information.
- .gitignore — Defines files Git should ignore.
- .DS_Store — macOS system metadata file; this file is not part of the analysis and can be removed or ignored.

## Running the Project
Open the .ipynb files in:
- GitHub Codespaces
- Jupyter Notebook
- JupyterLab
- Visual Studio Code
Make sure the related CSV file is stored in the same directory as the notebook.

## Main Python Libraries
- pandas
- numpy
- matplotlib
- scipy
- statsmodels
- scikit-learn

## Data Quality and Modelling Rules
To maintain scientific rigour:
- No same-match statistics are used as predictors.
- Only information available before kickoff is used.
- The eight explanatory variables are defined before modelling.
- Model performance is evaluated using cross-validation.
- Multicollinearity is checked using correlation analysis and VIF.
- Residual diagnostics are used to assess linear regression assumptions.
- Model recommendations are based on predictive performance and interpretability.

## Key Limitations
- Early tournament matches have limited previous World Cup 2026 form.
- Football outcomes contain unpredictable events not included in the models.
- The sample size is relatively small.
- Some predictors may be correlated.
- Linear regression may not capture all nonlinear relationships in football.
- In Objective 2.2, each match contributes two team-level observations, so observations are not completely independent.

## Contribution and Collaboration
Team members contribute to:
- data preparation
- variable design
- exploratory data analysis
- regression modelling
- visualisation
- model evaluation
- report writing
- documentation
