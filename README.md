**TITLE**
A Boosting-Based Framework with LLM-Assisted Decision Support for Accurate Vehicular Insurance Prediction (BF-LADS-VIP)

**Description**
This project develops BF-LADS-VIP (A Boosting-Based Framework with LLM-Assisted Decision Support for Accurate Vehicular Insurance Prediction), integrating advanced boosting algorithms, including XGBoost, LightGBM, Gradient Boosting, and CatBoost, to predict motor insurance premiums based on vehicle and insurance-related attributes. The framework incorporates Large Language Model (LLM)-assisted decision support using LLaMA 3 to generate interpretable explanations of model predictions and risk profiles, supporting underwriters in making informed decisions. The study systematically evaluates model performance, predictive accuracy, and generalization under consistent experimental conditions, emphasizing accurate premium prediction, explainability, and human-in-the-loop decision-making to facilitate transparent, scalable, and trustworthy AI adoption in vehicular insurance underwriting.

**Dataset Information**
The model was trained on the Kaggle Motor Insurance dataset comprising 3,998 vehicle records. Of these, 790 motorcycle-related records were filtered out to maintain the study's focus on other vehicle categories, leaving 3,208 records for processing and model development. The remaining dataset was used for data preprocessing, feature selection, model training, and evaluation to predict vehicular insurance premiums and support risk-informed underwriting decisions.

**Code Overview**
The proposed BF-LADS-VIP pipeline integrates the following components:

1. Data Preprocessing
- Uses the Kaggle Motor Insurance dataset containing 3,998 vehicle records.
- Filters out 790 motorcycle-related records, leaving 3,208 records for processing and model development.
- Prepares relevant vehicle and insurance-related attributes for predicting the target variable, policy_premium.
- Applies appropriate data cleaning, feature preparation, and train–test splitting procedures.
2. Model Training (Boosting Algorithms)
- Trains and compares four boosting-based regression models independently: XGBoost, LightGBM, Gradient Boosting Regressor, and CatBoost.
- Uses model-specific configurations and hyperparameters to optimize predictive performance.
- Evaluates model predictions against actual insurance premiums to assess learning performance and generalization.
- Stores model training results for comparative analysis and subsequent decision support.
3. Performance Evaluation
- Computes regression performance metrics, including R^2, Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE), where appropriate.
- Compares training and testing performance to assess predictive accuracy and potential overfitting.
- Identifies the best-performing boosting model based on the evaluation results.
- Analyzes the models' ability to estimate motor insurance premiums accurately and consistently.
4. Risk Profiling and Premium Prediction
- Develops vehicle risk profiles using relevant vehicle and insurance-related attributes.
- Generates estimated insurance premiums using the trained boosting models.
- Stores prediction results and risk profiles in a PostgreSQL database for retrieval and analysis.
- Supports underwriters by presenting data-driven premium estimates and relevant risk information.
5. LLM-Assisted Decision Support and Explainability
- Integrates the LLaMA 3 model through Ollama to generate natural-language explanations of premium predictions and risk profiles.
- Translates model outputs and relevant input attributes into understandable insights for insurance underwriters.
- Supports human-in-the-loop decision-making by allowing underwriters to interpret model-generated recommendations alongside their professional judgment.
- Aims to improve transparency and usability without replacing human underwriting authority.
6. Visualization and Comparative Analysis
- Presents comparative performance results for XGBoost, LightGBM, Gradient Boosting, and CatBoost.
- Visualizes actual versus predicted premiums and regression performance metrics.
- Examines training and testing performance to assess generalization.
- Integrates prediction results, vehicle risk profiles, and LLM-generated explanations into the decision-support interface.
- Supports systematic analysis of predictive accuracy, interpretability, and practical applicability in motor insurance underwriting.
Implementation technologies: Python, Scikit-learn, XGBoost, LightGBM, CatBoost, PostgreSQL, Streamlit, and Ollama with LLaMA 3.

**Usage Instructions**
1. Dataset Preparation
- Download the Motor Insurance dataset from Kaggle.
- Load the dataset containing 3,998 vehicle records and filter out 790 motorcycle-related records, leaving 3,208 records for processing.
- Configure the dataset path and database connection in the application as required.
- Prepare the relevant vehicle and insurance attributes, using policy_premium as the target variable for premium prediction.
2. Run Training and Evaluation
- Launch the Streamlit application using the Python environment with the required dependencies installed.
- Execute the training workflow to train and evaluate the four boosting models: XGBoost, LightGBM, Gradient Boosting Regressor, and CatBoost.
- Select or configure the model parameters as required by the implementation.
- Use the LLM-assisted decision-support functionality through Ollama with LLaMA 3 to generate explanations of premium predictions and risk profiles.
3. Outputs Generated
- Model training and evaluation results.
- Comparative regression performance metrics, including R^2, RMSE, MAE, and MAPE, where applicable.
- Actual-versus-predicted premium comparisons.
- Vehicle risk profiles and predicted insurance premiums.
- Stored model training results and prediction records in PostgreSQL.
- LLaMA 3-generated natural-language explanations to support underwriting decisions.
- Comparative analysis of model accuracy, generalization, and decision-support capabilities.

**Requirements**
| Component           | Version / Technology |
|---------------------|-----------------------|
| IDE                 | Visual Studio Code 1.143.1 |
| Operating System    | Windows 11 |
| Python              | 3.13.7|
| Database Management | PostgreSQL with pgAdmin 4 (v9.8) |
| Machine Learning    | Scikit-learn, XGBoost, LightGBM, CatBoost |
| LLM Integration     | Ollama with LLaMA 3 |
| Application Framework | Streamlit |
| Data Processing     | NumPy, Pandas |
| Visualization       | Matplotlib |

Install dependencies:
bash
python -m pip install numpy pandas scikit-learn matplotlib seaborn xgboost lightgbm catboost streamlit psycopg2-binary
ollama pull llama3

**Methodology Summary**

| Component       | Description |
|------------     |-------------|
| Dataset         | Kaggle Motor Insurance dataset |
| Initial Records | 3,998 vehicle records |
| Data Filtering  | 790 motorcycle-related records excluded |
| Processed Dataset | 3,208 vehicle records |
| Prediction Target | `policy_premium` |
| Base Models       | XGBoost, LightGBM, Gradient Boosting Regressor, CatBoost |
| Training Strategy | Supervised boosting-based regression |
| Data Preprocessing| Data cleaning, feature preparation, and encoding where required |
| Dataset Split     | Training/testing split according to the implemented experimental configuration |
| Hyperparameter Optimization | Model-specific hyperparameter configuration and tuning |
| Evaluation Metrics | R², RMSE, MAE, and MAPE where appropriate |
| Risk Profiling     | Vehicle risk profile generation using relevant vehicle and insurance attributes |
| Premium Prediction | Estimated motor insurance premiums using trained boosting models |
| LLM Integration    | LLaMA 3 through Ollama |
| Decision Support   | LLM-assisted natural-language explanations of premium predictions and risk profiles |
| Database           | PostgreSQL |
| Application Framework | Streamlit |
| Development Environment | Visual Studio Code 1.143.1 |
| Operating System   | Windows 11 |
| Python Version     | 3.13.7 |
| Database Management Tool | pgAdmin 4 (v9.8) |
| Data Processing Libraries | NumPy, Pandas |
| Machine Learning Libraries | Scikit-learn, XGBoost, LightGBM, CatBoost |
| Visualization      | Matplotlib, Seaborn |
| Explainability     | LLM-generated explanations to support human-in-the-loop underwriting decisions |
| Research Objective | Accurate, interpretable, and scalable vehicular insurance premium prediction and underwriting decision support |

**Performance Summary**
| ID | Model Name | Training R² Score | Training MAE | Training MSE | Training RMSE | Testing R² Score | Testing MAE | Testing MSE | Testing RMSE |
|----|-----------|-------------------|--------------|----------------|----------------|------------------|-------------|---------------|--------------|
| 14 | XGBoost Model | 0.9834 | 5683.0872 | 584836351.18 | 24183.39 | 0.9656 | 8471.0425 | 1285405663.22 | 35852.55 |
| 16 | LightGBM Model | 0.7895 | 16260.1756 | 7416192709.61 | 86117.31 | 0.8440 | 19047.9802 | 5829675230.87 | 76352.30 |
| 18 | Gradient Boosting Model | 0.9969 | 5210.5994 | 109153311.95 | 10447.64 | 0.9815 | 7570.7199 | 691013742.81 | 26287.14 |
| 22 | CatBoost Model | 0.9988 | 3745.5623 | 41269149.74 | 6424.10 | 0.9676 | 7312.2879 | 1211846119.56 | 34811.58 |

- Based on the reported testing results, Gradient Boosting achieves the highest testing R^2 score (0.9815) and lowest testing RMSE (26,287.14), while CatBoost achieves the lowest testing MAE (7,312.2879). Therefore, Gradient Boosting performs best overall on these testing metrics, whereas CatBoost has the strongest training performance.

**Visualization Examples**
- Confusion Matrices – class-wise accuracy comparison for training and testing sets
- Training/Testing Loss & Accuracy Curves – convergence analysis across epochs
- Five-Fold Cross-Validation Plots – robustness and generalizability assessment
- Grad-CAM Heatmaps – highlight retinal regions (hemorrhages, exudates, microaneurysms) influencing predictions

**Evaluation Environment**
Developed and tested on:
- Operating System: Windows 11
- Development Platform: Visual Studio Code 1.143.1
- Python Version: 3.13.7
- Machine Learning Frameworks: Scikit-learn, XGBoost, LightGBM, CatBoost
- LLM Integration: Ollama with LLaMA 3
- Application Framework: Streamlit
- Database Management: PostgreSQL with pgAdmin 4 (v9.8)
- Data Processing Libraries: NumPy, Pandas
- Visualization Libraries: Matplotlib, Seaborn
- Database Connectivity: psycopg2-binary
- Prediction Methodology: Boosting-based regression for vehicular insurance premium prediction
- Decision-Support Methodology: LLM-assisted interpretation of premium predictions and vehicle risk profiles

**Results and Discussion**
- Gradient Boosting achieved the best overall testing performance, with an R^2 score of 0.9815 and an RMSE of 26,287.14, demonstrating strong predictive accuracy for vehicular insurance premium estimation.
- CatBoost achieved the strongest training performance, with an R^2 score of 0.9988 and an RMSE of 6,424.10, although its testing performance was lower, indicating a larger generalization gap.
- XGBoost demonstrated competitive performance, achieving a testing R^2 score of 0.9656 and an RMSE of 35,852.55.
- LightGBM recorded the lowest performance among the four models, with a testing R^2 score of 0.8440 and an RMSE of 76,352.30.
- Comparative evaluation using R^2, MAE, MSE, and RMSE highlighted differences in predictive accuracy and generalization across the boosting algorithms.
- Feature importance and residual visualizations support the analysis of influential vehicle and insurance-related attributes, prediction errors, and potential model biases.
- The integration of LLaMA 3 through Ollama provides natural-language explanations of premium predictions and vehicle risk profiles, supporting transparent, human-in-the-loop underwriting decisions.
- The BF-LADS-VIP framework is designed as a decision-support solution for motor insurance underwriting, helping underwriters assess predicted premiums and relevant risk information while retaining human judgment and final decision-making authority.

**Limitations**
- The current analysis is restricted to single public dataset of 3,208 filtered records without claims history, telematics, or behavioural factors. 
- External validation is required to assess generalizability, subgroup fairness, computational cost, LLM latency, data privacy, regulatory compliance, and explanation reliability. 
- Accordingly, BF-LADS-VIP is presented as a decision-support prototype rather than a production-ready underwriting system.

**Conclusion**

- The Gradient Boosting model shows strong and well-balanced predictive performance across all major evaluation metrics. 
- The R^2 value of 0.9815 indicates that the model explains more than 98% of the total variation in insurance premium values, demonstrating a very strong relationship between the predicted and actual premiums. 
- The Mean Absolute Error (MAE) of 7,570.71 shows that the average difference between the predicted and actual premium values is reasonably low, confirming the model’s high prediction accuracy. 
- The Mean Squared Error (MSE) of 691,013,742.8 represents the squared size of error in prediction and demonstrates that the model is capable of controlling error effect, especially in high premium values, where the error effect is inherently high. 
- In the same manner, the Root Mean Squared Error (RMSE) of 26,287.14 indicates that the model has a stable error control towards the bigger error in prediction, particularly in policies with high premium values. 
- These findings affirm that the model is valid and can be trusted to be useful in predicting insurance premiums of vehicles.
- However, the precision of prediction can be further increased by adding more behavioural and telematics-based variables. Future research may take into account hybrid strategies that involve using Gradient Boosting with deep learning models to improve performance. 
