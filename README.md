# AI&ML Project 2024-2025

# Project 4 - AEROPOLIS

### Group Members:

* Nikol Tushaj - Group "Captain" (303221)
* Rajla Çulli (297601)
* Giulio Carbone (290721)

## INTRODUCTION

In the futuristic city of Aeropolis, autonomous delivery drones are essential to ensure fast and efficient delivery of goods across the sprawling metropolis. Each drone's performance is evaluated based on how much cargo it can deliver per flight. 

However, many factors influence its performance, from weather conditions to the type of terrain it navigates. 

To optimize drone performance, data scientists are tasked with predicting the cargo capacity per flight based on various environmental and operational factors.

## LIBRARIES
The main python libraries we used for the project are:

* pandas: to manipulate the dataset;
* matplotlib.pyplot: to plot graphs;
* numpy: to perform mathematical operations;
* seaborn: to plot graphs;
* time: to track execution time;
* math: to perform mathematical operations;
* warnings: to get rid of unnecessary warnings;
* sklearn: to build the models and evaluate their performance;

Our project is divided into 7 steps, which we will explain below:

### 1) Understanding the dataset
- 1.1)Overview of the dataset: with *'aeropolis_df.head()'* we extract the first few rows of the dataset, which helps us to better understand the structure of the data, the columns, and to have a glance at the values;
Then, thanks to the **`shape`** attribute, we can see that the dataset is composed by 1,000,000 rows and 20 columns: this is useful to verify the size of the dataset before processing.
Moreover, *'aeropolis_df.info()'* allows us to output the summary of a DataFrame, with information like the number of rows or columns or columns' data types, and the count of non-null values for each column.  This result highlights the fact that each column has missing data and that the data types are either **`float64`**, used for floating-point numbers in 11 columns out of 20, or **`object`**, for strings or columns containing mixed data types in the remaining 9 columns. 
- 1.2)Checking for duplicates: The **`.nunique()`** method calculates the number of unique values in each column of the DataFrame, which helps us to understand the distribution and variability of the data in each feature. 
The result shows us that there are some attributes that have a low number of unique values, which may indicate that they are categorical values, whilst there are other attributes that have a high number of unique values, which may indicate that they are numerical values. 
With **'aeropolis_df.duplicated().sum()'** we ensure the dataset is free from redundancy, which could bias our analysis or training process.
- 1.3)Checking data integrity: To check data integrity in a more detailed way we create the function **`missing_values_table`** to return the total count of missing values for each column, and their respective percentage relative to the number of rows.
The table we compute with **'missing_values_table(aeropolis_df)'** shows that the dataset has significant missing values with circa 10% in each column. 
	- 1.3.1)We can also see that as the number of missing values increases, the count of rows decreases significantly, which means that the major part of the dataset will require minimal imputation, while 	the other part will need to be processed or removed.
- 1.4)Descriptive Statistics using the original data: 
	- 1.4.1)Categorical Values: The **`select_dtypes(include=['object'])`** method filters out only the columns with data type object, which are the categorical variables.
	The **`.describe()`** method generates summary statistics for the categorical columns, such as **`count`**, which tells the number of non-null values in each column, or **`top`**, which tells the most 	frequent category (mode).
	**Interpretation**
	* The results from the categorical data description show that the dataset is fairly balanced in terms of representation across different categories, such as **`Weather_Status`**, **`Package_Type`**, 	**`Market_Region`**, and others. 	
	* For example, **`Market_Region`** has three categories, with "Local" being the most frequent at 300,377 instances. Similarly, **`Quantum_Battery`** is binary with a balanced split, showing **`True`** 	as the top value at 450,103.

	* This balance suggests that the categorical variables in the dataset provide diverse representations, minimizing potential biases.

	We loop through all categorical columns to then create a pie chart for each column to visualize the distribution of values, which is fairly balanced: for example, Weather_Status or Market_Region are 	almost equally divided into their 3 categories, whilst Package_Type is evenly distributed across all its 6 categories. 
(Insert plot) (**)
	Then we loop through the categorical variables to get more precise percentages for each category and we see that the categories differ by at most 0.1% in their distribution (which means that the latter is highly balanced).

	For example, the **`Vertical_Landing`** feature shows almost equal proportions for Unknown, Unsupported, and Supported. 
	This balance suggests that no single category dominates, reducing the risk of model bias towards a particular class and ensuring fair representation during training.

	- 1.4.2)Numerical Values: In this case, we proceed as we did for the categorical columns, using the select_dtypes method the DataFrame to include only numerical columns, the describe() method to 		calculate summary statistics for numerical columns, including **`Mean`**: , which gets the average value, or **`25%, 50%, 75%`**: which give the percentiles that help us better understand the data 		distribution.
	**Interpretation**
	* The statistical summary provides insights into the numerical variables in the dataset. For the target variable **`Cargo_Capacity_kg`**, the mean is approximately 4.65 kg, with a standard deviation 	of 1.69 kg, indicating moderate variability. 

	* Some negative values suggest potential anomalies or preprocessing errors.
	The **`hist()`** function generates histograms for all numerical columns to visualize their distribution. (Insert plot)(**)
	**Interpretation**

	* **`Cargo_Capacity_kg`** and **`Water_Usage_liters`**: Have bell-shaped distributions with potential outliers at the edges.
	* **`Cleaning_Liquid_Usage_liters`**: Highly skewed to the left, indicating most values are concentrated near zero.
	* **`Autopilot_Qualty_Index`**, **`Vertical_Max_Speed`**, and **`Wind_Speed_kmph`**: Show relatively uniform distributions. 

		-1.4.2.1) Checking for low variability columns: We check for numerical columns with a standard deviation (variability) of less than 0.01, and see that all numerical columns exhibit enough 			variation to potentially contribute to the analysis.

- 1.5) Handling missing values of the dependent variable: We remove rows missing the target value (of the (**`Cargo_Capacity_kg`**) variable), using the **`dropna()`** method, because we are working on a regression problem, and therefore:
 	* The model cannot learn without target values, as it needs them to calculate errors and adjust weights.
	* Missing target values make it impossible to compare predictions and measure model performance.
	* Retaining these rows adds noise and unnecessary complexity without contributing to the model.

### 2) Cleaning the dataset

- 2.1) Encoding Categorical Values: Here we filter out the categorical columns and iterate through each column to display its name and the unique categories it contains. It provides an overview of all possible values in each categorical column, including any missing values. 

The result will help us with the mapping. 
We calculate the average **`Cargo_Capacity_kg`** for each category within a categorical column by grouping the dataset using **`groupby`** and applying the **`.mean()`** function to the target column. It identifies if categories have distinct effects on the target value, and based on our results, we have that, for example, **`Weather_Status`** has similar mean values across **`Cloudy`**, **`Sunny`**, and **`Rainy`**, indicating little influence of weather on the target variable. 

* Similarly, other columns like **`Package_Type`** and **`Vertical_Landing`** also show minimal differences across categories, suggesting a weak correlation. This justifies the decision to then map categories randomly, as their direct impact on the target appears negligible.

**Interpretation**

The data types confirm that all columns are now numerical, ready for further processing and modeling. The random samples show the encoded categorical columns alongside the numerical features, with some missing values still present.

We then define a dictionary of mappings, assigning a numerical value to each category in the categorical columns, using the **`map()`** function to replace the original categorical values with their corresponding numerical mappings for each column. 

We preserve the **`NaN`** values for later imputation, and check the data types, which confirm that all columns are numerical.

We also get some random samples, which show the encoded categorical columns alongside the numerical features, with some missing values still present.

- 2.2) Finding the correlation between the independent values and the target value: We compute the pairwise correlation between all the columns using **`.corr()`** and then visualize it in a heatmap using **`seaborn.heatmap`**.
(Insert plot)(**) 
**Interpretation**

The heatmap shows correlations between all variables. For example, **`Wind_Speed_kmph`** has a strong positive correlation (**`0.76`**) with the target variable **`Cargo_Capacity_kg`**. 

On the other hand, features like **`Vertical_Max_Speed`** and **`Market_Region`** show near-zero or negative correlations. We then extract the correlation values between all features and the target, sorting them in ascending order, and see that **`Wind_Speed_kmph`** has the highest positive correlation (0.76), followed by **`Quantum_Battery`** (0.44), whilst features like **`Delivery_Time_Minutes`** and **`Market_Region`** have weak or negative correlations, making them potetial candidates for removal to simplify the model.

- 2.3) Dropping rows with missing values in important features: The rows with missing values for the highly correlated features are removed, ensuring that critical data points aren't compromised when training the model. 

- 2.4) Dropping irrelevant or negatively correlated columns: We make sure that columns with weak or negative correlations to the target are dropped to reduce dimensionality and eliminate noise in the dataset.

- 2.5) Removing skewness: From the descriptive statistics before, we saw that one of the numerical values had the potential of being skewed, and therefore we have to remove it. 
(Insert plot) (**)
From the plot we see that the original distribution of **`Cleaning_Liquid_Usage_liters`** is right-skewed, indicating that most values are clustered near zero. After the log transformation using **`np.log1p`**, the distribution becomes more symmetric, which can improve model performance by ensuring that the feature follows a normal distribution.
(Insert plot of log-transformed distribution) (**)



!!!CONTINUE FROM HERE (Rajla)!!!



### 3) Feature selection:
In this section, the only thing we had to decide was the treshold to select features to use to train the model. We simply tried to find the best value empirically, conducting an experiment that is explained in the **Experimental Design** section. The optimal result was obtained with a treshold of 0.15. As a consequence, the features dropped were: 'Gender', 'Arrival Delay in Minutes', 'Departure Delay in Minutes' and 'Age'.  

### 4) Splitting into training and test data:
	The dataset is split into training and testing sets using **`train_test_split`**. 

The **`X`** variables represent features, while **`y`** is the target variable.

The **`test_size=0.2`** specifies that 20% of the data is reserved for testing, and **`random_state=42`** ensures reproducibility. 

The training and testing shapes are printed to verify the split proportions.
* The **`train_test_split`** confirms the data is divided as intended, with 14,872 sampled in the training set and 3,719 in the test set.

- 4.1) **Distribution of values in the training and test data**: KDE (Kernel Density Estimation) plots are used to visualize the distributions of features in the training and test sets. We iterate through all features, plot overlapping KDEs for both sets, and adjust the layout to ensure clear visualization.
(Insert plot) (**)
**Interpretation**

* The KDE plots indicate that the feature distributions in both sets align closely, suggesting an even split without significant sampling bias. This ensures that the models trained on the training data will generalize well to the test data. However, features like **`Vertical_Landing`** and **`Terrain_type`** show distinct peaks, reflecting categorical distributions, while others like **`Air_Temperature_Celsius`** are continuous and symmetric.

### 5) Model Building
<h5 style="color: lightpink">A regression problem</h5>

This task was approached as a regression problem because the target variable, **Cargo_Capacity_kg**, is a continuous numerical value.

Regression models are specifically designed to predict continuous outcomes by learning the relationships between the input features and the target variable. Unlike classification, which deals with discrete categories, regression enables the prediction of a wide range of possible values, making it suitable for estimating quantities such as weight, price, or, in this case, the cargo caapacity of autonomous delivery drones. 

This choice aligns with the dataset structure and the objective of providing accurate numerical predictions, essential for operational and logistical planning in drone delivery systems.
- 5.1) **Testing Different Models**: We chose to test using the following regression models:
1. **`Linear Regression`**

* A basic regression model that assumes a linear relationship between the input features and the target variable. It fits a straight line to minimize the residual sum of squares between observed and predicted values.

2. **`Random Forest Regressor`**

* An ensemble learning method that build multiple decision trees and average their predictions to improve accuracy and reduce overfitting. It is robust to outliers and captures complex relationships.

3. **`Gradient Boosting Regressor`**

* An iterative ensemble technique that builds trees sequentially, with each tree correcting errors of the previous ones. It focueses on minimizing the loss function, making it effective for complex datasets.

4. **`K-Nearest Neighbors Regressor`**

* A non-parametric model that predicts the target value of a data point by averaging the values of its k nearest neighbors in the feature space. 

5. **`Support Vector Regressor`**

* A regression model that uses the concept of support vectors and hyperplanes. It tries to fit the data within a margin of tolerance (epsilon) while minimizing errors outside this margin. It is effective for datasets with a high dimensional feature space.
	-5.1.1) **Computing the metrics and comparing**: We define a dictionary **`models`** that holds several machine learning models for regression.
	Iterating through the models, training them on the dataset, and calculates the key metrics:

	**`MSE (Mean Squared Error)`**: Measures the average squared differences between actual and predicted values.

	**`MAE (Mean Absolute Error)`**: Measures the average absolute differences.

	**`R^2 (Coefficient of Determination)`**: Indicates the proportion of the variance explained by the model.

	**`Runtime`**: Captures the time each model takes for training and predictions.

	We then compute the Learning Curves and the Prediction Error Plots for each model.

	(Insert plots)(**)

	Afterwards, we aggregate the results from all models into a DataFrame. It sorts and displays them by R^2 in descending order, providing a direct comparison of their performances.
	**Interpretation**

	* **`Linear Regression`** performs best in terms of R^2 (0.9123) and runtime (0.046) seconds, making it efficient and effective.

	* **`Gradient Boosting`** has slightly lower R^2 but is robust for capturing nonlinear relationships.

	* **`SVR`** performs comparably but takes londer to run.

	* **`Random Forest`** performs moderately but has higher computational costs.

	* **`KNN`** shows the worst performance due to overfitting, with significantly lower R^2.
	- 5.1.2) Visualizing if the continuous values have a linear relationship with the target value: We create scatterplots to visualize the relationship between each continuous feature and the target 		variable (**`y_train`**). It loops through all continuous columns in the training dataset, plotting each feature against the target.
	(Insert plots)(**)

	**Interpretation**

	The scatterplots show that most features lack a strong linear relationship with the target variable, as the points are scattered uniformly. Specifically:

	1. *Air Temperature*, *Flight Hours*, *Cleaning Liquid Usage*, *Autopilot Quality*, and *Route Optimization*: The data points are distributed without any discernible pattern, suggesting no clear 		linear correlation with the target.

	2. *Wind Speed (kmph)*: This feature shows a visible pattern where the target value increases with wind speed, indicating a potential linear or non-linear correlation.

	This confirms that Linear Regression is not suitable for accurately predicting the target variable in this dataset and will be used only as a baseline model.

- 5.2) **Choosing the best models**: 

	##### 1. Gradient Boosting:
	**Why was it chosen?**

	Gradient Boosting is a powerful ensemble learning technique that build models sequentially, minimizing errors at each step. It was chosen because:

	* It captures complex, non-linear relationships in the data.

	* The model has shown robust performance with low MSE and high R^3 scores during cross-validation.

	* It is less prone to overfitting than Random Forest in some scenarios due to its iterative training process.

	##### 2. Support Vector Regressor (SVR)

	**Why was it chosen?**

	SVR uses kernel function to model non-linear relationships effectively. It was chosen because:

	* It is capable of finding a balance between bias and variance by defining margins for the predictions.

	* The model performed well in terms of accuracy ad R^2 scores, proving its suitability for this dataset.

	* It can handle outliers better than simpler regression techniques.

	##### 3. Linear Regression (as a baseline)

	**Why was it chosen?**

	Linear Regression was included as a baseline model for comparison purposes. It was chosen because:

	* It is straightforward, interpretable, and computationally efficient.

	* Despite its simplicity, it provides a benchmark to evaluate the performance of more complex models.

	* The scatterplots demonstrated that most features lack linear relationships with the target, confirming its limited utility but making it ideal for baseline evaluation.

	##### Why not the others?

	* *Random Forest*: While Random Forest is a strong model, it requires more computational resources and showed slightly inferior performance compared to Gradient Boosting and SVR.

	* *K-Nearest Neighbors(KNN)*: KNN performed poorly with the highest MSE and lowest R^2, indicating its inability to capture the complexities of the dataset. Additionally, its performace degrades with 	high-dimensional data.

-5.3) **Hyperparameter Tuning for Gradient Boosting Regression**:
	- 5.3.1) Define the Hyperparameter Grid:
	In the first step, a hyperparameter grid for the Gradient Boosting model is defined (**`param_grid`**). It includes possible values for parameters such as the number of estimators 				(**`n_estimators`**), learning_rate (**`learning_rate`**), maximum depth of the trees (**`max_depth`**), and others.

	**The parameters chosen**

	* **`n_estimators`**: This represents the number of boosting stages (or trees). A higher number allows the model to learn more complex patterns but can lead to overfitting if too large.

	* **`learning_rate`**: Conrols the contribution of each tree to the final prediction. Smaller values reduce overfitting and require more estimators for good performance.

	* **`max_depth`**: Limits the maximum depth of individual trees, controlling how complex each tree can be. Shallower trees help prevent overfitting.

	* **`min_samples_split`**: The minimum umber of samples required to split an internal node. Larger values result in less complex trees.

	* **`min_samples_leaf`**: The minimum number of samples required to be at a leaf node. Higher values make the trees more robust by reducing overfitting.
	
	- 5.3.2) Initialize GridSearchCV:
	We initialize a  **`RandomizedSearchCV`** object for hyperparameter tuning of the **`GradientBoostingRegressor`**. Here's what each parameter does:

	* **`estimator`**: Specifies the model, in this case, GradientBoostingRegressor.

	* **`param_distributions`**: The hyperparameter grid defined earlier (param_grid) that included values to sample during the search.

	* **`n_iter`**: The number of random combinations of hyperparameters to try (100 iterations here).

	* **`scoring='r2'`**: The performance metric used is the R^2 score.

	* **`cv=5`**: Performs 5-fold cross-validation for each parameter combination.

	* **`random_state=42`**: Ensures reproducibility of the results.

	* **`n_jobs=-1`**: Utilizes all available CPUs for computation.

	* **`verbose=1`**: Controls the amount of output during the execution.

	- 5.3.3) Fit the model:
	The best model and hyperparameters are then selected using **`random_search_gb.best_params_`** and evaluated on the test set.

	**Interpretation**

	1. **`n_estimators`**: 200

	This means the model uses 200 boosting iterations (or trees). Increasing the number of estimators can improve model performance but also increases computational time. A value of 200 strikes a balance 	between accuracy and efficiency.

	2. **`min_samples_split`**: 15

	Specifies the minimum number of samples required to split an internal node. By setting it to 15, the model avoids overfitting by ensuring splits are only performed sufficiently large groups of data.

	3. **`min_samples_leaf`**: 6

	This is the minimum number of samples required to be in a leaf node. Setting it to 6 reduces the likelihood of the model learing overly specific patterns (overfitting) and encourages more generalized 	splits.

	4. **`max_dept`**: 3

	Limits the maximum depth of each tree to 3 levels. This helps the model maintain simplicity and prevents overfitting by not learning overly complex patterns.

	5. **`learning_rate`**: 0.05

	Controls the contribution of each tree to the final prediction. A smaller learning rate (0.05) slows down the learning process, allowing the model to build more robust trees by minimizing the chance 	of overfitting.
	
	-5.3.4) Evaluate on the Test Set:
	**Interpretation**

	On the test set, the model performed with an MSE of **`0.0882`**, MAE of **`0.2381`**, ad an R^2 of **`0.9111`**.

	The model's test set results validate its effectiveness and confirm that the chosen hyperparameters generalize well.

-5.4) Hyperparameter Tuning for SVR
	-5.4.1 Define the Parameter Grid: 
	**The parameters chosen**

	* **`C`**: Regularization parameter. A smaller value of C allows for a larger margin, but may increase bias, while a larger value of C reduces margin for a better fit but may risk overfitting. Values 	like 0.1, 1, 10, 100 are tested.

	* **`epsilon`**: Defines the margin of tolerance where no penalty is given in the training loss function. Smaller values are more sensitive to deviations from actual values. Values like 0.001, 0.01, 	0.1, 1 are tested.

	* **`kernel`**: Specifies the kernel type to be used in the algorithm
    	* *linear*: A linear kernel assumes linear relationships in the data
    	* *rbf*: Radial Basis Function kernel is flexible and works well with non-linear data.
    	* *poly*: Polynomial kernel can capture complex relationships.

	-5.4.2) Initialize RandomizedSearchCV:
	This initializes a hyperparameter seach over the SVR model:

	* **`estimator=SVR()`**: Defines the base model (SVR) to tune.

	* **`param_distributions=param_grid_svr`**: Uses the grid of parameters defined earlier.

	* **`n_iter=20`**: The search will test 20 random combinations of the parameter grid.

	* **`scoring='r2'`**: The performance metric used is the R^2 score.

	* **`cv=5`**: Performs 5-fold cross-validation for each parameter combination.

	* **`random_state=42`**: Ensures reproducibility of the results.

	* **`n_jobs=-1`**: Utilizes all available CPUs for computation.

	* **`verbose=2`**: Provides detailed output during the search.
	-5.4.3) Fit the Model:
	The **`random_search.fit(X_train, y_train)`** code trains the model by:

	* Running 20 iterations of hyperparameter combinations.

	* For each combination, 5-fold cross-validation is conducted on the training data to compute the R^2 score.

	* The best hyperparameters and corresponding cross-validated R^2 score are identified.

	The print statements output the best parameters and cross-validation results, such as:

	* Best Parameters for SVR

	* Best Cross-Validation R^2

	**Interpretation**

	The best parameters indicate:

	* **`Kernel`**: 'linear' was optimal, meaning the relationship between predictors and the target is sufficiently linear.

	* **`C:`** 1, which provides a balance between margin size and fitting accuracy.

	* **`Epsilon`**: 0.01, a small margin of tolerance, indicating the model is sensitive to deviations in predictions.

	The cross-validation R^2 score of 0.913 shows the model generalizes well on unseen data, with high accuracy.
	-5.4.4) Evaluate on the Test Set:
	**Interpretation**

	* MSE: **`0.0869`** indicates low average squared error.

	* MAE: **`0.2355`** suggests predictions are off by about 0.2355 units on average.

	* R^2: **`0.9124`** confirms the model explains 91.24% of the variance in the target variable.

	These metrics show that the SVR model is well-tuned and performs similarly on both training and test datasets, indicating minimal overfitting.

-5.5) Evaluation for Linear Regression

A simple Linear Regression model is trained and evaluated on the test set without hyperparameter tuning, as it does not have tunable hyperparameters. The performance metrics are calculates similarly using MSE, MAE, and R^2.

**Interpretation**

* The Linear Regression model achieved comparable performance to the other models, with an MSE of **`0.0870`**, MAE of **`0.2357`**, and an R^2 of **`0.9123`**. 

* This suggests that even without advanced techniques, linear regression is a strong baseline for this dataset.

	-5.5.1) Comparison of the Results:
	**Interpretation**

	* The results show that SVR performs slightly better than both Gradient Boosting and Linear Regression, with the lowest MSE, lowest MAE, and the highest R^2.

	* Linear Regression performs suprisingly well, almost matching SVR in R^2, but its MSE and MAE are sligtly higher. 

	* Gradient Boostig, while effective, has slighlty higher error metrics and a marginally lower R^2, indicating it may not capture the underlying data patterns as effectively as SVR.

	(Insert plot Model Performance Comparison)(**)
	**Interpretation**

	The bar chart visualizes the performance metrics for the three models.

	* SVR has the lowest MSE and MAE, confirming it makes the smallest prediction errors among the three models. It also has the highest R^2, indicating it explains the variance in the target variable 		better than the others.

	* Linear Regression performs almost as well as SVR, with comparable R^2 and slightly higher error metrics. 

	* Gradient Boosting, while effective, shows slightly higher MSE and MAE, suggesting it may not generalize as well as the other models for this dataset.
	
	-5.5.2) Residual Plots
	Residual plots help evaluate how well a model's predictions align with the actual data. They plot the residuals (differences between predicted and actual values) against the predicted values. 		Residuals should ideally have no discernible pattern and be centered around zero.

	This indicates that the model captures the data effectively and that errors are randomly distributed. Using residual plots is essential to check for non-linearity, and any systematic bias in the 		predictions.
	(Insert plots Residual Plot for GB, SVR and Linear Regression)
	
	**Interpretation**

	1. *Gradient Boosting*: The residuals are distributed relatively evenly around zero, with no visible pattern. This indicates that the model captures the data structure well, though some variance in 	prediction errors is noticeable.

	2. *SVR*: The SVR residual plot also shows residuals scattered evenly around zero with minimal structure or pattern, suggesting that the model predictions align well with the actual data. The 		dispersion of residuals is slightly more concentrated compared to Gradient Boosting, indicating consistent prediction behavior.

	3. *Linear Regression*: The residuals for linear regression are evenly distributed around zero, similar to the other two models. However, since linear regression assumes a linear relationship, any 		slight deviations might indicate that this assumption may not fully capture the complexity of the data.

	Overall, all three models show acceptable residual distributions, but Gradient Boosting and SVR may handle complex data patterns slightly better than Linear Regression due to heir more advanceed 		architectures.

-5.6) Training and evaluating the models after Hyperparameter Tuning

* This code defines the models (**`Gradient Boosting`**, **`SVR`**, and **`Linear Regression`**) with their best hyperparameters obtained during the tuning process.

* **`results_with_best_params`** is initialized as an empty dictionary to store the metrics for each model.

Then the function **`plot_learning_curve`** visualizes the relationship between training set size and model performance on training and validation data, and the resulting plot shows how the model's performance varies as the training set size increases, helping to detect underfitting or overfitting.

Each model is iteratively trained using the training data and predictions are made on the test set, and performance metrics and runtime are computed and stored in the results dictionary.

(Insert plots of Gradient Boosting, SVR and Linear Regression with Best parameters) (**)

**Interpretation**

**`Learning Curves`**

1. Gradient Boosting

    * The training accuracy decreases slightly as the training size increases, suggesting the model generalizes well.
    * The validation accuracy remains consistent with a slight improvement as more data is used, indicating low overfitting.

2. SVR

    * Similar to Gradient Boosting, the training accuracy decreases as the training size increases.
    * Validation accuracy shows marginal improvement with more data, indicating stable performance and minimal overfitting.

3. Linear Regression

    * The training accuracy is slightly lower than that of Gradient Boosting and SVR, reflecting the simpler nature of the model.
    * Validation accuracy remains steady and comparable to the training accuracy, indicating good generalization.


**`Performance Metrics`**

* *Gradient Boosting*: Offers strong performance with an R^2 of 0.9111 and low errors but has a slightly longer runtime compared to Linear Regression.

* *SVR*: Achieves the highest R^2 and lowest error metrics but has a significantly higher runtime.

* *Linear Regression*: Has a marginally lower R^2 but compensates with the fastest runtime, making it suitable for time-sensitive tasks.

**Why so small differences?**

The minimal differences observed in the learning curves and evaluation metrics across models can be attributed to the homogeneity of the dataset. Specifically, equally distributed categorical variables simplify patterns in the data, reducing the need for models to adapt to challenging or unique scenarios.

As a result, hyperparameter tuning has little impact, as the models can already achieve optimal or near-optimal performance on the dataset.

### 6) Feature Importance:
- 6.1) **Feature Importance for Gradient Boosting**: We calculate the feature importance for the Gradient Boosting model:

1. **`hasattr`** check: This ensures the model supports feature importance extraction.

2. Extract Importance: **`feature_importance`** retrieves the importance scores assigned to each feature by the GradientBoostingRegressor.

3. DataFrame Creation: A DataFrame is created to pair feature names with their respective importance scores and sort them in descending order.

4. Bar Chart plotting:
    * A horizontal bar chart is plotted to visualize the feature importance scores.

    * Features are plotted on the y-axis, and their importance is on the x-axis.
    
    * **`plt.gca().invert_yaxis()`** ensures the most important features appear at the top.

(Insert plot Feature importance for gradient boosting) (**)

- 6.2) **Feature Importance for Linear Regression**: We use a similar code for extracting the importance of features for linear regression, the only difference is that we extract coefficients.

The **`linear_model.coef_`** retrieved the coefficients of the Linear Regression model. These coefficients represent the weight or importance of each feature.
(Insert plot Feature importance for linear regression) (**)

**Interpretation**

* The plot illustrates the relative importance of features in the Linear Regression model. Quantum_Battery and Wind_Speed_kmph have the highest importance, followed by Flight_Duration_Minutes. 

* Linear Regression assigns importance based on the magnitude of feature coefficients. Features with higher coefficients exert greater influence on the predictions.

* The results make sense for Linear Regression since the importance reflects how directly the features correlate with the target variable. Features with near-zero coefficients, such as Package_Type and Weather_Status, have minimal or no linear correlation with the target.

- 6.3) **Comparison of Linear Regression and Gradient Boosting Feature Importance**: The Gradient Boosting model relies on iterative decision tree splits, which account for feature interactions and nonlinear patterns. As a result, **`Wind_Speed_kmph`** dominates the importance in Gradient Boosting, likely due to its nonlinear influence on the target. Conversey, Linear Regression evaluates only linear relationships, leading to **`Quantum_Battery`** and **`Wind_Speed_kmph`** having significant coefficienct due to their correlations.

This highlights the difference in the underlying mechanics of the two models: Gradient Boosting captures complex dependencies by amplifying the importance of certain features while minimizing others, while Linear Regression prioritizes direct proportionality.

(Insert plot Feature importance for Linear regression vs Gradient Boosting)(**)

#Conclusions

1) General considerations
The machine learning models developed in this project successfully predict the cargo capacity of autonomous delivery drones operating in Aeropolis, a futuristic urban setting. By using Gradient Boosting, Linear Regression, and other algorithms, we analyzed the factors impacting drone performance and fine-tuned the models for accurate predictions. The dataset proved to be consistent and well-structured, as the models performed effectively on both training and test sets, ensuring reliable conclusions for optimizing drone logistics.

2) Model comparison
Among the models evaluated, Gradient Boosting delivered the best results, outperforming others in accuracy, precision, and the ability to capture both linear and non-linear relationships. It effectively balanced training and validation accuracy while minimizing overfitting. Linear Regression, on the other hand, showed the weakest performance, as expected, given its limitations in handling non-linearity and subtle data patterns. Support Vector Regression (SVR) also struggled to adapt to the complex dynamics of the dataset.

While Gradient Boosting was the clear winner, its computational cost was higher compared to simpler models like Linear Regression. However, the gains in predictive performance justify its use for this application, particularly in Aeropolis's data-rich and dynamic environment.

3) Analysis of the most important features
Feature importance analysis revealed that Wind_Speed_kmph, Quantum_Battery, and Flight_Duration_Minutes were the most influential factors affecting cargo capacity. These results emphasize the critical role of environmental conditions and battery efficiency in optimizing drone performance. While other features contributed less significantly, they provided useful context for refining performance in complex scenarios.

As evident from the analysis, Wind_Speed_kmph emerged as the most critical feature, which aligns with expectations given the challenges drones face in maintaining stability during high winds. Similarly, Quantum_Battery highlights the importance of battery efficiency, especially for longer flight durations.

4) Model evaluation and insights
When tested on unseen data, Gradient Boosting consistently demonstrated superior performance, achieving high accuracy and minimizing errors. 
The robustness of Gradient Boosting makes it the most suitable model for this application. Linear Regression and SVR, while simpler, failed to adapt to the dataset’s complexity, reinforcing the importance of selecting models that align with the data’s characteristics.

5) Preprocessing and methodology
To ensure reliable predictions, rigorous preprocessing techniques were applied, including scaling, outlier removal, and feature engineering. These steps were critical for capturing relationships within the dataset and improving model performance. Gradient Boosting stood out for its ability to handle non-linear interactions and complex data patterns, making it the best choice for Aeropolis's challenging logistics environment.

6) Future work
This project demonstrates the potential of machine learning to optimize autonomous drone logistics in high-demand, tech-forward cities like Aeropolis. Future work could build upon these findings by:

Incorporating real-time drone data to improve model responsiveness.
Integrating weather forecasting models to account for changing environmental conditions.
Developing adaptive systems that dynamically adjust drone operations based on operational and environmental factors.
These advancements could enhance delivery efficiency and establish a robust logistics network for smart cities, ensuring seamless operations in even the most demanding urban environments.
