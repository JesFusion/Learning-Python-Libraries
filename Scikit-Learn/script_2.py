import os
import joblib
import numpy as np
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from sklearn.datasets import load_iris
from jesse_custom_code.pandas_file import postgre_connect, PDataset_save_path as psp, dataset_save_path
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from jesse_custom_code.build_database import PSQLDataGenerator
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, ElasticNet
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score









# ===================================== Sourcing Data =====================================

iris_dset = load_iris()

print(f'''
======================================== iris .data (Features) ========================================

{iris_dset.data[:2]}


======================================== iris .target (Labels) ========================================

{iris_dset.target[:2]}


======================================== iris .feature_names (The Metadata (The Context)) ========================================

{iris_dset.feature_names}

Feature Size: {iris_dset.data.shape}
''')


iris_dataset = pd.DataFrame(data = iris_dset.data, columns = iris_dset.feature_names)

iris_dataset['target'] = iris_dset.target

iris_name = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

iris_dataset["iris_species_name"] = iris_dataset["target"].map(iris_name)

print(f'''
======================================== Data Extracted! Shape: ({iris_dataset.shape}) ========================================

{iris_dataset.sample(6).to_markdown()}
''')



# ===================================== Ingesting Data to PostgreSQL DataBase =====================================


database_engine = create_engine(postgre_connect)

iris_dataset.to_sql(
    name = "Scikit-Learn Iris",

    con = database_engine,

    if_exists = "replace",

    index = False,
)

print("\nDatabase Uploaded successfully!\n")



# ===================================== Extracting Data for Modeling =====================================

the_dataset = pd.read_sql(
    'SELECT * FROM "Scikit-Learn Iris"',

    database_engine
)

# viewing the dataset...

print(f'''
======================================== Extracted Dataset ========================================
      
{the_dataset.head().to_markdown()}
''')






















































































d_engine = create_engine(postgre_connect)

the_dataset = pd.read_sql(
    "SELECT * FROM car_prices_raw",

    d_engine
)

print(f'''
======================================== Extracted Dataset ========================================
      
{the_dataset.head().to_markdown()}
''')

# ===================================== Defining Concepts in Scikit-Learn and Data Science in General =====================================

# Target/Labels (y): This is what we want to predict (usually a single pandas Series/Column)

target_y = the_dataset["Price"]

print(f'''
======================================== Target (y), Shape: {target_y.shape} ========================================

{target_y.head().to_markdown()}

It is a Vector (ID)
''')

# Features (X): this is what the model tries to learn in order to see it's pattern with the targets

feature_x = the_dataset.drop(["Price"], axis = 1)

print(f'''
======================================== Features (X), Shape: {feature_x.shape} ========================================

{feature_x.head().to_markdown()}

It is a Matrix (2D)
''')

"""
WHY PREPROCESSING?

If we were to run this line below, Scikit-learn would CRASH:
```model.fit(feature_x, target_y)

WHY THIS WOULD FAIL:

1. Look at 'Fuel_Type': It contains strings like 'Electric'.

You cannot do: 'Electric' * 0.5 + 2. The math explodes.

Solution: We need Encoding

2. Look at 'Odometer_KM' vs 'Age_Years':
Max Odometer: {feature_x['Odometer_KM'].max()}
Max Age: {feature_x['Age_Years'].max()}

The model will think Odometer is 25,000x more important than Age just because the number is bigger.
Solution: We need Scaling
"""













































































# ===================================== Extracting Dataset =====================================

database_engine = create_engine(postgre_connect)

with database_engine.connect() as database_connection:

    user_credit_dataset = pd.read_sql(
        "SELECT age, credit_score FROM user_credit_logs",
        database_connection
    )



print(f'''
======================================== Extracted Dataset (Pre-Cleaning) ========================================
      
{user_credit_dataset.head().to_markdown()}
''')


# we'll use SimpleImputer to fill the missing values

'''
SimpleImputer is a class in sklearn that helps us fill missing values with the:

1. Mean: Good for normal data.

2. Median: Good for data with outliers (billionaires don't skew the median income, but they ruin the mean).

3. Mode (Most Frequent): Good for categorical data (fill missing color with "Red" if "Red" is most common).
'''


sk_imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")

cleaned_array = sk_imputer.fit_transform(user_credit_dataset) # .fit_transform returns a Numpy array, not a DataFrame. It strips the column of it's headers before working on it

cleaned_dataset = pd.DataFrame(cleaned_array, columns = ["age", 'credit_score'])


print(f'''
======================================== Cleaned Dataset (Post-Imputation) ========================================
      
{cleaned_dataset.head().to_markdown()}
''')














































































cs_dataset = pd.read_sql(# cs = customer_surveys
    "SELECT satisfaction, country, loyalty_years FROM customer_surveys",

    create_engine(postgre_connect)
)
print(f'''
======================================== Extracted Raw Dataset (Strings) ========================================
      
{cs_dataset.head().to_markdown()}
''')

# ===================================== ORDINAL ENCODING =====================================

# oridinal encoding is used on ordinal data, which is data with order/rank, where values are sequentially greater than each other
# For example, in the "satisfaction" column, we have Low < Medium < High
# This column is perfect for performing ordinal encoding on


# .
cat_order = [["Low", "Medium", "High"]] # we must define the order manually

cs_ord_enc = OrdinalEncoder(categories = cat_order) # instantiang the OrdinalEncoder class/model and passing parameters


# We use "OrdinalEncoder" .fit_transform() method to learn the categories AND change the data at once
cs_dataset["enc_satisfaction"] = cs_ord_enc.fit_transform(cs_dataset[['satisfaction']]) # notice we used cs_dataset[['satisfaction']] and not cs_dataset['satisfaction']
# the .fit_transform() method expects a DataFrame, not a Series


print(f'''
======================================== After Ordinal Encoding (satisfaction) ========================================

{cs_dataset.head().to_markdown()}
''')


# ===================================== ONE-HOT ENCODING =====================================

'''
One-Hot encoding is used on Nominal data, which is data with no rank/order. It creates a new "switch" (column) for every option. e.g: "is_usa", "is_france"

We use Scikit-Learn's OneHotEncoder to perform One-Hot encoding:

- sparse_output = False: forces it to return a numpy array we can see (not a compressed matrix)

- handle_unknown = 'ignore': is CRITICAL for production. If a new country 'Belarus' appears later, the model won't crash; it will just produce all zeros.
'''

cs_oht_enc = OneHotEncoder(
    sparse_output = False,
    handle_unknown = "ignore"
)

enc_country_column = cs_oht_enc.fit_transform(cs_dataset[["country"]])

# OneHotEncoder returns a numpy array with no column names. To retrieve the column names, we use the .get_feature_names_out() method

con_ft_names = cs_oht_enc.get_feature_names_out(['country'])

# converting the numpy array and column names to a new DataFrame...
cy_dframe = pd.DataFrame(enc_country_column, columns = con_ft_names)

# join it back to the original DataFrame

oh_final_dset = pd.concat([cs_dataset, cy_dframe], axis = 1)

print(f'''
======================================== After One-Hot Encoding (country) ========================================
      
{oh_final_dset.head().to_markdown()}
''')


# let's drop the categorical columns and save our final processed dataset

# dropping categorical columns...
cs_dataset = oh_final_dset.drop(['satisfaction', 'country'], axis = 1)

save_path = f"{psp}customer_survey_dataset.parquet"

# saving as a parquet file...
# cs_dataset.to_parquet(save_path, index = False)

print("\nDataset saved as parquet file!\n")
print(save_path)


















































































# creating synthetic dataset

subscription_logs = PSQLDataGenerator(connection_string = postgre_connect)

subscription_logs.create_and_populate_table(
    primary_key = 'user_id',
    table_name = "subscription_logs",
    column_config = {
        "user_id": ["TEXT", "prefixed_id"],
        "plan_tier": ['TEXT', "rand_ch"],
        "region": ['TEXT', "rand_ch"],
        "monthly_usage_hrs": ["INTEGER", "rand_intg"],
        'support_calls': ["INTEGER", "rand_intg"]
    },

    groups = {
        "user_id": 'log',
        'plan_tier': ["Basic", "Premium", "Enterprise"],
        'region': ['North America', 'Europe', 'Asia'],
        'monthly_usage_hrs': [0, 500],
        'support_calls': [0, 10]
    },

    num_entries = 15992 # mimics real dataset size
)

print("\nTable Uploaded to Database!\n")


# ===================================== Extracting the Data =====================================
Slogs_dset = pd.read_sql(
    "SELECT * FROM subscription_logs LIMIT 9500",

    create_engine(postgre_connect)
)

print(f'''
======================================== Original Dataset ========================================
      
{Slogs_dset.head().to_markdown()}
''')

# injecting nan values into the monthly_usage_hrs column...
def custom_nan_insertion(usage_hrs):

    if (usage_hrs % 7) == 0:
        return np.nan
    
    else:
        return usage_hrs

Slogs_dset["monthly_usage_hrs"] = Slogs_dset['monthly_usage_hrs'].apply(custom_nan_insertion)

Slogs_dset.info(memory_usage = "deep") # we have 8079/9500 non-null under the monthly_usage_hrs column


# imputing data and joining back to original DataFrame...

Slogs_dset["monthly_usage_hrs"] = SimpleImputer(missing_values = np.nan, strategy = "mean").fit_transform(Slogs_dset[["monthly_usage_hrs"]]).flatten() # The .fit_transform() method of SimpleImputer enables it to create a numpy array that fills the missing values in the "monthly_usage_hrs" column of "Slogs_dset" with the mean (as we specified in the "strategy" attribute of SimpleImputer).
# Notice that we passed a DataFrame to the .fit_transform() method using double square brackets instead of a single one, which would have returned a Series (inserting a Series would throw an error).
# The .flatten() method converts the resulting array to a 1 Dimensional Array, which is what pd.Series expects


print(f'''
======================================== Dataset After Imputation ========================================
      
{Slogs_dset.head().to_markdown()}
''')

Slogs_dset.info(memory_usage = "deep")


# ===================================== performing Ordinal Encoding =====================================

Slogs_dset["plan_tier"] = OrdinalEncoder(categories = [['Basic', 'Premium', 'Enterprise']]).fit_transform(Slogs_dset[['plan_tier']]) # Just like SimpleImputer, OrdinalEncoder expects a DataFrame instead of a Series in it's .fit_transform() method, so we use double instead of single square brackets. The "categories" attribute in OridinalEncoder collects a list of lists (i'm not sure if i'm correct by identifying it as a "list of lists") that contains the order/rank of the Ordinal Data


print(f'''
======================================== Dataset After Performing Oridinal Encoding ========================================
      
{Slogs_dset.head().to_markdown()}
''')

# ===================================== performing One-Hot Encoding =====================================

region_OH_encode = OneHotEncoder(
    sparse_output = False,
    handle_unknown = "ignore"
)

reg_encode_data = region_OH_encode.fit_transform(Slogs_dset[["region"]])

Slogs_dset = pd.concat([
    Slogs_dset,

    pd.DataFrame(
        data = reg_encode_data,

        columns = region_OH_encode.get_feature_names_out(["region"]),
    )
],
    axis = 1
).drop(["region", "user_id"], axis = 1)


print(f'''
======================================== Final Dataset After Pre-Processing ========================================
      
{Slogs_dset.head().to_markdown()}
''')


# saving pre-processed dataset as parquet file for training later...
# Slogs_dset.to_parquet(f"{PDataset_save_path}subscription_logs_dataset.parquet", index = False)
print("\nPre-Processed Dataset Saved as parquet file!")



















































































the_dataset = pd.DataFrame({
    'age': [25, 45, 30, 50],           # Range: 25 (Tiny)
    'salary': [50000, 120000, 60000, 110000] # Range: 70,000 (HUGE)
})

print(f'''
======================================== Original Dataset ========================================
      
{the_dataset.head().to_markdown()}
''')

c_names = list(the_dataset.columns)

# StandardScaler centers data around 0 with a spread (std dev) of 1
# it subtracts the mean of a column from each row, dividing the result by the Standard Deviation
# ==> (Value - Mean) / StdDev
std_dset = StandardScaler().fit_transform(the_dataset)


std_dframe = pd.DataFrame(
    std_dset,
    columns = c_names
).round(2)

print(f'''
=================================== Dataset Standardized with StandardScaler (Centered around 0) ===================================
      
{std_dframe.head().to_markdown()}
''')

# Normalization squashes the column values to numbers between a range of 0 - 1
# Formula: (Value - Min) / (Max - Min)

scd_dset = MinMaxScaler().fit_transform(the_dataset)

scd_dframe = pd.DataFrame(scd_dset, columns = c_names).round(2)\
# when a dataset is Squashed with MinMaxScaler, the lowest value in a column is always 0 and the highest is always 1


print(f'''
=============================== Dataset Normalized with MinMaxScaler (Squashed between 0 to 1) ===============================
      
{scd_dframe.head().to_markdown()}
''')

















































































# ===================================== loading & dividing the dataset =====================================

dataset = pd.DataFrame({
    'usage_hours': [10, 50, 5, 100] * 250,
    'payments': [1, 5, 0, 10] * 250,
    'churn': [1, 0, 1, 0] * 250
})

print(f'''
======================================== Original Dataset ========================================
      
{dataset.sample(6).to_markdown()}
''')

# dividing the dataset into features and labels...
features_x = dataset.drop(["churn"], axis = 1)

target_y = dataset["churn"]


# ===================================== Splitting the Dataset =====================================

X_train, X_test, y_train, y_test = train_test_split(features_x, target_y,test_size = 0.2, random_state = 19)

print(f'''
Training Rows: {len(X_train)}
Testing Rows: {len(X_test)}
''')



# we only fit on the training dataset and after we have split the dataset into training and testing.
# If you scale the whole dataset first before splitting, the "Mean" of the Test set leaks into the Training set. This is called "Data Leakage"

scaling_transformer = StandardScaler()

# we apply fit_transform on the training data, which learns the data parameters (e.g: mean, std) and transforms it using that information
X_train = scaling_transformer.fit_transform(X_train)

# We then transform the test set using the learned parameters from fit_transform in the training set
X_test = scaling_transformer.transform(X_test)

print(f'''
Data Type of x_train: {type(X_train)}
Shape: {X_train.shape}

Data Type of x_test: {type(X_test)}
Shape: {X_test.shape}
''')

print("\nData Split and Scaled correctly without leakage!\n")
















































































np.random.seed(19)

'''
house_dataset = pd.DataFrame({
    "sq_ft": np.random.randint(1000, 9999, size = 59999),
    "age": np.random.randint(18, 66, size = 59999),
    "price": np.random.randint(100, 1000, size = 59999), # Target 1 (Continuous)
    "sold_fast": np.random.randint(0, 2, size = 59999) # Target 2 (Category: 1=Yes, 0=No)
})
'''

house_dataset = pd.read_parquet(f"{dataset_save_path}house_specs.parquet")


print(f'''
======================================== Original Dataset ========================================
      
{house_dataset.sample(7).to_markdown()}''')

# assigning features and targets...
# Notice that to converted the features and targets to a numpy array, because it strips the DataFrame of it's column names. If I didn't, I'll get this warning when predicting with the model:
# UserWarning: X does not have valid feature names, but KNeighborsRegressor was fitted with feature names

features_x = house_dataset.drop(["price", "sold_fast"], axis = 1).to_numpy()

target_price = house_dataset["price"].to_numpy()

targets_sale = house_dataset["sold_fast"].to_numpy()


"""
KNN predicts the value of an input by comparing it's entries with that of similar (ie, closely similar to) entires in the training data.


It is used in both classification and regression:
1. Classification: If 2 out of 3 neighbors churned, the new user will likely churn. (Voting).
2. Regression: If the 3 neighbors spend $100, $120, and $110, the new user will likely spend $110. (Averaging).

A small K (e.g: 1) is highly sensitive to noise, while a large one (e.g: 45) is smooth but might miss important details

A "K" value is passed by inserting a value to the "n_neighbors" attribute in KNeighborsRegressor or KNeighborsClassifier)
"""

# ===================================== KNN FOR REGRESSION (Predicting price) =====================================

knn_model_reg = KNeighborsRegressor(n_neighbors = 4)

knn_model_reg.fit(features_x, target_price)

house_data = np.array([[5623, 20]]) # our testing data will be a 20 year old house with 5623 sqft capacity

# let's predict using our model...
price_predict = knn_model_reg.predict(house_data)

print(f'''
Square feet: {house_data[0][0]} sqft
House Age: {house_data[0][1]} years
Predicted Price from Model: ${price_predict[0]:.2f}''')


# ===================================== KNN FOR CLASSIFICATION (Predicting Speed) =====================================

# here we classify data into groups. 0 means a house won't sell fast (slow group), while 1 means that the house will sell fast (fast group)

knn_model_cls = KNeighborsClassifier(n_neighbors = 5).fit(features_x, targets_sale)

# Predicting if the house will sell fast...

sale_pred = knn_model_cls.predict(house_data)

category = ["No", "Yes"] # let's map the groups to a list. It'll help us understand the model's prediction easily

print(f'''
Will the {house_data[0][1]}-year old house of {house_data[0][0]} sqft sell fast? {category[int(sale_pred[0])]}
''')









































load_dotenv()

dbase_string = os.getenv("POSTGRE_CONNECT")
model_save_path = os.getenv("MODEL_SAVE_PATH")

# extracting dataset from postgre server
database_engine = create_engine(dbase_string)

diamond_sales_dataset = pd.read_sql(
    sql = "SELECT carat_weight, cut_rating, price_usd FROM diamond_sales LIMIT 60700",

    con = database_engine    
)

# using sklearn's OridinalEncoder to convert the "cut_rating" categorical column to a numerical one with order
diamond_sales_dataset["cut_rating"] = OrdinalEncoder(categories = [['Fair', 'Good', 'Ideal']]).fit_transform(diamond_sales_dataset[['cut_rating']])


# splitting dataset into training and testing
features_x = diamond_sales_dataset.drop(["price_usd"], axis = 1)

target_y = diamond_sales_dataset['price_usd']


X_train, X_test, y_train, y_test = train_test_split(features_x, target_y,test_size = 0.2, random_state = 19)


# scaling training and testing features

"""
We .fit_transform() on the training set, and use the learned parameters to .transform() the test set
"""
scaling_algo = StandardScaler()

X_train = scaling_algo.fit_transform(X_train.to_numpy()) # I added .to_numpy() because I kept getting this warning when I was building the API:
# /home/jesfusion/Documents/ml/ml-env/lib/python3.12/site-packages/sklearn/utils/validation.py:2749: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names

X_test = scaling_algo.transform(X_test.to_numpy())


# training KNeighborsRegressor on dataset

knnR_model = KNeighborsRegressor(n_neighbors = 5).fit(X_train, y_train)

# saving model and scaler
# the model only understands what the scaler is saying, like it's own personal translator.
# If you passed unscaled data or data not scaled by the models own scaler, you'll simply be speaking jibberish to the model

# let's save the model and it's scaler (personal assistant)

joblib.dump(knnR_model, f"{model_save_path}/KNN/diamond_model.pkl")

joblib.dump(scaling_algo, f"{model_save_path}/KNN/diamond_scaler.pkl")








































































random_number_generator = np.random.RandomState(42)


# generating features and targets
features_X = random_number_generator.rand(1230, 1) # 1230 rows, 1 column

targets_Y = (features_X > 0.51).astype(int).ravel()

actual_label = targets_Y[:1100]

predicted_label = actual_label.copy()

errors = random_number_generator.choice(1100, 50, replace = False)

predicted_label[errors] = 1 - predicted_label[errors]

# accuracy_score gives the percentage of correct model prediction
model_accuracy = accuracy_score(actual_label, predicted_label)




# ===================================== LINEAR REGRESSION AND REGRESSION METRICS =====================================

num_entries = 57600

side_X = 14.893 * random_number_generator.rand(num_entries, 1)

random_noise = random_number_generator.randn(num_entries, 1) * 2157.23

side_Y = (2319.64 * side_X) + 26575.31 + random_noise

# Splitting x and y into train and test sets...
X_train, X_test, y_train, y_test = train_test_split(side_X, side_Y,test_size = 0.25, random_state = 19)


# model Instantiation
LR_model = LinearRegression()

# we fit the model on the training sets
LR_model.fit(X_train, y_train)

# fetching out the model prediction on our test set
model_prediction = LR_model.predict(X_test)


# Evaluation metrics in Regression, MSE & R2:
""" 
Mean Squared Error (MSE): 
We take the distance between the real value and your prediction (the "error"), square it, and then average all those squared errors.
Why square it? Two reasons:
1. It removes negative signs (an error of -5 is just as bad as +5).
2. It punishes big mistakes heavily. Being off by 10 is way worse than being off by 5 (10^2 = 100 vs 5^2 = 25). It screams at the model to fix the big outliers.



R-squared (R^2):
MSE gives you a number like "24500.5", which is hard to interpret. Is that good? Bad? R^2 (Coefficient of Determination) gives you a score between 0 and 1 (usually) that represents "How much of the variance in the target did the model explain?"
- 1.0: The model explains 100% of the patterns. Perfect prediction.
- 0.0: The model is no better than just guessing the average value for everyone.
"""

# fetching the Mean Squared Error and R-squared Score...
MSE = mean_squared_error(y_true = y_test, y_pred = model_prediction)

R2 = r2_score(y_true = y_test, y_pred = model_prediction)


print(f'''
======================================== Regression Results ========================================

Learned Coefficient (Slope): {LR_model.coef_[0][0]:.2f}

Learned Intercept: {LR_model.intercept_[0]:.2f}

Mean Squared Error (MSE): {MSE:.2f}

R-squared Score: {R2:.4f}

{model_accuracy}
''')


































































np.random.seed(20)


# generating non-linear data to practice with
feature_x = np.sort(5 * np.random.rand(40, 1), axis = 0)

target_y = np.sin(feature_x).ravel()


# adding noise to the data
target_y[::5] += 3 * (0.5 - np.random.rand(8))

target_y = target_y + np.random.normal(0, 0.1, feature_x.shape[0])


# we define a list of models.
# each element is a tuple containing model name, model and degree
the_models = [
    (
        "Linear Model (Underfit)",
        LinearRegression(),
        1
    ),

    (
        "Polynomial Model (Overfit)",
        LinearRegression(),
        15
    ), # degree is set to 15 to make model overfit

    (
        "Ridge (L2)",
        Ridge(alpha = 1.0),
        15
    ),

    (
        "Lasso (L1)",
        Lasso(alpha = 0.1),
        15
    ),

    (
        "ElasticNet (Hybrid)",
        ElasticNet(
            alpha = 0.1,
            l1_ratio = 0.5
        ),
        15
    )
]


# let's plot the raw data so we can compare our various model output to it
figure, axes = plt.subplots(figsize = (14, 10))

axes.scatter(feature_x, target_y, c = 'black', label = "Noisy Data")


# specifying colors of each line to be plotted on the canvas
the_colors = [
    'red',
    'green',
    'blue',
    'orange',
    'purple'
]

# looping through each model and having them trained
for x, (model_name, model_type, model_degree) in enumerate(the_models):

    # we create a list of chained steps from data pre-preprocessing to training on model
    pipeline_steps = [
        (
            'poly',
            PolynomialFeatures(
                degree = model_degree,
                include_bias = False
            )
        ),

        (
            'the_scaler',
            StandardScaler()
        ),

        (
            'the_model',
            model_type
        )
    ]

    # create a pipeline from the list
    training_pipeline = Pipeline(pipeline_steps)


    # we fit using the pipeline. Data is pre-processed: 
    # 1. Multiple columns are added using PolynomialFeatures() based on the degree specified
    # 2. Features are scaled and then passed to the model for training
    training_pipeline.fit(feature_x, target_y)


    # generating random test data
    x_axis = np.linspace(0, 5, 100).reshape(-1, 1)

    # creating y_axis using predictions from the model
    y_axis = training_pipeline.predict(x_axis)


    axes.plot(
        x_axis, y_axis,
        c = the_colors[x],
        linewidth = 2,
        label = f"{x + 1}. {model_name}"
    )


    if hasattr(
        training_pipeline.named_steps['the_model'],

        'coef_'
    ):# if the model object has an attribute names 'coef_', run this code:
        
        # fetch the coefficient value
        coefficients = training_pipeline.named_steps['the_model'].coef_

        print(f'''
================================== {model_name} ==================================

List of coefficients: {coefficients}

Sum of Weights: {np.sum(np.abs(coefficients)):.2f}

Zeroed-Out Weights: {np.sum(coefficients == 0)}
    ''')


# adding finishing touches to the canvas:
# Main title, x and y labels and legend
figure.suptitle("Polynomial Regression: The Full Battle (inc. ElasticNet)")

axes.set_xlabel("Input Feature (X)")

axes.set_ylabel("Target (y)")

axes.legend()

# .ylim() limits the Y-axis view.
# The 'Polynomial (Overfit)' model might shoot up to y = 1000 at the edges, so we crop the view so we can actually see the sine wave details.

plt.ylim(-2, 3)
plt.close()
plt.show()





































































































database_engine = create_engine(os.getenv("POSTGRE_CONNECT"))

with database_engine.connect() as connection:

    house_prices_dataset = pd.read_sql(

        text(
            "SELECT * FROM house_prices"
        ),

        con = connection
    )

    exam_results_dataset = pd.read_sql(

        text(
            "SELECT * FROM exam_results",
        ),

        con = connection
    )


# ===================================== REGRESSION METRICS (MAE/RMSE) =====================================

feature_x_1 = house_prices_dataset[['size']].values

target_y_1 = house_prices_dataset[['price']].values

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
    feature_x_1,
    target_y_1,
    test_size = 0.22,
    random_state = 19
)

model_1 = LinearRegression()

model_1.fit(
    X_train_1,
    
    y_train_1
)

y_pred_1 = model_1.predict(X_test_1)


# MAE calculates the absolute difference between the prediction and the truth, then averages it
model_1_MAE = mean_absolute_error(y_true = y_test_1, y_pred = y_pred_1)

model_1_RMSE = np.sqrt( # RMSE is the square root of MSE, so we use np.sqrt()
    mean_squared_error(
        y_true = y_test_1,

        y_pred = y_pred_1
    )
)

# setting up our figure and canvas
figure, (axes_1, axes_2) = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 6))

# Adding a custom window title
figure.canvas.manager.set_window_title("MAE, RMSE & Logistic Regression")

axes_1.scatter(
    X_train_1[::20],

    y_train_1[::20],

    c = 'black',

    label = 'Original Data'
)

axes_1.plot(
    X_test_1,
    y_pred_1,
    c = 'blue',
    label = f'Model Prediction.\nMAE: {model_1_MAE:.2f}, RMSE: {model_1_RMSE:.2f}'
)

axes_1.set_title("MAE and RMSE in Regression")

axes_1.set_xlabel("House Size")

axes_1.set_ylabel("Price of House")

axes_1.legend()





# ===================================== LOGISTIC REGRESSION (LINEAR REGRESSION WITH SIGMOID) =====================================

# Logistic Regression is simply just Linear Regression with it's output passed to a Sigmoid Function
# the function takes a continous value from the model and squashes the value between 0 and 1.
# Values greater that 0.5 is in one class, while values less than 0.5 is in another
# This is called BINARY CLASSIFICATION

feature_x_2 = exam_results_dataset[['hours']].values

target_y_2 = exam_results_dataset['result'].values

X_train, X_test, y_train, y_test = train_test_split(feature_x_2, target_y_2,test_size = 0.22, random_state = 19)

# we instantiate a LogisticRegression model and train it on our data
LGR_model = LogisticRegression(random_state = 19)

LGR_model.fit(X_train, y_train)

axes_2.scatter(
    X_train,
    y_train,
    c = 'gray',
    alpha = 0.2,
    label = 'Student Data (0=Fail, 1=Pass)'
)

x_test_r = np.linspace(0, 12, 300).reshape(-1, 1)

model_prediction = LGR_model.predict_proba(x_test_r)[:, 1]

axes_2.plot(
    x_test_r,
    model_prediction,
    c = 'red',
    linewidth = 3,
    label = 'Sigmoid Probability Curve'
)

# drawing a line at the Decision Boundary (0.5) for better clarity
axes_2.axhline(
    y = 0.5, # this draws a straight line at the 0.5 point on the y-axis
    color = 'green',
    linestyle = '--',
    label = 'Decision Boundary (0.5)'
)

axes_2.set_xlabel("Hours Studied")

axes_2.set_ylabel("Probability of Passing")

axes_2.set_title("Logistic Regression: The Sigmoid Function")

axes_2.legend()

figure.suptitle("""Left: Linear Regression Model with MAE & RMSE metrics shown
Right: Logistic Regression Plot""",
    fontsize = 16,
)

# .subplots_adjust() helps us push the subplots down, so they don't touch the suptitle
figure.subplots_adjust(
    top = 0.83, # this pushes the top edge of the subplots to 83% of the figure's height
    bottom = 0.1 # this pushes the bottom edge of the subplots to 10% of the figure's height
)

plt.show()


























































































database_engine = create_engine(
    os.getenv("POSTGRE_CONNECT")
)

class ETL:

    def ingest_data(self):
        """
        Simulates the ETL (Extract, Transform, Load) process.
        """

        

        classification_data = pd.DataFrame(    
            data = {
            'col_1': self.generate_data(info = ['uniform']),

            'col_2': self.generate_data(info = ['uniform']),

            'is_spam?': self.generate_data(info = ['randint', 0, 2]) # either 0 or 1 (np.random.randint(0, 2, data_size))
        })

        regression_data = pd.DataFrame(
            data = {
                'square_feet': self.generate_data(info = ['randint', 500, 4000]),

                'number_of_rooms': self.generate_data(info = ['randint', 1, 10]),

                'house_price': self.generate_data(info = ['randint', 100000, 1000000]),
            }
        )

        classification_data.to_sql(
            "classification_data",
            database_engine,
            if_exists = 'replace',
            index = False
        )

        regression_data.to_sql(
            name = "house_dataset",
            if_exists = 'replace',
            index = False,
            con = database_engine
    )
        
    @classmethod
    def generate_data(self, info: list) -> np.array:

        data_size = 40000

        if info[0] == "uniform":

            return np.random.rand(data_size)

        elif info[0] == "randint":

            return np.random.randint(
                
                int(info[1]),

                int(info[2]),
                
                data_size
            )
        
        else:
            return None



    def extract_data(self):

        """
        Pull data from SQL database for Modeling
        """

        class_data = pd.read_sql_query(
            "SELECT * FROM classification_data",
            con = database_engine,
        )

        reg_data = pd.read_sql_query(
            "SELECT * FROM house_dataset",
            con = database_engine
        )

        return class_data, reg_data


etl_process = ETL()

if False:

    etl_process.ingest_data()

    print("Done Ingesting Dataset!")

if True:

    classification_dataset, regression_dataset = etl_process.extract_data()


# ===================================== CLASSIFICATION METRICS (Accuracy) =====================================

features_x = classification_dataset[['col_1', 'col_2']]

target_y = classification_dataset['is_spam?']

X_train, X_test, y_train, y_test = train_test_split(features_x, target_y,test_size = 0.2, random_state = 20)

# instantiating a LogisticRegression model to test our metrics
logistic_model = LogisticRegression()

# training the model...
if True:
    logistic_model.fit(X_train, y_train)

    print("Done Training Logistic Model!")


logistic_pred = logistic_model.predict(X_test)



logistic_accuracy = accuracy_score(
    y_true = y_test,
    y_pred = logistic_pred
)


print(f'''
(Logistic Regression): True Labels vs. Model Predictions
{
    pd.DataFrame({
        'true_value': y_test.to_numpy()[:10],

        'model_prediction': logistic_pred[:10]
    })
}

Accuracy Score: {logistic_accuracy:.2f} (or {logistic_accuracy * 100:.1f}%)
''')




# ===================================== REGRESSION METRICS (MSE & R2) =====================================

X_regression = regression_dataset[['square_feet', 'number_of_rooms']]

y_regression = regression_dataset['house_price']


X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_regression, y_regression, test_size = 0.2, random_state = 20)

# instantiating the linear regression model and training it
linear_model = LinearRegression()

if True:

    linear_model.fit(X_train_reg, y_train_reg)

    print("Done Training Linear Model!")



linear_pred = linear_model.predict(X_test_reg)

# MSE = ((True_Value - Predicted_Value)^2) / (n)

linear_mse = mean_squared_error(
    y_true = y_test_reg,

    y_pred = linear_pred
)

# R2 SCORE is a statistical measure of how close the data are to the fitted regression line. It has a range of -infinity to 1.0. (1.0 is perfect).
linear_r2 = r2_score(
    y_true = y_test_reg,

    y_pred = linear_pred
)

test_vs_pred = pd.DataFrame({
    'true_value': y_test_reg.to_numpy(),

    'model_prediction': linear_pred
})

for col in list(test_vs_pred.columns):

    test_vs_pred[col] = test_vs_pred[col].map('${:,.2f}'.format)


print(f'''
(Linear Regression): True Prices vs. Model Predicted Prices
{test_vs_pred.sample(11).to_markdown()}

Mean Squared Error: {linear_mse:,.2f}
MSE is hard to understand because the dollar amount was squared. To fix this we use RMSE, which is the square root of MSE

Root Mean Squared Error (RMSE): {np.sqrt(linear_mse):,.2f} (This is the actual value in dollars)

R2 Score: {linear_r2:.4f}
''')












