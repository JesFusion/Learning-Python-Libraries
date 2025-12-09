import pandas as pd
from sqlalchemy import create_engine
from sklearn.datasets import load_iris
from jesse_custom_code.pandas_file import postgre_connect


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






