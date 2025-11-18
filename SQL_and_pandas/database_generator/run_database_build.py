from build_database import create_and_populate_table
import sqlite3
from jesse_custom_code.pandas_file import database_path as d_path
import os
# import random

DB_FILE = d_path




if os.path.exists(DB_FILE):
    os.remove(DB_FILE)

conn = sqlite3.connect(DB_FILE)

# --- Table 1: Employees ---
# --- CHANGED: Now uses "prefixed_id" ---
create_and_populate_table(
    conn=conn,
    table_name="Employees",
    primary_key="Employee ID",
    
    column_names = {
        # We set the TYPE to TEXT and the GEN_TYPE to "prefixed_id"
        "Employee ID": ["TEXT", "prefixed_id"], 
        "First Name": ["TEXT NOT NULL", "rand_name"],
        "Last Name": ["TEXT", "rand_name"],
        "Salary": ["REAL", "rand_intg"],
        "Department": ["TEXT", "rand_ch"],
    },
    
    group = {
        # We add the prefix for "Employee ID" here
        "Employee ID": "emp", 
        "Department": ['IT', 'Marketing', 'Operations'],
        "First Name": "f",
        "Last Name": "l",
        "Salary": ["f", 45000, 120000],
    },
    
    num_entries = 714
)

# --- Table 2: Contractors (NEW) ---
# --- This demonstrates how we avoid collisions ---
create_and_populate_table(
    conn=conn,
    table_name="Contractors",
    primary_key="Contractor ID",
    
    column_names = {
        # This also uses "prefixed_id" but with a *different* prefix
        "Contractor ID": ["TEXT", "prefixed_id"], 
        "Name": ["TEXT NOT NULL", "rand_name"],
        "Pay Rate": ["REAL", "rand_intg"],
    },
    
    group = {
        # The "con" prefix ensures no ID will *ever*
        # match an "emp" ID. This solves the problem!
        "Contractor ID": "con", 
        "Name": "f",
        "Pay Rate": ["f", 50, 200], # $/hr
    },
    
    num_entries = 534
)
# --- END NEW ---

# --- Table 3: Sales (UNCHANGED) ---
# We can still use "serial" for simple tables where
# 1, 2, 3... is fine and won't be confused with other tables.
create_and_populate_table(
    conn=conn,
    table_name="sales",
    primary_key="Order ID",
    column_names={
        "Order ID": ["INTEGER", "serial"],
        "Store": ["TEXT", "rand_ch"],
        "Product": ["TEXT", "rand_ch"],
        "Sales": ["REAL", "rand_intg"],
    },
    group={
        "Store": ['Store_A', 'Store_B', 'Store_C'],
        "Product": ['Apples', 'Oranges', 'Bananas'],
        "Sales": ["f", 5, 500],
    },
    num_entries = 1278
)

# Close the connection when all tables are built
conn.close()

print("\nAll tables created...")
# print("Run this script, then connect to my_database.db with SQLTools to see the new IDs!")