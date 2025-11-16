import random
from datetime import datetime, timedelta
import string
import names
import sqlite3 # I added this import
import os # I added this to delete the old DB
from jesse_custom_code.pandas_file import database_path as d_path



# This path needs to be set to your actual database file path
DB_FILE = d_path

# --- Helper Functions (No Changes) ---
def random_date(start_year=1991, end_year=2024):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    return (start + timedelta(days=random_days)).strftime('%Y-%m-%d')


def random_string(length):
    characters = string.ascii_letters + string.digits  # a-z, A-Z, 0-9
    return ''.join(random.choices(characters, k=length))

# --- Main Generator Function (Refactored) ---

def create_and_populate_table(
    conn: sqlite3.Connection,
    primary_key: str,
    group: dict[str, list[str]],
    column_names: dict,
    table_name: str,
    num_entries: int = 29
    ):

    cursor = conn.cursor()

    # --- 1. CREATE TABLE Logic ---
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

    column_definitions = []
    CN_list = []

    if " " in table_name:
        table_name = f"`{table_name}`"

    if primary_key not in column_names.keys():
        print("Error! Primary key not found!")
        return

    # adding column names and types to list
    for column_name, type_and_content in column_names.items():
        type_and_content = list(type_and_content)
        
        safe_column_name = f"`{column_name}`" if " " in column_name else column_name
        CN_list.append(safe_column_name)

        col_type = type_and_content[0]
        c_gen_type = type_and_content[1] # The "generation" type

        # --- CHANGED: Upgraded "serial" logic ---
        # If the type is "serial" AND it's the primary key,
        # we now *properly* set it to AUTOINCREMENT.
        if c_gen_type == "serial" and column_name == primary_key:
            column_definitions.append(
                f"  {safe_column_name} INTEGER PRIMARY KEY AUTOINCREMENT"
            )
        # --- END CHANGE ---
        elif column_name == primary_key:
            # This handles our new "prefixed_id" PK case
            column_definitions.append(
                f"  {safe_column_name} {col_type} PRIMARY KEY"
            )
        else:
            column_definitions.append(
                f"  {safe_column_name} {col_type}"
            )

    new_line = ''',
'''
    create_table_query = f"""
        CREATE TABLE {table_name} (
        {new_line.join(column_definitions)}
        );
    """
    
    print("--- Generated CREATE TABLE Query ---")
    print(create_table_query)
    cursor.execute(create_table_query)


    # --- 2. INSERT Data Logic ---

    CN_tuple = str(tuple(CN_list)).replace("'", "")
    
    placeholders = ", ".join(["?"] * len(CN_list))
    
    insert_query = f"INSERT INTO {table_name} {CN_tuple} VALUES ({placeholders})"
    
    print("\n--- Generated INSERT Query ---")
    print(insert_query)

    all_rows = []

    for i in range(1, num_entries + 1): # Changed to start from 1 for serial

        row_data = []

        for col_name, type_and_content in column_names.items():

            type_and_content = list(type_and_content)
            c_gen_type = "" # Renamed from c_type to "generation type"
            amount = 0

            if " " in str(type_and_content[1]):
                try:
                    amount_str, c_gen_type = str(type_and_content[1]).split(" ", 1)
                    amount = int(amount_str)
                except ValueError:
                    c_gen_type = str(type_and_content[1])
            else:
                c_gen_type = str(type_and_content[1])
            
            # --- Data Generation (Logic is the same) ---

            if c_gen_type == "date":
                row_data.append(random_date())

            elif c_gen_type == "rand_name":
                name_type = group.get(col_name)
                if name_type == "f":
                    row_data.append(names.get_first_name())
                elif name_type == "l":
                    row_data.append(names.get_last_name())
                else:
                    row_data.append(names.get_full_name())

            elif c_gen_type == "rand_str":
                row_data.append(random_string(amount))

            elif c_gen_type == "rand_ch":
                the_group = group.get(col_name)
                if the_group:
                    row_data.append(random.choice(the_group))
                else:
                    print(f"Error! List for column name '{col_name}' not found in group!")
                    row_data.append(None)

            elif c_gen_type == "rand_intg":
                the_range = group.get(col_name)
                if the_range:
                    if str(the_range[0]) == "f": # Float
                        the_number = round(random.uniform(int(the_range[1]), int(the_range[2])), 2)
                        row_data.append(the_number)
                    else: # Integer
                        the_number = random.randint(int(the_range[0]), int(the_range[1]))
                        row_data.append(the_number)
                else:
                    row_data.append(None)
            
            # --- CHANGED: This is the old "serial" logic ---
            # It appends None, which triggers the AUTOINCREMENT
            elif c_gen_type == "serial":
                row_data.append(None) 
            # --- END CHANGE ---

            # --- NEW: This is your "realistic ID" solution ---
            elif c_gen_type == "prefixed_id":
                # It looks in the 'group' dict for a prefix, e.g., "emp"
                prefix = group.get(col_name, "id") # Default to "id"
                
                # We'll use a random string for the unique part
                unique_part = random_string(8) 
                
                # This creates IDs like "emp_aB1cD2eF"
                row_data.append(f"{prefix}_{unique_part}")
            # --- END NEW ---

            else:
                row_data.append(None)
        
        all_rows.append(tuple(row_data))

    # --- 3. Execute and Save ---
    cursor.executemany(insert_query, all_rows)
    
    conn.commit()

    print(f"\nSuccessfully created and populated table '{table_name}' with {num_entries} entries.")


# ============================= testing the code =============================

if __name__ == '__main__':
    
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
        
        num_entries=50
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
        
        num_entries=25
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
        num_entries=100
    )

    # Close the connection when all tables are built
    conn.close()

    print(f"\n--- All tables created in {DB_FILE} ---")
    print("Run this script, then connect to my_database.db with SQLTools to see the new IDs!")