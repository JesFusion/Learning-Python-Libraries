import os
import random
import string
import names
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

postgre_connect = os.getenv("POSTGRE_CONNECT")



class PSQLDataGenerator:
    # This class is our main factory for creating and destroying data in Postgres.
    # It acts as a wrapper around the SQLAlchemy engine.

    def __init__(self, connection_string):
        # We start by trying to establish a connection to the database.
        try:
            # create_engine manages the pool of connections to the database.
            # It's better than opening a raw cursor every time.
            self.engine = create_engine(connection_string)
            
            # We immediately test the connection to ensure credentials work.
            # 'with' context manager ensures the connection closes automatically.
            with self.engine.connect() as conn:
                # executing "SELECT 1" is the standard way to ping a SQL database.
                conn.execute(text("SELECT 1"))
            
            # If we get here, the connection is alive and healthy.
            print("✅ Successfully connected to the PostgreSQL Database!")
        except Exception as e:
            # If the database is down or password is wrong, we crash loudly here.
            print(f"❌ Connection Failed. Error: {e}")
            raise

    def _random_date(self, start_year=2020, end_year=2024):
        # Define the earliest possible date (Jan 1st of start_year).
        start = datetime(start_year, 1, 1)
        
        # Define the latest possible date (Dec 31st of end_year).
        end = datetime(end_year, 12, 31)
        
        # Calculate the total time span (delta) between start and end.
        delta = end - start
        
        # Pick a random number of days within that time span.
        random_days = random.randint(0, delta.days)
        
        # Add the random days to the start date and format it as YYYY-MM-DD.
        # Postgres loves ISO 8601 format (YYYY-MM-DD).
        return (start + timedelta(days=random_days)).strftime('%Y-%m-%d')

    def _random_string(self, length):
        # We create a pool of characters: a-z, A-Z, and 0-9.
        characters = string.ascii_letters + string.digits
        
        # We select 'length' number of characters randomly and join them.
        return ''.join(random.choices(characters, k=length))

    def create_and_populate_table(self, table_name, primary_key, column_config, groups, num_entries=50):
        # Let the user know we are starting work on a specific table.
        print(f"\n--- Processing Table: {table_name} ---")

        # We will build the SQL string to create columns here.
        cols_sql = []
        
        # Iterate through the configuration dictionary provided by the user.
        for col_name, (sql_type, gen_type) in column_config.items():
            # We wrap column names in double quotes to handle spaces (e.g., "Dept ID").
            safe_col = f'"{col_name}"' 
            
            # Check if this column is the designated Primary Key.
            if col_name == primary_key:
                # If it's a 'serial' type, Postgres handles the auto-incrementing.
                if gen_type == 'serial':
                    cols_sql.append(f"{safe_col} SERIAL PRIMARY KEY")
                else:
                    # Otherwise, use the specified SQL type (e.g., TEXT PRIMARY KEY).
                    cols_sql.append(f"{safe_col} {sql_type} PRIMARY KEY")
            else:
                # Regular column definition.
                cols_sql.append(f"{safe_col} {sql_type}")

        # Construct the full CREATE TABLE query.
        # CASCADE ensures if this table is linked to others, it drops cleanly.
        create_query = f"""
        DROP TABLE IF EXISTS "{table_name}" CASCADE;
        CREATE TABLE "{table_name}" (
            {', '.join(cols_sql)}
        );
        """

        # Execute the schema creation query.
        with self.engine.begin() as conn:
            conn.execute(text(create_query))
        
        print("   -> Table structure created.")

        # Initialize a list to hold all our generated row dictionaries.
        rows_to_insert = []
        
        # Loop 'num_entries' times to generate the data rows.
        for i in range(1, num_entries + 1):
            row = {}
            # For each row, loop through every column to generate its value.
            for col_name, (sql_type, gen_type) in column_config.items():
                
                # Check if the generation instruction has a length parameter (e.g., "10 rand_str").
                if " " in str(gen_type):
                    parts = str(gen_type).split(" ", 1)
                    # If the first part is a number, it's the length.
                    if parts[0].isdigit():
                        amount = int(parts[0])
                        actual_type = parts[1]
                    else:
                        amount = 0
                        actual_type = gen_type
                else:
                    # No length parameter found.
                    amount = 0
                    actual_type = gen_type

                val = None

                # -- Generation Logic Switch --
                
                # SERIAL columns are auto-filled by the DB, so we skip generating a value.
                if actual_type == 'serial':
                    continue 

                # Generate a random date.
                elif actual_type == 'date':
                    val = self._random_date()

                # Generate a random name (First, Last, or Full).
                elif actual_type == 'rand_name':
                    # Check the 'groups' dict to see if we want 'f', 'l', or full name.
                    mode = groups.get(col_name, 'full')
                    if mode == 'f': val = names.get_first_name()
                    elif mode == 'l': val = names.get_last_name()
                    else: val = names.get_full_name()

                # Generate a random alphanumeric string of specific length.
                elif actual_type == 'rand_str':
                    # Use the parsed 'amount' or default to 8 chars.
                    val = self._random_string(amount if amount > 0 else 8)

                # Pick a random choice from a provided list (Categorical data).
                elif actual_type == 'rand_ch':
                    choices = groups.get(col_name, [])
                    val = random.choice(choices) if choices else None

                # Generate a random integer or float.
                elif actual_type == 'rand_intg':
                    constraints = groups.get(col_name)
                    if constraints:
                        # If the constraint starts with "f", it's a float (e.g., money).
                        if str(constraints[0]) == 'f':
                            val = round(random.uniform(constraints[1], constraints[2]), 2)
                        else:
                            # Otherwise, it's a standard integer range.
                            val = random.randint(constraints[0], constraints[1])
                    else:
                        val = 0

                # Generate a custom ID with a prefix (e.g., "emp_X7s9d").
                elif actual_type == 'prefixed_id':
                    prefix = groups.get(col_name, 'id')
                    unique_part = self._random_string(8)
                    val = f"{prefix}_{unique_part}"

                # Assign the generated value to the column key in the dictionary.
                row[col_name] = val
            
            # Add the completed row to our list.
            rows_to_insert.append(row)

        # Only proceed if we actually generated data.
        if rows_to_insert:
            # We map original column names (e.g. "Dept ID") to Python-safe keys ("Dept_ID").
            # This is required because SQLAlchemy bind params don't like spaces.
            col_map = {orig: orig.replace(" ", "_") for orig in rows_to_insert[0].keys()}
            
            # Create a clean version of the data using the safe keys.
            clean_rows = []
            for row in rows_to_insert:
                clean_row = {col_map[k]: v for k, v in row.items()}
                clean_rows.append(clean_row)

            # Build the SQL column list using the REAL names (with quotes).
            col_part = ", ".join([f'"{real_col}"' for real_col in col_map.keys()])
            
            # Build the VALUES part using the SAFE keys (prefixed with colon for binding).
            val_part = ", ".join([f':{safe_key}' for safe_key in col_map.values()])
            
            # Construct the final INSERT statement.
            insert_query = text(f'INSERT INTO "{table_name}" ({col_part}) VALUES ({val_part})')

            # Open a transaction and bulk insert all rows at once.
            with self.engine.begin() as conn:
                conn.execute(insert_query, clean_rows)
        
        print(f"   -> Inserted {len(rows_to_insert)} rows.")

    def inject_foreign_key_column(self, source_table, source_col, target_table, target_col_name):
        # This method mimics a JOIN logic by taking existing IDs from one table 
        # and randomly distributing them into another table.
        print(f"\n--- Injecting Foreign Key: {source_table}.{source_col} -> {target_table}.{target_col_name} ---")
        
        with self.engine.connect() as conn:
            # Pull all valid IDs from the source table (e.g., Departments).
            query = text(f'SELECT "{source_col}" FROM "{source_table}"')
            valid_ids = [row[0] for row in conn.execute(query)]
            
            # Safety check: can't link to an empty table.
            if not valid_ids:
                print("Error: Source table is empty!")
                return

            # Try to add the new column to the target table.
            # We wrap in try/except because if it runs twice, it might fail on "Column exists".
            try:
                conn.execute(text(f'ALTER TABLE "{target_table}" ADD COLUMN "{target_col_name}" TEXT'))
                conn.commit() # Commit DDL change immediately.
                print(f"   -> Column '{target_col_name}' added.")
            except Exception:
                print(f"   -> Column '{target_col_name}' already exists or other error.")

            # Get the unique row identifiers (ctid) of the target table so we can update them one by one.
            target_rows = conn.execute(text(f'SELECT ctid FROM "{target_table}"')).fetchall()
            
            updates = []
            # Prepare the update logic: assign a random source ID to every target row.
            for row in target_rows:
                random_id = random.choice(valid_ids)
                updates.append({'new_val': random_id, 'row_id': row[0]})
            
            print("   -> Assigning random relationships...")
            
            # Execute the updates.
            # Note: For massive tables, a batch update or temporary table join is faster.
            # But for simulation data (<10k rows), this loop is acceptable.
            for update in updates:
                conn.execute(
                    text(f'UPDATE "{target_table}" SET "{target_col_name}" = :new_val WHERE ctid = :row_id'),
                    update
                )
            conn.commit()
            
        print("   -> Relationship injection complete.")

    def corrupt_table_data(self, table_name, corruption_rate=0.1, target_columns=None):
        # Chaos Engineering: Randomly set values to NULL to simulate missing data.
        print(f"\n--- ⚠️  CHAOS MODE: Nullifying data in {table_name} (Rate: {corruption_rate}) ---")
        
        if not target_columns:
            return

        with self.engine.begin() as conn:
            for col in target_columns:
                # We use Postgres' native random() function.
                # If random() < rate (e.g. 0.1), that row gets nuked to NULL.
                query = text(f"""
                    UPDATE "{table_name}"
                    SET "{col}" = NULL
                    WHERE random() < :rate
                """)
                result = conn.execute(query, {"rate": corruption_rate})
                print(f"   -> Blasted {result.rowcount} NULLs into column '{col}'.")

    def sabotage_data(self, table_name, sabotage_type="mix_types", target_column=None):
        # Advanced Chaos: Intentionally break data types or logic.
        print(f"\n---SABOTAGE MODE: Applying '{sabotage_type}' to {table_name}.{target_column} ---")
        
        if not target_column:
            print("   -> Error: Must specify target_column.")
            return

        with self.engine.begin() as conn:
            
            if sabotage_type == "mix_types":
                # Scenario: A column that should be numbers has strings in it.
                # First, we must cast the column to TEXT type to allow strings.
                try:
                    conn.execute(text(f'ALTER TABLE "{table_name}" ALTER COLUMN "{target_column}" TYPE TEXT USING "{target_column}"::TEXT'))
                except Exception as e:
                    print(f"   -> Warning: Could not convert column type. {e}")
                
                # List of garbage strings to inject.
                garbage_values = ["N/A", "Unknown", "TBD", "missing", "Error_404"]
                
                # Randomly update 5% of rows for each garbage type.
                for trash in garbage_values:
                    conn.execute(
                        text(f'UPDATE "{table_name}" SET "{target_column}" = :trash WHERE random() < 0.05'),
                        {"trash": trash}
                    )
                print(f"   -> Column '{target_column}' is now TEXT and contains garbage values.")

            elif sabotage_type == "outliers":
                # Scenario: Valid types, but invalid logic (e.g., negative salary).
                
                # Flip 5% of values to negative.
                conn.execute(text(f'UPDATE "{table_name}" SET "{target_column}" = "{target_column}" * -1 WHERE random() < 0.05'))
                
                # Multiply 5% of values by 100 to create massive outliers.
                conn.execute(text(f'UPDATE "{table_name}" SET "{target_column}" = "{target_column}" * 100 WHERE random() < 0.05'))
                
                print(f"   -> Column '{target_column}' now has negative values and massive outliers.")

            elif sabotage_type == "typos":
                # Scenario: Text data has trailing/invalid characters.
                # We append '_???' to the end of existing strings for 10% of rows.
                conn.execute(text(f"""
                    UPDATE "{table_name}" 
                    SET "{target_column}" = "{target_column}" || '_???' 
                    WHERE random() < 0.1
                """))
                print(f"   -> Column '{target_column}' now has string corruption.")

    def drop_table(self, table_name):
        # A utility to completely remove a table from the DB.
        print(f"\n--- 🗑️  DELETING Table: {table_name} ---")
        # CASCADE ensures that any Foreign Keys pointing to this table don't stop the deletion.
        query = text(f'DROP TABLE IF EXISTS "{table_name}" CASCADE')
        
        with self.engine.begin() as conn:
            conn.execute(query)
            
        print(f"   -> Table '{table_name}' has been dropped successfully.")


# ============================= EXECUTION =============================

if __name__ == '__main__':
    
    # Initialize our generator class with the imported connection string.
    generator = PSQLDataGenerator(postgre_connect)

    # --- 1. Create 'Departments' Table ---
    # We define the schema and the random generation rules here.
    generator.create_and_populate_table(
        table_name="Departments",
        primary_key="Dept ID",
        column_config={
            "Dept ID": ["TEXT", "prefixed_id"],
            "Dept Name": ["TEXT", "rand_ch"],
            "Budget": ["INTEGER", "rand_intg"]
        },
        groups={
            "Dept ID": "dep", # Generates IDs like 'dep_Xy7z...'
            "Dept Name": ["Engineering", "HR", "Sales", "Legal"],
            "Budget": [50000, 500000] # Random integer between 50k and 500k
        },
        num_entries=10
    )

    # --- 2. Create 'Employees' Table ---
    generator.create_and_populate_table(
        table_name="Employees",
        primary_key="Emp ID",
        column_config={
            "Emp ID": ["TEXT", "prefixed_id"],
            "First Name": ["TEXT", "rand_name"],
            "Last Name": ["TEXT", "rand_name"],
            "Salary": ["INTEGER", "rand_intg"]
        },
        groups={
            "Emp ID": "emp",
            "First Name": "f", # 'f' for First name
            "Last Name": "l", # 'l' for Last name
            "Salary": [60000, 150000]
        },
        num_entries=50
    )

    # --- 3. Link Tables ---
    # This takes the IDs generated in Departments and assigns them to a new column in Employees.
    generator.inject_foreign_key_column("Departments", "Dept ID", "Employees", "department_id")

    # --- 4. Chaos Engineering (Breaking the Data) ---
    
    # A. Introduce NULLs into 'First Name' (Data Cleaning practice).
    generator.corrupt_table_data("Employees", 0.1, ["First Name"])

    # B. Turn 'Salary' into a mixed-type column with string garbage ("N/A", "Unknown").
    generator.sabotage_data("Employees", sabotage_type="mix_types", target_column="Salary")

    # C. Add logical errors (negative numbers) and outliers to 'Budget'.
    generator.sabotage_data("Departments", sabotage_type="outliers", target_column="Budget")

    # D. Add string typos (e.g., "Engineering_???") to 'Dept Name'.
    generator.sabotage_data("Departments", sabotage_type="typos", target_column="Dept Name")

    print("\n--- DONE! Your database is now a mess. Good luck cleaning it! ---")