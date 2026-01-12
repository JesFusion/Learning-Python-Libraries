-- Data Definition Language

-- creating a new table
CREATE TABLE f_table(
    un_ID SERIAL PRIMARY KEY,

    the_name VARCHAR(15) NOT NULL,

    model_type VARCHAR(45),

    model_loss REAL,

    model_pushed BOOLEAN DEFAULT FALSE,

    creation_time TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);


CREATE TABLE model_logs(
    id_of_the_log SERIAL,

    message TEXT
);


INSERT INTO model_logs ("message") VALUES ('Jesse'), ('Favour');


SELECT * FROM model_logs;

-- use TRUNCATE to delete all data in a table without deleting the table itself
TRUNCATE TABLE model_logs;





























































-- we use INSERT INTO to add new rows to a table
INSERT INTO f_table (the_name, model_type, model_loss, model_pushed, creation_time)
VALUES ('Model A', 'XGBoost', 0.2214, TRUE, '2029-10-27 10:00:00');


INSERT INTO f_table (the_name, model_type, model_pushed)
VALUES ('model B', 'XGBoost', FALSE)
RETURNING un_id, creation_time;


INSERT INTO f_table (the_name, model_type, model_loss, model_pushed)
VALUES 
		('model_c', 'sklearn', 2.233, FALSE),
		('model D', 'pytorch', NULL, TRUE),
		('Model E', 'Deep Learning', NULL, FALSE), -- this model has no loss because it crashed during training
		('Model F', 'Resnet', 0.1477, FALSE);


--we use UPDATE and SET to change change values in a row. WHERE helps us apply a condition
UPDATE f_table
SET model_loss = 0.2214,
model_type = 'Jesse is cool'
WHERE un_id = 6;

-- we use DELETE FROM to remove a row in a table
DELETE FROM f_table WHERE the_name = 'Price Predictor';


SELECT * FROM f_table;


UPDATE f_table
SET un_id = 1
WHERE un_id = 2;


































































--we use SELECT & FROM to obtain columns from a table
SELECT the_name, model_loss FROM f_table;


-- pagination enables us to load data (ie, rows) in chunks, to avoid overwhelming a system
SELECT un_id, the_name, creation_time
FROM f_table
ORDER BY creation_time ASC
LIMIT 2; -- this brings out rows 2 at a time


SELECT un_id, the_name, creation_time
FROM f_table
ORDER BY un_id ASC
LIMIT 4
OFFSET 2; -- this makes us skip the first 2 rows and take the next 2 rows 


--we use WHERE and BETWEEN to filter and obtain rows that pass requirements we set
SELECT * FROM f_table
WHERE model_loss > 0.1
AND creation_time BETWEEN '2026-01-01' AND '2030-12-31'

-- LIKE and ILIKE
SELECT * FROM f_table
WHERE model_type LIKE '%net%';

SELECT * FROM f_table
WHERE model_type ILIKE '%torch%'; -- ILIKE is case insensitive, unlike LIKE which is

SELECT * FROM f_table;


/*
 HANDLING NULLS:
 
 we don't do:
 
 column_name  = NULL
 
 instead we do:
 
 column_name IS NULL
*/

SELECT model_type, model_pushed
FROM f_table
WHERE model_loss IS NOT NULL;





