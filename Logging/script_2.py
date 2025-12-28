import logging
from jesse_custom_code.pandas_file import logs_path



print('''
======================================== Standard Print ========================================
      
This is a print statement. It has no metadata. Just Text
''')

# the following won't print to the console by default...
logging.debug("1. DEBUG: Something bad happened! Check code!")
logging.info("2. INFO: Model has finished training!")

# the following will print because of their high importance...
logging.warning(" 3. RAM usage at 91%!")
logging.error(" 4. ERROR: Couldn't find CNN_v1.pkl!")
logging.critical(" 5. CRITICAL: AWS SERVER ABOUT TO SELF-DESTRUCT!")


# ===================================== The ANATOMY of a RECORD =====================================

st_rep = logging.LogRecord(
    name = "status_report",
    level = logging.ERROR,
    pathname = __file__,
    lineno = 40,
    msg = "Manual Error Creation",
    args = (),
    exc_info = None
)

print(f'''
Timestamp: {st_rep.created}

Level Name: {st_rep.levelname}

File Path: {st_rep.pathname}

Message: {st_rep.getMessage()}
''')
















































































# configuring our logging Process...

"""
We configure this ONCE at the start of the program:

1. filename: Writes to a file instead of the terminal.
2. level: We set it to DEBUG so we see EVERYTHING.
3. format: We make it look like a timestamped record.
4. filemode: 'w' overwrites the file each time. 'a' would append (add to bottom).
"""
logging.basicConfig(
    filename = f"{logs_path}oop_status_report.log",
    filemode = "w",
    level = logging.DEBUG,
    format = "%(asctime)s ::: %(name)s ::: %(levelname)s ::: %(message)s"
)

# Check 'oop_status_report.log' in the logs folder. The output is going there!


class DataPipeLine:

    """
    The Parent Class. 
    It knows how to load data, but its training is very basic.
    """

    def __init__(self, d_name): # d_name = dataset name
        self.name_of_dataset = d_name

        logging.warning(f"[Base Class] Initialized [DataPipeLine] with dataset: {self.name_of_dataset}")

    def model_train(self):
        # A generic method that the child might want to override...

        logging.info("Instantiating model training!")

    def model_save(self):

        logging.info("[DataPipeLine] Instantiating model save to drive!")

        if self.name_of_dataset == "csv":

            logging.warning("[DataPipeLine] Model about to be saved as csv file!")

            logging.debug("[DataPipeLine] Model saved as csv file.")

            logging.critical("Not sure this is right")

        else:

            logging.critical("A serious error occured! Diverting to model_v2.pkl")




class ClassicalMLPipeLine(DataPipeLine):
    """
    The Child Class.
    It inherits from DataPipeLine.
    """

    def __init__(self, the_dataset_name, model_LR):

        super().__init__(the_dataset_name) # super().__init__() activates the parents attributes sets, before activating the Child's. Here we passed the needed attribute to the Parent Class
        # It is advised to always run super().__init__() before assigning attributes to the Child class

        self.model_learning_rate = model_LR

        logging.debug(f"[ClassicalMLPipeLine] Model Hyperparameter set: lr = {self.model_learning_rate}")

        logging.error("A situation occured while fetching pre-config parameters!")

    def model_train(self):

        logging.info("[ClassicalMLPipeLine] Starting deep neural network training...")

        logging.debug(f"[ClassicalMLPipeLine] Training with learning rate = {self.model_learning_rate}")

        # we do not call super().model_train() here because we want to override it with our own custom function




# ===================================== LOGIC EXECUTION =====================================

logging.info("\n===================================== STARTING SCRIPT =====================================\n")

# Instantiating the Subclass...

the_model = ClassicalMLPipeLine("staff_logs_dataset.parquet", 0.0019) # because we called super().__init__() in ClassicalMLPipeLine, appropriate attributes will be passed to it's parent class 'DataPipeLine'


the_model.name_of_dataset = "csv"

# calling the overridden method...
the_model.model_train()

# calling the inherited method that wasn't overridden...

the_model.model_save()














