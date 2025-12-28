# Procedural Programming (a simple list of instructions) gets messy and hard to maintain ("spaghetti code") because data is global and separate from the functions that use it.
import time
import random
from abc import ABC, abstractmethod
import numpy as np
from jesse_custom_code.pandas_file import logs_path
import logging
logging.basicConfig(level = logging.INFO, format = '%(message)s')

the_name, the_balance = "Jesse", 1000

def balance_update(the_amount):
    """
    A procedural function to modify the global user_balance.

    Args:
        the_amount (_type_): 
            Collects a variable
    """

    global the_balance

    print(f'''
Updating balance for {the_name}...
    ''')

    old_balance = the_balance

    the_balance = the_balance + the_amount

    print(f'''
Old Balance: {old_balance}

New Balance: {the_balance}
    ''')


def refund_proc_func(amount_to_ref):

    """
    Another procedural function that also modifies the global state.
    """

    global the_balance # we have to reach out to the global variable (the one outside the function) to modify it

    print(f'''
Processing refund for {the_name}...
    ''')

    o_bal = the_balance

    the_balance = the_balance + amount_to_ref

    print(f'''
Old Balance (2): {o_bal}

New Balance (2): {the_balance}
    ''')



def user_display():

    """
    A function to display the global user's info.
    """

    print(f'''
============================= User Report =============================

Name: {the_name}

Balance: ${the_balance}

Report Done!
    ''')

# let's run the program

user_display()

balance_update(-137)

refund_proc_func(81)


print('============================= End of Program =============================')

user_display()

# what if we wanted two users or more?
# we'd have to create data variables for each and duplicate all functions

# debugging and maintaining code would be extremely difficult!







































































# Core OOP vocabulary...

class user_bank:

    """
    ## user_bank
    This is the blueprint for creating `User` objects.
    - It bundles the data `(attributes)` and behavior `(methods)` together.
    """

    

    def __init__(self, user_name, user_init_balance): # "self" refers to the object/instance being created

        print(f'User "{user_name}" account created successfully!')


        self.name = user_name

        self.user_balance = user_init_balance

    # other functions apart from __init__ are called methods. Think of them as the abilities of a CLASS

    def deposit_money(self, the_amount): # "self" is always the first argument in a method, so that an instance can access it's attributes and other methods

        try:
            the_amount = float(the_amount)

        except Exception as an_error:

            print(f"\nAn error occured in converting amount ${the_amount} to a number\nCheck it out: {an_error}")

            return

        if the_amount > 0: # safety for catching deposit of negative numbers

            self.user_balance = self.user_balance + the_amount

            print(f"\nSafely deposited ${the_amount} into {self.name}'s account")

            self.show_bal()

        else:

            print(f"Yo! The amount you deposited \"{the_amount}\" isn't positive. Try depositing a positive amount")

    def show_bal(self):

        print(f"User {self.name}'s balance is ${self.user_balance}")

    def withdraw_money(self, w_amount):

        try:
            w_amount = float(w_amount)

        except Exception as an_error:

            print(f"\nAn error occured in converting amount ${w_amount} to a number\nCheck it out: {an_error}")

            return

        if w_amount <= self.user_balance and w_amount > 0:

            self.user_balance = self.user_balance - w_amount

            print(f"\nAmount ${w_amount} successfully withdrawn!\nBalance: ${self.user_balance}")

        else:

            print(f"Error! Amount ${w_amount} is either less than your balance ${self.user_balance} or is negative")



    
# now, let's run the OOP code:

# let's create two objects/instances of  the "user_bank" class

# using the class automatically calls the __init__ method

print('\n============================= Running for "user_jesse" =============================\n')

user_jesse = user_bank("Jesse", 15000)


user_jesse.withdraw_money("9975k")

user_jesse.deposit_money(1230)

user_jesse.withdraw_money("15675")


print('\n============================= Running for "user_diana" =============================\n')


user_diana = user_bank("Diana", 23005)

user_diana.deposit_money("7800NGN")

user_diana.withdraw_money(19999)

user_diana.deposit_money(-119)

user_diana.deposit_money(13119)















































































# Procedural Programming


book_shelf = []

book = {
    "title": None,

    "author": None
}

def add_procedural_book(title, author):

    global book, book_shelf

    print(f'\nAdding book "{title}" to the book shelf...')

    book["title"] = title

    book["author"] = author

    book_shelf.append(book)

    return print(f"\nBook added successfully!\nHere's the shelf:\n{book_shelf}")






def find_procedural_book(title):

    global book_shelf

    print(f'\nSearching for book "{title}"...')

    for a_book in book_shelf:

        if a_book["title"] == title:

            return print(f'Book "{title}" has been found!\nHere it is\n{a_book}')


        
    return print(f"Book {title} not found. Please try another book")





add_procedural_book("Kelly", "Lingah")

add_procedural_book("Rango", "Miley")

find_procedural_book("kaild")

find_procedural_book("Rango")
    








# Object-Oriented Programming


class Library:

    def __init__(self):
        self.book_shelf = []

    
    def add_book(self, the_book):

        print(f'\nAdding book "{the_book["title"]}" to the book shelf...')

        self.book_shelf.append(the_book)


        print(f"\nBook added successfully!\nHere's the shelf:\n{self.book_shelf}")

    def find_book(self, book_title):

        print(f'\nSearching for book "{book_title}"...')

        for a_book in self.book_shelf:

            if a_book["title"] == book_title:
               
                return a_book


        return None

    def change_book_status(self, bkT,  procedure):

        if procedure == "in":
            
            for TT_book in self.book_shelf:

                if TT_book["title"] == bkT:
                    
                    TT_book["status"] = "available"

                    return print(f'\nStatus of Book "{bkT}" changed to "available"')

                else:

                    continue


        elif procedure == "out":

            
            for the_book in self.book_shelf:

                if the_book["title"] == bkT:
                    
                    the_book["status"] = "checked out"

                    return print(f'\nStatus of Book "{bkT}" changed to "checked out"')

                else:
                    
                    continue
            



class Book:

    the_library = Library()
    
    def __init__(self, book_title = None, book_author = None, book_status = "available", library = the_library):

        # assigning variables to the object...

        self.title = book_title

        self.author = book_author

        self.status = book_status

        self.the_library = library

        self.book = {
            "title": self.title,

            "author": self.author,

            "status": self.status
        }

        if self.title is not None:
            self.the_library.add_book(the_book = self.book)

      


    def check_out(self, Book_title):
        self.the_library.change_book_status(bkT = Book_title, procedure = "out")
    

    def check_in(self, Book_title):

        self.the_library.change_book_status(bkT = Book_title, procedure = "in")



    def book_search(self, bk_title):

        the_result = self.the_library.find_book(book_title = bk_title)

        if the_result:

            print(f'Book "{bk_title}" has been found!\nHere it is\n{the_result}')

        else:
            print(f"Book \"{bk_title}\" not found. Please try another book")






oriley_book = Book("O-Pyhsics", "Oriley")

disney_book = Book("Aladdin", "Disney")

zv_book = Book("Lioness", "Zonder Van", book_status = "checked out")

v_book = Book("Linak", "Zonn Lenn")

Book().book_search("Lioness")

Book().book_search("Mank")

Book().check_out("Linak")

Book().check_in("Lioness")


kinka = Library()

zv_book = Book("Bulonait", "Mandpqn", book_status = "checked out", library = kinka)

alakan = Book("Pioneer", "Zodln", library = kinka)


"""
Question 1. Look at your find_procedural_book and your Library's "find a book" method. What is the Time Complexity (Big O notation) of this search?

Answer: Linear Time

Question 2. Why does it have this Big O notation? What is the "worst-case" scenario that you are measuring?

Answer: The worst thing that could happen is having the book you're looking for at the end of a list of 10,000,000,000 elements, or not even there. I'll personally mourn your laptop

Question 3. In your own words, why is the OOP solution in Part 2 better and more "maintainable" than the procedural solution in Part 1?

Answer: Well for starters, i don't have to create a new dictionary manually each time i want to create a book, and i don't have to create a new library manually whenever i need a container for my books
"""













































# the "CLASS" keyword

class HumanBody:

    """
    A simple Jesse Class

    Currently it's set to doing absolutely nothing
    """

    pass # pass is like telling python to do nothing and just move along, but don't crash



# INSTANTIATING A CLASS:
# instantiating a class is like building something new using the class blueprint


person_1 = HumanBody()

person_2 = HumanBody()

print(f'''
Instantiated first person: {person_1}

Instantiated second person: {person_2}

Type: {type(person_1)}
''') # notice that the memory address (0x...) for the two objects are different, meaning that they're different things in memory

# let's verify if they are instances of the class...

print(f"Is person_1 a person? {isinstance(person_1, HumanBody)}")




































































class Person:

    """
    ## A class to create a person object

    ### Parameters:
    - `name`: The name of the person being created
    - `age`: The age of the person
    - `gender`: The sex of the person (male or female)
    - `height`: The height of the person
    """

    # the __init__ method is called whenever we instantiate a class

    # self is always the first parameter and refers to the object being created

    def __init__(self, name, age, gender, height):

        print(f"Creating new body for {name}...")

        self.na_me = name # interpreted as "attach the attribute 'na_me' to the object being created, it's value is gotten from the 'name' variable"
        
        self.a_ge = age
        
        self.gen_der = gender
        
        self.hei_ght = height

        self.nationality = "Nigerian" # we can also set constant values in the class

        # If we did NOT use 'self' (e.g., just 'nationality = Nigerian'), the variable
        # would act like a normal local variable: it would vanish as soon 
        # as the __init__ function finished running!

        print(f'''
Body successfully created!
Name: {self.na_me}
Age: {self.a_ge}
Gender: {self.gen_der}
Height: {self.hei_ght}
        ''')


    def grow_tall(self, increase):

        # we use self.hei_ght to access the specific object's height

        self.hei_ght = self.hei_ght + increase

        print(f"{self.na_me} just grew by {increase}m. Total height is {self.hei_ght}m")


# let's instantiate the class by creating objects

print("\n============================= Instantiating Person Objects =============================\n")

person_1 = Person("Jesse", 19, "male", 4.5)

person_2 = Person("Favour", 23, 'female', 3.1)

# each instance/object has it's own data...

print(f'''
============================= Person Objects and their data =============================

person_1 Name: {person_1.na_me}
person_1 Height: {person_1.hei_ght}

person_2 Name: {person_2.na_me}
person_2 Height: {person_2.hei_ght}
''')


# we modify object states using methods

# calling a method only changes the data for that instance

person_2.grow_tall(1.5)

print(f'''
person_1 Height: {person_1.hei_ght}

person_2 Height: {person_2.hei_ght}
''') # person_1's height remains unchanged, while that of person_2 has been modified



















































































class Item:

    def __init__(self, name, rarity):
        
        self.item_name = name

        self.item_rarity = rarity

        print(f'\nItem "{self.item_name}" created. Rarity: {self.item_rarity}')


class Inventory:

    server_region = "US-East"

    def __init__(self, p_name):
        
        self.player_name = p_name

        self.capacity = 2

        self.items = []

        self.items = list(self.items)

        print(f'\nPlayer "{self.player_name}" created in region "{Inventory.server_region}".')

    
    def add_item(self, item_object):

        new_capacity = self.capacity * 2

        if len(self.items) == self.capacity:

            print(f"Inventory full! Resizing from {self.capacity} to {new_capacity}...\n")

            self.capacity = new_capacity

        
        self.items.append(item_object)

        print(f"{self.player_name} picked up {item_object} (Capacity: {len(self.items)}/{self.capacity})")



jesse = Inventory(p_name = "Jesse")


for item_name in ["Sword", "Shield", "Potion", "Map", "Key"]:

    the_rarity = random.choice(list(range(200)))

    the_item = Item(item_name, the_rarity)

    jesse.add_item(the_item)


enemy = Inventory("Enemy")

print(f'''
Jesse server_region: {jesse.server_region}

Enemy server_region: {enemy.server_region}
''')

Inventory.server_region = "EU-West"

print(f'''
Jesse server_region: {jesse.server_region}

Enemy server_region: {enemy.server_region}
''')

"""
What is the Big O when the inventory is NOT full?
Answer: Constant Time (O(1)), because it takes the same time to add and item to the end of a list of 4 items as it does with a list of 200k items

What is the Big O when the inventory IS full and has to resize?
Answer: Linear Time (O(n)), because we first have to find the size of the list, then double it

As you add more items, how does the memory usage of your self.items list grow relative to the number of items (N)? Is it O(1), O(N), or O(N^2)?
Answer: O(N^2), because the memory size is the same when we double it, until it's full again. Memory usage is the square of input size

If you change self.server_region = "Asia" inside the add_item method, will it change the region for all players? 
Answer: False
Explain why or why not:
You're just creating a new instance attribute called "server_region" for that particular object. It doesn't change the Class attribute for all players
"""











































































# create a Simple Linear Regression class 
class SimLinReg:

    """
    ## SimpleLinearRegression
    A `class` representing a basic linear regression model: y = mx + c
    """

    # methods are special functions inside a class that gives an instance of the class certain abilities. The __init__ method is automatically run whenever you instantiate a class

    def __init__(self, l_rate):

        self.learning_rate = l_rate

        self.model_weights = 0.0

        self.model_bias = 0.0

        self.is_model_trained = False

    
    def train_model(self, iter):

        """
        ## Simulates the training process (The Behavior).
        
        Args:
            - iterations (int): How many times to update the weights.
        """

        print(f"\nStarting training for {iter} iterations...")

        # let's simulate a training loop

        for x in range(1, iter + 1):

            self.model_weights += (self.model_weights * 0.78)

            self.model_bias += (self.model_bias * 0.53)

        
        self.is_model_trained = True

        print("\nTraining Complete!")

    # a method for the model to predict a value
    def model_predict(self, val) -> float:

        """
        ### Makes a prediction using y = mx + c.
        
        Args:
            input_value (float): The input 'x'.
            
        Returns:
            float: The predicted 'y'.
        """

        if not self.is_model_trained:

            print(f"Model is not yet trained. Status: {self.is_model_trained}")

        model_pred = (self.model_weights * val) + self.model_bias

        return model_pred


# let's call the class, train and predict with a model
ai_model = SimLinReg(l_rate = 0.01)

ai_model.train_model(iter = 39)

pred_result = ai_model.model_predict(val = 7.5)

print(f"\nPrediction for input 7.5: {pred_result:.3f}")























































































class SecureModelLoad:

    def __init__(self, m_name, API_key):
        
        # PUBLIC ATTRIBUTE
        self.model_name = m_name # there was no underscore in front of this instance attribute. This means is public, meaning it can be accessed and altered by anyone


        # PROTECTED ATTRIBUTE
        self._model_status = "instantiating" # one underscore in front of an attribute protects that attribute, but it can still be accessed.
        # Think of it as a way to tell others that this attribute isn't meant to be touched, but you can if you know what you're doing


        # PRIVATE ATTRIBUTE
        self.__api_key = API_key # two underscores in front of the attribute shows we seriously want to prevent others from accessing or altering the attribute. Python mangles the attribute name so nobody can touch it

    
    def connect_hub(self):

        """
        Uses the `private key` to 'connect' (simulate connection).
        """

        print(f"Connecting to Model Hub using key: {self.__api_key}...")

        self._model_status = "connected"

    
    def check_its_status(self):

        return self._model_status # we can create a method to make others only be able to view an attribute without being able to alter it
    

# let's instantiate the class

model_loader = SecureModelLoad("Gemini-3/5", API_key = "efbu#**#24u!83t89q#$3hgnw4")

print(f'''
Loading Model: {model_loader.model_name}

Current Status (accessing directly): {model_loader._model_status}
''') # it's Possible to access protected attribues directly. But co-devs and some IDE's will warn you against it

# let's try to access the private attribute

try:

    print(f"\n{model_loader.__api_key}")

except AttributeError as an_error:

    print(f"\nAn error occured due to a private attribute being accessed. Check it out --> {an_error}\n")


# there's a secret way to access private attribues. When you create a private attribue, python mangles it to "_{class name}__{secret attribue}"

# let's try to access __api_key this way...

print(f"API key: {model_loader._SecureModelLoad__api_key}")























































































class TrainingConfiguration:

    """
    ## Configuration Management
    Manages configuration for a training run.
    Demonstrates Pythonic encapsulation using @property.
    """

    def __init__(self, l_rate, l_epochs):
        
        # we use a protected attribute to store the actual data, then use a public attribute to trigger validation and prevent check bypassing
        self._learning_rate = None

        self._learning_epochs = None

        # let's use public attribute so we can't bypass checks
        self.learning_rate = l_rate
        self.learning_epochs = l_epochs

        self._exp_ID = "experiment_000"

    
    # ============================= GETTER =============================

    # the getter is called when you want to view the value of the attribute. Setting only the getter for a method makes the method read-only, meaning it's value can't be modified

    @property
    def the_learning_rate(self):

        """
        ### Learning Rate Retrieval
        Retrieves the learning rate.
        Usage: `config.learning_rate` (no parentheses!)
        """

        return self._learning_rate


    # ============================= SETTER =============================

    # the setter is called when you want to modify the value of the attribute. It's normally used to write complex logic, that verifies the authenticity and state of the new value before modifying the attribute

    @the_learning_rate.setter
    def the_learning_rate(self, the_value):

        """
        Sets the learning rate with strict validation (The Pre-commit Hook).
        Usage: `config.learning_rate` = 0.01
        """

        print(f"Attempting to set learning_rate to {the_value}...")


        if not isinstance(the_value, (int, float)):

            raise TypeError("Error! Learning Rate must be a number!")

        
        if the_value <= 0 or the_value >= 1:

            raise ValueError("Learning Rate must be between 0 and 1!")

        
        self._learning_rate = the_value

        print("Learning rate successfully updated!")


    @property
    def epochs(self):

        return self._learning_epochs

    @epochs.setter
    def epochs(self, value):

        if not isinstance(value, int) or value <= 0:

            raise ValueError("Epochs must be a positive integer.")

        
        self._learning_epochs = value

    
    @property
    def experiment_ID(self): # experiment_ID will be set to a read-only, as we won't be creating it's setter

        """
        ## Read-Only

        This property has NO setter\n
        It is [*read-only*](https://www.merriam-webster.com/dictionary/read-only)
        """

        return self._exp_ID



# ============================= instantiating and using the class =============================


try:

    configuration = TrainingConfiguration(l_rate = 0.02, l_epochs = 15)

    print(f'''
Current Learning Rate: {configuration.the_learning_rate}
    ''')

    configuration.the_learning_rate = 0.05

    print(f'''
Current Learning Rate: {configuration.the_learning_rate}
    ''')

except Exception as an_error:

    print(f"\nError: {an_error}")
    


# try to change the value of a read-only method

try:

    configuration.experiment_ID = "experiment_001"

except Exception as an_error:

    print(f"Error occured: {an_error}")


















































































class TheBaseModel(ABC):

    """
    ### TheBaseModel
    The `Abstract Base` Class (The Contract).
    Inherits from ABC (Abstract Base Class) module.
    """

    def __init__(self, the_name, the_config: TrainingConfiguration): # the_config is of type "TrainingConfiguration" meaning we can acess attributes under the TrainingConfiguration class

        self.model_name = the_name

        self.model_configuration = the_config

        self._is_model_trained = False

    @abstractmethod
    def train_model(self, the_data):

        """
        @abstractmethod is used for defining a method that must be implemented whenever a subclass is created

        This abstract method contains no implementation and must be overridden by child classes
        """

        pass


    @abstractmethod
    def predict_model(self, the_data):

        """
        Child classes MUST override this too.
        """

        pass

    def save_the_model(self): # it's possible for abstract classes to have solid methods that are shared by all children

        print(f"Saving {self.model_name} to disk...")



class LRModel(TheBaseModel):

    """
    A concrete implementation of the blueprint.
    """

    def train_model(self, the_data):
        print(f"Training {self.model_name} with Learning rate = {self.model_configuration.learning_rate}") # we accessed the "learning_rate" attribute in the TrainingConfiguration class because it was used as a type for the abstract class "TheBaseModel"

        self._is_model_trained = True


    def predict_model(self, the_data):

        if not self._is_model_trained:

            return "Model isn't trained!"
        
        return np.random.randn(3).round(2).tolist() # These are Fake predictions, just for testing


setup_config = TrainingConfiguration(l_rate = 0.013, l_epochs = 73)


# let's try to instantiate the abstract class "TheBaseModel" directly. It'll result in an error

try:

    base_model = TheBaseModel(the_name = "Gemini", the_config = setup_config)

except TypeError as an_error:

    print(f"Cannot instantiate Abstract Base Class: {an_error}")
    

# Instantiating a derived Class

the_model = LRModel("Neural-Network-v1.2", setup_config)


the_model.train_model(np.random.randint(0, 9, size = (3)).tolist())


the_model.save_the_model()














































































# ABC = Abstract Based Class

class ThePaymentProcessor(ABC):

    def log_of_transactions(self, user_name):

        print(f'\nLogging transactions for "{user_name}" to DataBase...')

    @abstractmethod
    def payment_processing(self, input_amount):

        pass



class Stripe(ThePaymentProcessor):

    def payment_processing(self, input_amount):

        print(f'Accepting ${input_amount} for processing...')


# let's create another subclass that didn't implement the abstract method of ThePaymentProcessor
class SomeStupidFintechApp(ThePaymentProcessor):

    # Oh no! we didn't implement payment_processing()!

    def move_person_money(input_amount):

        print(f'Stealing $"{input_amount} and running away..."')



# let's try using the two sublasses

try:

    some_random_user = Stripe()

    some_random_user.log_of_transactions(user_name = "Jesse")

    some_random_user.payment_processing(55712)

    print("Payment processed successfully!\n")

except Exception as an_error:

    print(f"Yo! We got an error. Check it out --> {an_error}")



# let's try using the other subclass that didn't implement the abstract method

try:

    fake_app = SomeStupidFintechApp()

except Exception as an_error:

    print(f"Error: {an_error}")


























































































class DBConnect: # DBConnect = DataBase Connect

    """
    Class that enables one to connect to a database efforlessly
    """

    def __init__(self, database_URL):
        self.d_baseURL = database_URL
        self.is_DB_connected = False

        logging.info(f"\nParent Class Initialized generic connector for {self.d_baseURL}")


    def dbase_connect(self): # method for connecting

        self.is_DB_connected = True
        
        self.connect_status()

    def dbase_disconnect(self):

        self.is_DB_connected = False

        self.connect_status()

    
    def connect_status(self):

        if self.is_DB_connected is True:

            logging.info(f"\nParent Class connected to {self.d_baseURL} successfully")
        
        elif self.is_DB_connected is False:
            logging.info("\nParent Class disconnected!")


class SQLite(DBConnect):

    """
    The Child class. It's a database connector, but specific to SQLite.
    
    It automatically inherits dbase_connect, dbase_disconnect and connect_status
    """

    def start_engine(self):

        # the start_engine method is only available in the Subclass. The parent class doesn't inherit it, even though the subclass inherited some methods from the superclass

        logging.info(f"\nInitiating database engine and connecting to SQLite database at {self.d_baseURL}") # subclasses can access the attributes of superclasses

class PostgreSQLDataBase(DBConnect):
    pass # 'pass' means "I add nothing new, just give me exactly what my parent has."



# ===================================== Testing the classes =====================================

con_link = "->dbaselink://user:pass@localhost:5432/mydb"


# instantiating Postgres class (Subclass)...
post_dbase = PostgreSQLDataBase(con_link)

post_dbase.dbase_connect()

post_dbase.dbase_disconnect()


try:
    post_dbase.start_engine() 
    # if you tried to run the code above, an error will be thrown as PostgreSQLDataBase doesn't have the .start_engine() method. However, if we wanted PostgreSQLDataBase to inherit the method, we would make PostgreSQLDataBase a subclass of SQLite, as doing such would make it inherit SQLite's methods and also DBConnect methods (since SQLite is a subclass of DBConnect)

except Exception:
    pass

finally: # finally is used after a try/except to place code that should run whether an error was thrown during the try/except or not

    PostgreSQLDataBase.__bases__ = (SQLite,) # changing PostgreSQLDataBase superclass to SQLite
    
    post_dbase = PostgreSQLDataBase(con_link)

    post_dbase.start_engine() # it should apply the start_engine() method to PostgreSQLDataBase



# instantiating SQLite database connector (Subclass)...

sqlite_dcon = SQLite(con_link)

sqlite_dcon.start_engine()

sqlite_dcon.dbase_connect()





























































































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

































































































# ===================================== OVERRIDING & EXTENDING =====================================


class SomeRandomClass:

    '''
    The Parent Class. 
    It defines the basic structure of any model.
    '''

    def __init__(self, particular_name):

        # the parent class handles the name attribute
        self.name = particular_name
        self.is_trained = False

        print(f'''
[SomeRandomClass] ::: Initialized Random Model (Class): {self.name}
        ''')

    
    
    def model_train(self):
        # a generic placeholder method
        
        print(f'''
[SomeRandomClass --> model_train] ::: Generic Training Loop. Not much done
              
2 + 2 = {2 + 2}
        ''')

        self.is_trained = True




    
class DeepLearningClass(SomeRandomClass):
    """
    The Child Class.
    It overrides 'train' and extends '__init__'.
    """

    def __init__(self, NN_name, NN_layers):
        
        # what super().__init__() does is that it inherits it's parent's attributes 

        # we tell the class to replace particular_name with NN_name
        super().__init__(particular_name = NN_name) # self.name = NN_name

        self.model_layers = NN_layers

        print(f'''
[DeepLearningClass] ::: Added {self.model_layers} layers to the architecture
        ''')

    
    def model_train(self):
        # we overridde .model_train() method in the SomeRandomClass class, inserting our own new logic
        print(f'''
[DeepLearningClass --> model_train] ::: Spinning up GPU for {self.name}...

[DeepLearningClass --> model_train] ::: Backpropagation in Progess...
        ''')

        self.is_trained = True





# ===================================== MULTIPLE INHERITANCE & MRO =====================================

class FastAIClass:

    """
    Docstring for StupidCLass1
    """

    def ran_func(self):

        print('''
[FastAIClass] ::: Saving Logs to text file
        ''')


class FavourIsCool:

    """
    Docstring for FavourIsCool
    """

    def ran_func(self):

        print('''
[FavourIsCool] ::: Saving Data to PostgreSQL DataBase...
        ''')


class DataPipeline(FastAIClass, FavourIsCool):

    """
    The Child Class. inherits from BOTH FastAIClass and FavourIsCool.
    Notice the order: (FastAIClass, FavourIsCool)
    """

    pass # do nothing. just chill



# ===================================== SCRIPT EXECUTION =====================================

# instantiating the `DeepLearningClass` class
ai_model = DeepLearningClass(NN_name = "YOLOv1", NN_layers = 12)

print(ai_model.is_trained) # is_trained = False

# calling the method that was overridden
ai_model.model_train()

print(ai_model.is_trained) # is_trained = True


training_pipeline = DataPipeline()

# since both parent classes [FastAIClass & FavourIsCool] have the .ran_func() method, python looks from Left-to-Right and runs the FastAIClass version of the .ran_func() method first
training_pipeline.ran_func()

# we debug MRO cases using the .mro() method

print(DataPipeline.mro()) # [<class '__main__.DataPipeline'>, <class '__main__.FastAIClass'>, <class '__main__.FavourIsCool'>, <class 'object'>]

















































































# ===================================== DUCK TYPING (The Flexible Way) =====================================

'''
Two classes with no relationship with each other, only that they have a method with the same name. You can call this method from each class, even if they're not related
'''

class AmazonS3Connect:

    """
    Simulates connecting to Amazon S3.
    """

    def __init__(self, name_of_bucket):
        
        self.b_name = name_of_bucket

    def data_retrieval(self):

        print(f"[AmazonS3Connect] ::: Connecting to bucket '{self.b_name}'...")

        time.sleep(2)

        print("[AmazonS3Connect] ::: Download Complete")

        return np.linspace(12, 34, 3).round(2).tolist()


class DeviceLocalConnect:

    """
    Simulates reading a CSV from your Laptop.
    """

    def __init__(self, path):
        
        self.path_to_file = path

    def data_retrieval(self):

        print(f'''
[DeviceLocalConnect] ::: Opening Local file '{self.path_to_file}'...
        ''')

        return np.random.randint(12, 34, size = (3)).round(2).tolist()


def ingestion_pipeline(connection_interface: object):

    """
    This function represents your Data Pipeline.
    
    CRITICAL: It does NOT check 'if isinstance(connection_interface, AmazonS3Connect)'.
    It simply assumes 'connection_interface' has a .data_retrieval() method.
    This is Duck Typing.
    """

    print(f"\n===================================== Starting Ingestion with {connection_interface.__class__.__name__} =====================================\n")

    input_data = connection_interface.data_retrieval()

    print(f"Pipeline received Data: {input_data}\n")

    return input_data



# ===================================== POLYMORPHISM WITH INHERITANCE (The Strict Way) =====================================

"""
Two classes that inherits from a base class, with their different versions of overriden methods. Calling each class method makes each class run it's own method version
"""

class ClassBase:

    """
    The Parent. Defines the 'Interface' that all children must have.
    """

    def __init__(self, the_name):

        self.name = the_name

        self.model_accuracy = 0.0

    
    def model_train(self, the_data):

        # children are expected to override this

        print(f'''
[ClassBase --> {self.name}] ::: Default training...
        ''')
        
    
    def model_evaluation(self):

        # returns a random accuracy for simulation

        self.model_accuracy = random.uniform(0.70, 0.99)

        print(f'''
[ClassBase --> {self.name}] ::: Accuracy: {self.model_accuracy:.4f}
        ''')

        return self.model_accuracy
    

class RandomForestModel(ClassBase):
    """
    Child 1: Complex, slow model.
    """

    def model_train(self, the_data):

        # we're trying to override the parent's .model_train() method
        # If we don't it'll run the parents logic

        print(f'''
[RandomForestModel --> {self.name}] ::: Building {np.random.randint(100, 234)} Decision Trees on data: {the_data}
        ''')

        time.sleep(2.54)


class DeepNN(ClassBase):

    """
    Child 2: Heavy, deep model.
    """

    def model_train(self, the_data):
        
        print(f'''
[DeepNN --> {self.name}] ::: Forward and Backward Propagation on data: {the_data}
        ''')

        time.sleep(2.66)



# ===================================== Polymorphic Execution =====================================

def ML_run_grid(list_of_models: list[object], input_data):

    """
    Runs a list of different models. 
    It treats them all exactly the same because they are all 'BaseModels'.
    """

    print("\n===================================== Starting AutoML Grid Search =====================================\n")

    global HP_model # HP_model = Highest Performance Model

    highest_acc = -1.9

    for the_model in list_of_models:

        the_model.model_train(input_data)

        m_acc = the_model.model_evaluation()

        if m_acc >= highest_acc:

            highest_acc = m_acc

            
            HP_model = the_model

            print(f'''
The Winner is: {the_model.name} with {highest_acc:.4f} accuracy
        ''')




# ===================================== EXECUTION =====================================


if __name__ == '__main__':

    cloud_data = AmazonS3Connect('my-bucket-v1')

    local_data = DeviceLocalConnect('/home/jesfusion/Documents/ml/ML-Learning-Repository/Saved_Datasets_and_Models/Datasets/Bullshit_Dataset/raw_user_logs.csv')

    data_V1 = ingestion_pipeline(cloud_data)
    
    data_V2 = ingestion_pipeline(local_data)

    model_list = [
        DeepNN('Gemini-Flash-v.2'),

        RandomForestModel('Forest-v4.5'),
        
        # DeepNN('Gemini-Flash-v.2')
    ]

    # procesing them uniformly...

    ML_run_grid(input_data = data_V1, list_of_models = model_list)


































































































# ===================================== The Base Layer (Inheritance) =====================================

class BaseModel:
    """Parent class establishing the basic blueprint for all models."""
    def __init__(self, model_name, model_version):
        self.name = model_name
        self.version = model_version
        print(f'\n{self.__class__} ::: Class initiated with name: {self.name} and version: {self.version}')
    
    def deploy(self):
        print(f'\n{self.__class__} ::: Deploying {self.name} v{self.version}..')

    def predict(self, input_data):
        # Default logic for all child classes
        print('\n[BaseModel] Generic prediction...')
        model_prediction = np.random.rand(1, 1)[0][0].round(6)
        return model_prediction

# ===================================== The Specializations (Overriding & Extending) =====================================

class LegacyModel(BaseModel):
    """Inherits from BaseModel and adds specific attributes."""
    def __init__(self, sk_ML_name, sk_ML_version, sk_ML_overhead_cost):
        # Calls the Parent constructor to handle name and version
        super().__init__(model_name = sk_ML_name, model_version = sk_ML_version)
        self.overhead_cost = sk_ML_overhead_cost

    def predict(self, sk_inp_data):
        # Method Overriding: Replaces parent logic with a specific version
        print('\n[LegacyModel] CPU chugging along...')
        return 0.1

class ModernModel(BaseModel): 
    """Inherits from BaseModel and extends existing methods."""
    def deploy(self):
        # Extension: Adds new logic BEFORE calling the parent's original deploy logic
        print('\n[ModernModel] GPU warm-up sequence initiated...')
        super().deploy()

# ===================================== The Capabilities (Multiple Inheritance & Mixins) =====================================

class AWSMixin:
    """Standalone functionality (Mixin) for AWS integration."""

    def upload_to_s3(self, filename):
        self.AWS_file = filename
        print(f'\n[AWSMixin] Uploading {self.AWS_file} to S3...')

class AzureMixin:
    """Standalone functionality (Mixin) for Azure integration."""
        
    def push_to_blob(self, filename):
        self.AZURE_file = filename
        print(f'\n[AzureMixin] Pushing {self.AZURE_file} to Blob Storage...')

class HybridModel(ModernModel, AWSMixin, AzureMixin):
    """Multiple Inheritance: Combines logic from three different classes."""
    def __init__(self, hyb_model_name, hyb_model_version):
        super().__init__(model_name = hyb_model_name, model_version = hyb_model_version)

# ===================================== The Chaos (Polymorphism & Duck Typing) =====================================

class ChaosScript:
    """A completely unrelated class that still has a 'deploy' method."""
    def deploy(self):
        print('\n[ChaosScript] I do what I want!')

# ===================================== The Execution Engine =====================================

def universal_deployer(object_list: list = []):
    """
    Polymorphism: Treats different objects the same way based on shared method names.
    """
    for model in object_list:
        # 'Duck Typing': If it looks like a model (has deploy), treat it like a model
        if hasattr(model, 'deploy') is True:
            model.deploy()
        else:
            continue

if __name__ == '__main__':
    # Initialize instances
    hyb_model = HybridModel(hyb_model_name ='hyb-gemini', hyb_model_version = '2.1.1')

    model_list = [
        hyb_model,

        LegacyModel(sk_ML_name = 'KNN-oppo', sk_ML_overhead_cost = '$44335', sk_ML_version = '1.2.0'),

        ChaosScript() # Unrelated object works because of the shared 'deploy' method
    ]

    # Run deployment logic
    universal_deployer(object_list = model_list)

    # Use Mixin capabilities
    hyb_model.upload_to_s3(filename = 'file1.csv')
    hyb_model.push_to_blob(filename = 'file2.csv')




























































































# ===================================== OPERATOR OVERLOADING (Polymorphism in Disguise) =====================================


class D2Vector:

    """
    A class representing a 2D mathematical vector (x, y).
    Demonstrates how to overload +, *, and ==.
    """
        
    def __init__(self, x_value, y_value):
        
        self.x_axis = x_value
        
        self.y_axis = y_value

    @classmethod
    def type_verification(self, item, type, error: str):

        if not isinstance(item, type):

            raise TypeError(error)

    def __add__(self, value):

        """
        WHAT: Defines behavior for 'vector_a + vector_b'.
        WHY: Python translates 'a + b' to 'a.__add__(b)'.
        """
        
        self.type_verification(item = value, type = D2Vector, error = 'Can only add Vector2D to Vector2D')
        
        return D2Vector(self.x_axis + value.x_axis, self.y_axis + value.y_axis)
    
    def __mul__(self, scalar_value):

        """
        WHAT: Defines behavior for 'vector * number'.
        WHY: Allows scaling the vector (e.g., v * 3).
        """

        self.type_verification(item = scalar_value, type = (int, float), error = "Can only multiply Vector2D by a number")

        return D2Vector(
            self.x_axis * scalar_value, self.y_axis * scalar_value
        )
    
    
    def __eq__(self, value):
        """
        WHAT: Defines behavior for 'vector_a == vector_b'.
        WHY: Default Python equality checks if they are the SAME object in memory.
             We want to check if they have the SAME values.
        """

        if not isinstance(value, D2Vector):

            return False
        
        return self.x_axis == value.x_axis and self.y_axis == value.y_axis
    

    # string representation
    def __str__(self):
        # this is the Human-readable output for print()
        return f"Vector({self.x_axis}, {self.y_axis})"
    
    def __repr__(self):

        # WHAT: Unambiguous dev output. 
        # Ideally, this string IS valid Python code to recreate the object.
        
        return f'D2Vector(x_axis = {self.x_axis}, y_axis = {self.y_axis})'




# ===================================== CONTAINER EMULATION (Making Lists Smarter) =====================================


class ModelLogs:

    """
    A wrapper around a list of log messages.
    Demonstrates __len__, __getitem__, and __setitem__.
    """

    def __init__(self, ep_name):
        self.name_of_experiment = ep_name

        self._logs = [] # private list in the class


    def addAlog(self, log_message):

        self._logs.append(log_message)

    
    # ===================================== MAGIC METHODS =====================================

    def __len__(self):

        # This allows len(my_logs).
        # Standard Python behavior.

        return len(self._logs)
        
    

    def __getitem__(self, index_number):
        """
        WHAT: Allows my_logs[0] or my_logs[-1].
        WHY: Makes the object iterable and indexable.
        
        DSA NOTE:
        Accessing a list by index is O(1) Constant Time.
        """

        return f'[{self.name_of_experiment}] ::: {self._logs[index_number]}'
    
    def __setitem__(self, the_index, the_value):

        """
        WHAT: Allows my_logs[0] = "New Value".
        WHY: Mutable access.
        """

        print(f"Overwriting log at index {the_index}...")

        self._logs[the_index] = the_value



# ===================================== EXECUTION =====================================


if __name__ == '__main__':

    vector_a = D2Vector(2, 4)

    vector_b = D2Vector(-1, 3)

    vector_c = vector_a + vector_b # this calls the __add__ method
    # vector_a.__add__(vector_b)

    vector_d = vector_c * 4.2 # calls the __mul__ method
    # vector_c.__mul__(4.2)

    vector_r = D2Vector(2, 4)

    print(f'''
======================================== OPERATOR OVERLOADING ========================================
          
Addition: {vector_a} + {vector_b} = {vector_c}

Multiplication: {vector_a} * 3 = {vector_d}

Equality check (vector_a = vector_r): {vector_a == vector_r}

STR Output: {str(vector_a)}

REPR Output: {repr(vector_a)}
    ''')


    logs_of_exp = ModelLogs(ep_name = 'Experiment-v.1.0.3')

    log_no = [0.9, 0.5, 0.2]

    for num in log_no:

        logs_of_exp.addAlog(log_message = f'Epoch {log_no.index(num) + 1}: Loss {num}')

    
    print('''
======================================== CONTAINER EMULATION ========================================
    ''') 
    
    print(f"Total Logs: {len(logs_of_exp)}") # we can call len() on our object because we defined the __len__ method
    
    
    print(f'First Log: {logs_of_exp[0]}\n')# indexing is possible because of the __getitem__ method

    logs_of_exp[2] = 'Epoch 3: Loss 0.923' # (CORRECTED)

    print(f"Modified Log: {logs_of_exp[2]}\n") # __setitem__

    print("Iterating through loops:\n")

    for a_log in logs_of_exp:

        print(a_log)



