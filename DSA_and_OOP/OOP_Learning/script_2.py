# Procedural Programming (a simple list of instructions) gets messy and hard to maintain ("spaghetti code") because data is global and separate from the functions that use it.

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

