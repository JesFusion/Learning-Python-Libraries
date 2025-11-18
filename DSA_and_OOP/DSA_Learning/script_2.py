
import time


print("\n============================= Practicing Python's Built-in Data Structures =============================\n")

# let's practice the different data structures python provided us...

# 1. Lists:
# Lists are ordered, changeable and allows duplicate values

a_list = ["Jesse", "Favour", "Caleb"]

print(f'''
============================= Lists =============================

Original List: {a_list}

First Item (Index 0): {a_list[0]}
''')

# let's change the last Item

a_list[-1] = "Goodness"

# add a new item...

a_list.append("Chiedozie")
a_list.append("Jesse")

print(f'Final List: {a_list}')



# 2. Tuples:
# Tuples are ordered, unchangeable and allows duplicates

a_tuple = (10, 30, 40, 45, 67)

print(f'''
Original Tuple: {a_tuple}

First Item (index 0): {a_tuple[0]}
''')

# let's try to change the tuple...

try:
    a_tuple[1] = 39 # this will fail and flag an error, which we'll handle below

except TypeError as an_error:

    print(f"Told ya! The tuple can't be changed --> \"{an_error}\"")


# 3. Dictionaries:

# dictionaries store data in pairs (e.g --> "name": "Jack Ma"), enable fast lookups through "keys" and are changeable



# let's create a dictionary using my student data...

user_info = {
    "Name": "Nwachukwu Jesse",

    "Age": 19,

    "Department": "Mechatronics Engineering",

    "Reg. No": "20221329023"
}


print(f'''
Original Dictionary: {user_info}

Student Department: {user_info['Department']}
''')

# changing a value
user_info["Age"] = 20

# adding a new pair

user_info['Gender'] = "Male"

print(f"Final Dictionary: {user_info}")



# 4. Sets

# sets are unordered, changeable and don't allow duplicates


# let's try creating a set from a list that has duplicates and see what would happen

t_list = ["Enugu", "Anambra", "Ebonyi", "Lokoja", "Lekki", "Enugu", "Lekki"]

a_set = set(t_list) # all duplicates will be removed the moment it's converted to a set

print(f'''
Original List with duplicates: {t_list}

List converted to set (duplicates removed): {a_set}
''')

# let's check if "Enugu" is in the set

status = "Enugu" in a_set


print(f'''
Is "Enugu" in the set? {status}

Is "Zamfara" in the set? {"Zamfara" in a_set}
''')


# we can add new items to sets
a_set.add("Zamfara")

a_set.remove("Enugu") # "remove" raises an error if the item to be removed doesn't exist in the set initially


a_set.discard("Kano") # "discard" does nothing if the item to be removed doesn't exist in the set initially


print(f'''
Final Set after adding and removing items: {a_set}
''')












































































# ==> Big O Analysis


# O(1) --> Constant Time
def ac_first_item(a_list: list) -> str:

    # This function returns the first item from a list

    if not a_list:
        return None
    
    return a_list[0] # this returns the first item of any list


S_list = [True, "a", "f", 4.5]

L_list = list(range(1000000))

# it takes the same amount of time to get the first item from the small and large list

print(f'''
============================= O(1) --> Constant Time =============================
First Item from small list: {ac_first_item(S_list)}

First Item from large list: {ac_first_item(L_list)}
''')


# O(2) --> Linear Time

# here the runtime grows linearly with the number of inputs. As input size increases, the run time increases

def item_find_in_list(a_list: list, find_item):

    """Searches a list for an item by checking one by one.
    This is called a "Linear Search."
    """

    # go through each element of the list one by one. If you find the item, return True. If not, return False

    # Notice we didn't use an else statement for returning False. This is because when the for loop returns True, it immediately exits the function 

    for an_item in a_list:

        if an_item == find_item:

            return True

    return False # return false if you didn't find anything

s_t_s = time.time()

print(f'''
============================= O(n) --> Linear Time =============================

Searching small list (n = {len(S_list)}): {item_find_in_list(S_list, 4.5)}
''')

s_t_e = time.time()

print(f"Searching large list (n = {len(L_list)}): {item_find_in_list(L_list, -1)}")

l_t_e = time.time()


print(f'''
Small list time duration: {s_t_e - s_t_s}

Large list time duration: {l_t_e - s_t_e}
''') # notice that the larger list took a longer time to go through it entirely. It may be a fraction of a second, but when handling large databases, it'll take longer than that





































































# Examining the speed difference between different Big O Notations



# ============================= O(1) - Constant Time =============================

# in constant Time, the run time is always the same, regardless of the input size

def ex_first_item(the_list: list): # ex for extract

    return the_list[0]



# ============================= O(n) - Linear Time =============================

# in Linear Time, the run-time grows linearly with the input size

def list_search_linear(the_list: list, the_item):

    no_steps = 0

    for an_item in the_list:

        no_steps += 1

        if an_item == the_item:

            return True, no_steps
        
    return False, no_steps




# ============================= O(n^2) - Quadratic Time =============================

# in Quadratic time, the run-time is the square of the input-size (input-size^2)
# if n = 5, t = 25 (5 ^ 2)

def element_match(the_list: list):

    no_of_steps = 0

    no_of_elements = len(the_list)

    # the outer loop runs with the input size number

    for x in range(no_of_elements):

        # the inner loop runs with the input size number for each run in the outer loop

        for y in range(no_of_elements):

            no_of_steps += 1

    return no_of_steps




# ============================= O(2^n) - Exponential Time =============================

# in exponential time, the run-time doubles with the addition of a new element. You might wanna avoid this notation when your input size is large

def fibonacci_recursive(input_size):

    if input_size <= 1:

        return input_size

    return fibonacci_recursive(input_size - 1) + fibonacci_recursive(input_size - 2)




# ============================= running each function =============================


small_list = [9, 0, "Jesse", None]

large_list = list(range(1000000))

time_1 = time.time()

f_small = ex_first_item(small_list)




time_2 = time.time()

print(f"First item from large list: {f_small}\nTime: {time_2 - time_1}")

f_large = ex_first_item(large_list)




time_3 = time.time()

print(f"\nFirst item from small list: {f_large}\nTime: {time_3 - time_2}")


the_steps = element_match(small_list)




time_4 = time.time()

print(f"\nO(n^2) steps  for small list: {the_steps}\nTime: {time_4 - time_3}")



l_steps = element_match(list(range(1000)))

time_5 = time.time()


print(f"\nO(n^2) steps  for large list: {l_steps}\nTime: {time_5 - time_4}") # notice that it's larger than that for the small list


s_fibo = fibonacci_recursive(10)

time_6 = time.time()

l_fibo = fibonacci_recursive(30)

time_7 = time.time()

print(f'''
Fibonacci result for small value: {s_fibo}
Time: {time_6 - time_5}


Fibonacci result for large value: {l_fibo}
Time: {time_7 - time_6}
''') # notice the astronomical difference between the time for the small and the large value