# Interview Questions: Week 1 - Python Fundamentals and Agile Foundations

## Beginner (Foundational)

### Q1: What are the four core values of the Agile Manifesto?
**Keywords:** Individuals, Working Software, Customer Collaboration, Responding to Change
<details>
<summary>Click to Reveal Answer</summary>

The four core values of the Agile Manifesto are:
1. **Individuals and interactions** over processes and tools
2. **Working software** over comprehensive documentation
3. **Customer collaboration** over contract negotiation
4. **Responding to change** over following a plan

These values prioritize flexibility, collaboration, and delivering value over rigid processes.
</details>

---

### Q2: What is the difference between a CVCS and a DVCS?
**Keywords:** Centralized, Distributed, Local Copy, Single Point of Failure
<details>
<summary>Click to Reveal Answer</summary>

**CVCS (Centralized Version Control System):**
- A single central server stores all versioned files
- Developers check out files from the central server
- Single point of failure (if server is down, no one can commit)
- Examples: SVN, Perforce

**DVCS (Distributed Version Control System):**
- Every developer has a complete copy of the entire repository including full history
- Operations are fast because they are local
- No single point of failure
- Examples: Git, Mercurial
</details>

---

### Q3: What is the difference between an interpreter and a compiler?
**Keywords:** Line by Line, Entire Program, Runtime, Executable
<details>
<summary>Click to Reveal Answer</summary>

**Compiler:**
- Translates the entire source code into machine code before the program runs
- Produces an executable file
- Errors are caught at compile time
- Fast execution since code is already translated
- Examples: C, C++, Go

**Interpreter:**
- Translates and executes code line by line at runtime
- No separate compilation step
- Errors appear at runtime when that line executes
- Slower execution since translation happens at runtime
- Examples: Python, JavaScript, Ruby
</details>

---

### Q4: What is indentation used for in Python, and why is it important?
**Keywords:** Code Blocks, Scope, Structure, Four Spaces
<details>
<summary>Click to Reveal Answer</summary>

In Python, indentation is used to define code blocks and structure. Unlike languages that use braces `{}`, Python uses indentation to determine:
- What code belongs inside a function, loop, or conditional
- The scope and hierarchy of code

It is important because:
- Incorrect indentation causes IndentationError
- It makes code readable and consistent
- Best practice is to use 4 spaces per indentation level
</details>

---

### Q5: What is the purpose of the `None` value in Python?
**Keywords:** Absence of Value, NoneType, Return, Null
<details>
<summary>Click to Reveal Answer</summary>

`None` is Python's null value. It represents the intentional absence of a value. Key characteristics:
- There is exactly one `None` object in Python
- Its type is `NoneType`
- Functions without an explicit return statement return `None`
- Used when a variable should have no meaningful value yet
- Best practice: Use `is None` to check for None (not `== None`)
</details>

---

### Q6: What is the difference between a list and a tuple in Python?
**Keywords:** Mutable, Immutable, Square Brackets, Parentheses
<details>
<summary>Click to Reveal Answer</summary>

| Aspect | List | Tuple |
|--------|------|-------|
| Syntax | Square brackets `[1, 2, 3]` | Parentheses `(1, 2, 3)` |
| Mutability | Mutable (can be changed) | Immutable (cannot be changed) |
| Methods | Many (`append`, `remove`, etc.) | Limited (`count`, `index`) |
| Use Cases | Dynamic collections | Fixed data, dictionary keys, function returns |
| Performance | Slightly slower | Slightly faster for iteration |
</details>

---

### Q7: What is the purpose of the `__init__` method in a Python class?
**Keywords:** Constructor, Initialize, Attributes, Self
<details>
<summary>Click to Reveal Answer</summary>

The `__init__` method is the constructor in Python classes. It:
- Runs automatically when you create a new object (instance)
- Initializes the object's attributes with values
- Takes `self` as the first parameter (referring to the instance being created)
- Can accept additional parameters for setup
- Performs any setup needed when the object is created

Example:
```python
def __init__(self, name, age):
    self.name = name
    self.age = age
```
</details>

---

### Q8: What are the three Scrum roles?
**Keywords:** Product Owner, Scrum Master, Development Team
<details>
<summary>Click to Reveal Answer</summary>

The three Scrum roles are:

1. **Product Owner:**
   - Represents the customer and stakeholders
   - Maintains and prioritizes the Product Backlog
   - Defines what features to build and in what order

2. **Scrum Master:**
   - Facilitates the Scrum process
   - Removes impediments blocking the team
   - Coaches the team on Agile practices

3. **Development Team:**
   - Cross-functional group (3-9 members) that builds the product
   - Self-organizing; they decide how to accomplish the work
   - Collectively accountable for delivering increments
</details>

---

### Q9: What is string slicing and how does it work in Python?
**Keywords:** Start, Stop, Step, Index, Substring
<details>
<summary>Click to Reveal Answer</summary>

String slicing extracts substrings using the syntax `[start:stop:step]`:
- `start`: Beginning index (inclusive, defaults to 0)
- `stop`: Ending index (exclusive, defaults to end)
- `step`: Stride/increment (defaults to 1)

Examples:
```python
text = "Python"
text[0:3]   # "Pyt" (index 0 to 2)
text[2:]    # "thon" (index 2 to end)
text[:3]    # "Pyt" (start to index 2)
text[::2]   # "Pto" (every 2nd character)
text[::-1]  # "nohtyP" (reversed)
```
</details>

---

### Q10: What is the difference between `break` and `continue` in loops?
**Keywords:** Exit, Skip, Loop Control, Iteration
<details>
<summary>Click to Reveal Answer</summary>

| Statement | Effect |
|-----------|--------|
| `break` | Exits the loop entirely; no more iterations execute |
| `continue` | Skips the rest of the current iteration and moves to the next one |

Example:
```python
for i in range(5):
    if i == 2:
        continue  # Skips 2
    if i == 4:
        break     # Stops at 4
    print(i)      # Prints: 0, 1, 3
```
</details>

---

## Intermediate (Application)

### Q11: You need to safely access a dictionary key that might not exist. How do you prevent a KeyError?
**Hint:** Think about the `.get()` method.
<details>
<summary>Click to Reveal Answer</summary>

Use the `.get()` method instead of bracket notation:

```python
person = {"name": "Alice"}

# Unsafe - raises KeyError if key missing
# email = person["email"]

# Safe - returns None if key missing
email = person.get("email")

# Safe with default value
email = person.get("email", "N/A")
```

Alternatively, check if the key exists first:
```python
if "email" in person:
    email = person["email"]
```
</details>

---

### Q12: How would you iterate over a list while also tracking the index of each element?
**Hint:** There is a built-in function designed for this purpose.
<details>
<summary>Click to Reveal Answer</summary>

Use the `enumerate()` function, which returns both the index and value:

```python
fruits = ["apple", "banana", "cherry"]

# Using enumerate (preferred)
for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")

# With custom starting index
for i, fruit in enumerate(fruits, start=1):
    print(f"{i}: {fruit}")

# Output:
# 0: apple
# 1: banana
# 2: cherry
```

This is more Pythonic than using `range(len(fruits))`.
</details>

---

### Q13: Explain the difference between using `git fetch` and `git pull`.
**Hint:** One does more than the other.
<details>
<summary>Click to Reveal Answer</summary>

| Command | What It Does |
|---------|--------------|
| `git fetch` | Downloads changes from the remote repository but does NOT merge them into your local branch |
| `git pull` | Downloads changes AND merges them into your current branch (equivalent to `git fetch` + `git merge`) |

Use `git fetch` when you want to see what changes exist before integrating them. Use `git pull` when you want to update your branch immediately.

```bash
# Fetch first to review
git fetch origin main
git log origin/main

# Or pull directly
git pull origin main
```
</details>

---

### Q14: What is the difference between instance attributes and class attributes?
**Hint:** Think about what is shared vs. what is unique.
<details>
<summary>Click to Reveal Answer</summary>

**Instance Attributes:**
- Belong to each specific object (instance)
- Each instance has its own copy
- Defined in `__init__` using `self.attribute`

**Class Attributes:**
- Belong to the class itself
- Shared by all instances
- Defined at the class level (outside any method)

Example:
```python
class Dog:
    species = "Canis familiaris"  # Class attribute (shared)
    
    def __init__(self, name):
        self.name = name  # Instance attribute (unique)

dog1 = Dog("Max")
dog2 = Dog("Buddy")
print(dog1.species)  # "Canis familiaris" (shared)
print(dog1.name)     # "Max" (unique to dog1)
```
</details>

---

### Q15: How would you write a function that accepts any number of keyword arguments?
**Hint:** Think about the `**kwargs` syntax.
<details>
<summary>Click to Reveal Answer</summary>

Use `**kwargs` to accept arbitrary keyword arguments as a dictionary:

```python
def print_info(**kwargs):
    """Accept and print any keyword arguments."""
    for key, value in kwargs.items():
        print(f"{key}: {value}")

# Usage
print_info(name="Alice", age=30, city="NYC")
# Output:
# name: Alice
# age: 30
# city: NYC
```

Similarly, `*args` accepts any number of positional arguments as a tuple.

Order in function signature: `def func(required, default=val, *args, **kwargs)`
</details>

---

### Q16: What are story points and why do Agile teams use them instead of hours?
**Hint:** Think about relative sizing.
<details>
<summary>Click to Reveal Answer</summary>

**Story points** measure the complexity, effort, and uncertainty of a task using relative sizing (not absolute time).

Teams commonly use Fibonacci numbers: 1, 2, 3, 5, 8, 13, 21

**Why use story points instead of hours:**
1. Humans are better at comparing than absolute estimation
2. Removes pressure of "hours" which vary by individual skill level
3. Velocity (points completed per sprint) becomes predictable over time
4. Focuses on complexity rather than duration
5. Prevents artificial precision (estimating to the hour is often inaccurate)
</details>

---

### Q17: How do you define and use a lambda function in Python?
**Hint:** They are anonymous, single-expression functions.
<details>
<summary>Click to Reveal Answer</summary>

Lambda functions are small, anonymous functions defined with the `lambda` keyword:

**Syntax:** `lambda arguments: expression`

```python
# Regular function
def square(x):
    return x ** 2

# Equivalent lambda
square = lambda x: x ** 2

# Common use with built-in functions
numbers = [1, 2, 3, 4, 5]

# With map
squares = list(map(lambda x: x ** 2, numbers))

# With filter
evens = list(filter(lambda x: x % 2 == 0, numbers))

# With sorted
Associates = [{"name": "Bob", "grade": 85}, {"name": "Alice", "grade": 92}]
sorted_students = sorted(Associates, key=lambda s: s["grade"])
```
</details>

---

## Advanced (Deep Dive)

### Q18: Explain Python's LEGB rule for variable scope resolution. Provide an example showing all four scopes.
<details>
<summary>Click to Reveal Answer</summary>

Python resolves variable names using the LEGB rule:

1. **L - Local:** Inside the current function
2. **E - Enclosing:** In enclosing functions (closures)
3. **G - Global:** At the module level
4. **B - Built-in:** Python's built-in names

Example demonstrating all four:
```python
x = "global"           # Global scope

def outer():
    x = "enclosing"    # Enclosing scope
    
    def inner():
        x = "local"    # Local scope
        print(len(x))  # Built-in 'len' function
        print(x)       # Prints "local"
    
    inner()
    print(x)           # Prints "enclosing"

outer()
print(x)               # Prints "global"
```

To modify outer scopes: use `global` for global variables, `nonlocal` for enclosing variables.
</details>

---

### Q19: Describe the four pillars of Object-Oriented Programming with a brief example for each.
<details>
<summary>Click to Reveal Answer</summary>

**1. Encapsulation:** Bundles data and methods, restricting direct access.
```python
class BankAccount:
    def __init__(self):
        self._balance = 0  # "Private" attribute
    
    def deposit(self, amount):
        self._balance += amount  # Controlled access
```

**2. Abstraction:** Hides complexity behind simple interfaces.
```python
class EmailSender:
    def send(self, to, message):  # Simple interface
        self._connect()    # Complex implementation hidden
        self._transmit()
```

**3. Inheritance:** Classes inherit from parent classes.
```python
class Animal:
    def speak(self): pass

class Dog(Animal):
    def speak(self):
        return "Woof!"
```

**4. Polymorphism:** Different objects respond to the same method call.
```python
def make_sound(animal):
    print(animal.speak())  # Works for any Animal subclass

make_sound(Dog())  # "Woof!"
make_sound(Cat())  # "Meow!"
```
</details>

---

### Q20: How would you implement a custom iterator class in Python? What methods must it implement?
<details>
<summary>Click to Reveal Answer</summary>

A custom iterator must implement two methods:
- `__iter__()`: Returns the iterator object itself
- `__next__()`: Returns the next value; raises `StopIteration` when exhausted

Example:
```python
class CountDown:
    """Iterator that counts down from n to 1."""
    
    def __init__(self, start):
        self.current = start
    
    def __iter__(self):
        return self  # Return itself as the iterator
    
    def __next__(self):
        if self.current <= 0:
            raise StopIteration  # Signal end of iteration
        value = self.current
        self.current -= 1
        return value

# Usage
for num in CountDown(5):
    print(num)  # 5, 4, 3, 2, 1
```

The for loop internally calls `iter()` to get the iterator, then repeatedly calls `next()` until `StopIteration` is raised.
</details>

---

### Q21: Why should you avoid using mutable default arguments in function definitions? Provide an example of the problem and its solution.
<details>
<summary>Click to Reveal Answer</summary>

**The Problem:** Default arguments are evaluated once when the function is defined, not each time it is called. Mutable defaults (like lists or dicts) are shared between all calls.

```python
# WRONG - mutable default is shared
def add_item(item, items=[]):
    items.append(item)
    return items

print(add_item("a"))  # ['a']
print(add_item("b"))  # ['a', 'b'] - Unexpected!
```

**The Solution:** Use `None` as the default and create a new object inside the function.

```python
# CORRECT - new list created each call
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items

print(add_item("a"))  # ['a']
print(add_item("b"))  # ['b'] - Correct!
```

This ensures each function call gets its own fresh list.
</details>
