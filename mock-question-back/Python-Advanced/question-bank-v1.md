# Interview Questions: Week 2 - Python Advanced

## Beginner (Foundational)

### Q1: What is a Python module and how do you create one?
**Keywords:** Python File, Reusable Code, Import, .py
<details>
<summary>Click to Reveal Answer</summary>

A module is simply a Python file (`.py`) containing definitions and statements. Any Python file you create is a module.

To create a module:
1. Create a file like `math_utils.py`
2. Add functions, classes, or variables
3. Import it in another file with `import math_utils`

Modules allow you to:
- Organize code logically
- Reuse code across projects
- Avoid naming conflicts through namespaces
</details>

---

### Q2: What is the purpose of the `if __name__ == "__main__":` pattern?
**Keywords:** Main, Import, Direct Execution, Side Effects
<details>
<summary>Click to Reveal Answer</summary>

This pattern allows a module to behave differently when run directly vs. when imported:

```python
if __name__ == "__main__":
    main()
```

- When run directly: `__name__` is `"__main__"`, so `main()` executes
- When imported: `__name__` is the module name, so `main()` does NOT execute

This prevents unintended side effects when importing a module.
</details>

---

### Q3: What is the difference between `json.loads()` and `json.load()`?
**Keywords:** String, File, Parse, Deserialize
<details>
<summary>Click to Reveal Answer</summary>

| Function | Input | Use Case |
|----------|-------|----------|
| `json.loads()` | JSON **string** | Parse JSON from a variable or API response |
| `json.load()` | **File** object | Parse JSON from a file |

Similarly:
- `json.dumps()` creates a JSON string
- `json.dump()` writes JSON to a file

The "s" stands for "string" - functions with "s" work with strings.
</details>

---

### Q4: What are the main exception handling keywords in Python and their purposes?
**Keywords:** try, except, else, finally, raise
<details>
<summary>Click to Reveal Answer</summary>

| Keyword | Purpose |
|---------|---------|
| `try` | Wrap code that might raise an exception |
| `except` | Handle specific exceptions |
| `else` | Runs if no exception occurred |
| `finally` | Always runs (cleanup) |
| `raise` | Throw an exception intentionally |

Example structure:
```python
try:
    risky_operation()
except SpecificError:
    handle_error()
else:
    success_action()
finally:
    cleanup()
```
</details>

---

### Q5: What are the different file modes in Python and when would you use each?
**Keywords:** r, w, a, b, Read, Write, Append, Binary
<details>
<summary>Click to Reveal Answer</summary>

| Mode | Description | Use Case |
|------|-------------|----------|
| `'r'` | Read (default) | Reading existing files |
| `'w'` | Write (truncates) | Creating new files or overwriting |
| `'a'` | Append | Adding to existing files |
| `'x'` | Exclusive creation | Creating new files (fails if exists) |
| `'b'` | Binary mode | Images, executables, non-text |
| `'+'` | Read and write | Updating files |

Modes can be combined: `'rb'` for binary read, `'w+'` for read and write.
</details>

---

### Q6: What is a context manager and why should you use one for file operations?
**Keywords:** with, Automatic Cleanup, Close, Exception Safe
<details>
<summary>Click to Reveal Answer</summary>

A context manager (using the `with` statement) automatically handles setup and cleanup:

```python
with open('file.txt', 'r') as file:
    content = file.read()
# File is automatically closed here
```

Benefits:
- File is guaranteed to close, even if exceptions occur
- No need to remember to call `close()`
- Cleaner, more readable code
- Exception-safe resource management
</details>

---

### Q7: What is pytest and how does it differ from Python's built-in unittest?
**Keywords:** Simple Assert, Fixtures, Automatic Discovery, Detailed Output
<details>
<summary>Click to Reveal Answer</summary>

pytest is a testing framework that offers:
- **Simple assertions**: Uses plain `assert` statements (no special methods)
- **Automatic test discovery**: Finds test files and functions automatically
- **Detailed failure output**: Shows exactly why assertions failed
- **Fixtures**: Provides reusable setup/teardown with `@pytest.fixture`
- **Parameterized tests**: Run same test with different data

Example:
```python
def test_add():
    assert add(2, 3) == 5  # Simple assert, no self.assertEqual
```
</details>

---

### Q8: How do you create a NumPy array and what are its key properties?
**Keywords:** np.array, shape, dtype, ndim, size
<details>
<summary>Click to Reveal Answer</summary>

Create arrays:
```python
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
arr2d = np.array([[1, 2], [3, 4]])
```

Key properties:
```python
arr.shape   # (5,) - dimensions
arr.ndim    # 1 - number of dimensions
arr.size    # 5 - total elements
arr.dtype   # int64 - data type
```

Creation functions: `np.zeros()`, `np.ones()`, `np.arange()`, `np.linspace()`
</details>

---

### Q9: What is the Counter class and how is it better than using a regular dictionary for counting?
**Keywords:** collections, Counting, most_common, No KeyError
<details>
<summary>Click to Reveal Answer</summary>

Counter is a dictionary subclass designed for counting:

```python
from collections import Counter

words = ['apple', 'banana', 'apple']
count = Counter(words)  # Counter({'apple': 2, 'banana': 1})
```

Advantages over dictionary:
- **No KeyError**: Missing keys return 0
- **most_common(n)**: Get top n elements easily
- **Arithmetic**: Supports +, -, &, | operations
- **Cleaner code**: One line instead of loop with conditionals
</details>

---

### Q10: What is the difference between `read()`, `readline()`, and `readlines()` for file reading?
**Keywords:** Entire File, Single Line, List of Lines, Memory
<details>
<summary>Click to Reveal Answer</summary>

| Method | Returns | Memory Use |
|--------|---------|------------|
| `read()` | Entire file as one string | High for large files |
| `readline()` | One line at a time | Low |
| `readlines()` | List of all lines | High for large files |

For large files, iterate directly:
```python
for line in file:  # Memory efficient
    process(line)
```
</details>

---

## Intermediate (Application)

### Q11: How would you handle multiple types of exceptions differently in Python?
**Hint:** Think about catching specific exceptions in sequence.
<details>
<summary>Click to Reveal Answer</summary>

Use multiple except blocks or a tuple:

**Different handling:**
```python
try:
    result = data[index] / value
except IndexError:
    print("Index out of range")
    result = None
except ZeroDivisionError:
    print("Cannot divide by zero")
    result = 0
except TypeError:
    print("Invalid type")
    result = None
```

**Same handling:**
```python
try:
    risky_operation()
except (IndexError, ZeroDivisionError, TypeError) as e:
    print(f"Error: {e}")
    result = None
```

Best practice: Catch specific exceptions, not broad `Exception`.
</details>

---

### Q12: Explain how pytest fixtures work and when you would use them.
**Hint:** Think about reusable setup code.
<details>
<summary>Click to Reveal Answer</summary>

Fixtures provide reusable setup (and teardown) for tests:

```python
import pytest

@pytest.fixture
def sample_list():
    return [1, 2, 3, 4, 5]

def test_sum(sample_list):  # Fixture injected as parameter
    assert sum(sample_list) == 15

def test_length(sample_list):
    assert len(sample_list) == 5
```

Use fixtures when:
- Multiple tests need the same data
- Setup requires database connections, files, or complex objects
- Cleanup is needed (use `yield` for teardown)

Share fixtures across files with `conftest.py`.
</details>

---

### Q13: What is broadcasting in NumPy and why is it useful?
**Hint:** Think about operations between arrays of different shapes.
<details>
<summary>Click to Reveal Answer</summary>

Broadcasting allows NumPy to perform operations on arrays of different shapes:

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

# Add scalar to array (scalar broadcast to all elements)
arr + 10
# [[11, 12, 13], [14, 15, 16]]

# Add 1D array to 2D array (row broadcast to all rows)
row = np.array([1, 0, -1])
arr + row
# [[2, 2, 2], [5, 5, 5]]
```

Benefits:
- Eliminates loops for element-wise operations
- Memory efficient (no copies)
- Concise, readable code
</details>

---

### Q14: How would you create and use a custom exception in Python?
**Hint:** Custom exceptions inherit from Exception.
<details>
<summary>Click to Reveal Answer</summary>

Create custom exceptions by inheriting from Exception:

```python
class InsufficientFundsError(Exception):
    """Raised when account has insufficient funds."""
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        super().__init__(f"Cannot withdraw ${amount}. Balance: ${balance}")

def withdraw(balance, amount):
    if amount > balance:
        raise InsufficientFundsError(balance, amount)
    return balance - amount

try:
    withdraw(100, 150)
except InsufficientFundsError as e:
    print(e)  # Cannot withdraw $150. Balance: $100
    print(f"Short by: ${e.amount - e.balance}")
```
</details>

---

### Q15: Explain the difference between vectorized operations and Python loops in NumPy.
**Hint:** Think about performance.
<details>
<summary>Click to Reveal Answer</summary>

**Vectorized operations** apply to entire arrays without explicit loops:

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# Vectorized (fast)
result = arr * 2  # [2, 4, 6, 8, 10]

# Loop (slow)
result = [x * 2 for x in arr]
```

**Why vectorized is faster:**
- Operations run in optimized C code
- No Python loop overhead
- Memory-efficient
- Typically 10-100x faster for large arrays

Always prefer NumPy operations over Python loops when working with arrays.
</details>

---

### Q16: How do you parse JSON that contains a non-serializable type like datetime?
**Hint:** Think about preprocessing or custom encoders.
<details>
<summary>Click to Reveal Answer</summary>

Two approaches:

**1. Preprocess before serialization:**
```python
from datetime import datetime
import json

data = {
    "event": "Meeting",
    "date": datetime.now().isoformat()  # Convert to string
}
json.dumps(data)
```

**2. Custom encoder class:**
```python
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

data = {"event": "Meeting", "date": datetime.now()}
json.dumps(data, cls=DateTimeEncoder)
```
</details>

---

### Q17: What is the pathlib module and why use it over os.path?
**Hint:** Think about object-oriented path handling.
<details>
<summary>Click to Reveal Answer</summary>

pathlib provides an object-oriented approach to file paths:

```python
from pathlib import Path

path = Path('data/reports/summary.txt')

# Path components
path.name      # 'summary.txt'
path.stem      # 'summary'
path.suffix    # '.txt'
path.parent    # Path('data/reports')

# Operations
path.exists()
path.is_file()
path.read_text()
path.write_text("content")

# Create directories
Path('new/folder').mkdir(parents=True, exist_ok=True)
```

Benefits over os.path:
- Cleaner, more readable syntax
- Cross-platform compatibility
- Method chaining
- Integrated file reading/writing
</details>

---

## Advanced (Deep Dive)

### Q18: Explain how logging levels work in Python and how you would configure logging for a production application.
<details>
<summary>Click to Reveal Answer</summary>

**Logging Levels (in order of severity):**
- `DEBUG`: Detailed information for debugging
- `INFO`: Confirmation that things work
- `WARNING`: Something unexpected (still works)
- `ERROR`: Serious problem, some functionality failed
- `CRITICAL`: Program may not be able to continue

**Production Configuration:**
```python
import logging

logging.basicConfig(
    level=logging.INFO,  # Set minimum level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()  # Also print to console
    ]
)

logger = logging.getLogger(__name__)
logger.info("Application started")
logger.error("Database connection failed")
```

Best practices:
- Use `DEBUG` in development, `INFO` or `WARNING` in production
- Use `__name__` for logger names (module-based)
- Never log sensitive data (passwords, tokens)
</details>

---

### Q19: Describe how regular expressions work in Python. Provide an example of pattern matching and substitution.
<details>
<summary>Click to Reveal Answer</summary>

Regular expressions (regex) provide pattern matching for strings:

**Basic Operations:**
```python
import re

text = "Contact: alice@example.com or bob@test.org"

# Find all matches
emails = re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', text)
# ['alice@example.com', 'bob@test.org']

# Search for first match
match = re.search(r'(\w+)@(\w+)\.(\w+)', text)
if match:
    print(match.group(0))  # Full match
    print(match.group(1))  # First group (alice)

# Substitution
cleaned = re.sub(r'\d+', 'X', 'Order 12345 shipped')
# 'Order X shipped'
```

**Common Patterns:**
- `\d`: Digit
- `\w`: Word character (alphanumeric + underscore)
- `\s`: Whitespace
- `+`: One or more
- `*`: Zero or more
- `[]`: Character class
- `()`: Capture group
</details>

---

### Q20: How would you implement a custom iterator and what methods must it define?
<details>
<summary>Click to Reveal Answer</summary>

A custom iterator must implement `__iter__` and `__next__`:

```python
class CountDown:
    def __init__(self, start):
        self.current = start
    
    def __iter__(self):
        return self  # Return the iterator object
    
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

**How iteration works internally:**
1. `for` calls `iter()` on the object (invokes `__iter__`)
2. Repeatedly calls `next()` (invokes `__next__`)
3. Stops when `StopIteration` is raised
</details>

---

### Q21: Explain pytest parameterization and when you would use it over writing separate tests.
<details>
<summary>Click to Reveal Answer</summary>

Parameterization runs the same test with different inputs:

```python
import pytest

@pytest.mark.parametrize("input,expected", [
    (2, True),
    (3, False),
    (0, True),
    (-2, True),
    (-3, False),
])
def test_is_even(input, expected):
    assert is_even(input) == expected
```

**Benefits:**
- Each parameter set runs as a separate test
- Easier to add new test cases
- Clearer test output (shows which inputs failed)
- Less code duplication

**When to use:**
- Same logic, different inputs
- Testing edge cases systematically
- Validating multiple valid inputs

**When to write separate tests:**
- Different logic or assertions
- Tests need different fixtures
- Failure of one should not affect others' results
</details>
