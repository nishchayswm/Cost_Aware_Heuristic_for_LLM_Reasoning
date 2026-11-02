"""
Benchmark test dataset for Cost-Aware Heuristic Search for LLM Reasoning.
Includes:
- STANDARD_PROBLEMS: Multi-step elementary math problems (GSM8K style)
- HARD_PROBLEMS: Multi-step reasoning problems with constraint optimization, rates, or systems of equations
"""

STANDARD_PROBLEMS = [
    ("Sarah has 3 boxes of apples with 12 apples each. She sells 15 apples. How many are left?", 21),
    ("A baker makes 8 trays of 6 cupcakes each. He sells 30 cupcakes. How many cupcakes does he have left?", 18),
    ("Tom has $50. He buys 4 notebooks that cost $6 each. How much money does he have left?", 26),
    ("A school has 5 classrooms with 24 students in each. If 18 students are absent today, how many students are present?", 102),
    ("Maria reads 15 pages of a book every day. After 6 days, how many pages has she read?", 90),
    ("A farmer has 7 rows of corn with 20 plants in each row. If 25 plants die, how many plants are left?", 115),
    ("A store had 200 shirts. It sold 45 shirts on Monday and 38 shirts on Tuesday. How many shirts are left?", 117),
    ("Jake has 4 packs of pencils with 10 pencils each. He gives 12 pencils to his friends. How many pencils does he have left?", 28),
    ("A parking lot has 9 rows with 15 cars in each row. If 40 cars leave, how many cars remain?", 95),
    ("A library has 6 shelves with 35 books on each shelf. If 90 books are checked out, how many books remain on the shelves?", 120),
]

HARD_PROBLEMS = [
    ("A school is organizing a field trip. There are 4 buses, each holding 45 students. 172 students signed up, but 15 of them later cancelled. The school also needs to send 1 teacher for every 12 students who attend, rounding up. How many teachers are needed?", 14),
    ("Kushal wants two bags of milk. He can convert 30gm of protein into 1 bag of milk. One lassi contains 10gm of protein. How many lassi does Kushal need to drink?", 6),
    ("A tank is filled by pipe A at 20 liters per minute and drained by pipe B at 8 liters per minute, both running at the same time. If the tank starts empty and needs 600 liters to be full, how many minutes will it take?", 50),
    ("A shop sells pens in packs of 6 for $9, and separately as single pens for $2 each. Priya needs exactly 20 pens and wants to spend as little money as possible. How much does she spend?", 31),
    ("A train travels 60 km in the first hour, then its speed increases by 10 km/h every following hour. How far has it traveled after 4 hours in total?", 300),
    ("You have 30 coins in dimes and quarters. Their total value is $5.10. Find the number of dimes you have.", 16),
]

PROBLEMS = STANDARD_PROBLEMS
