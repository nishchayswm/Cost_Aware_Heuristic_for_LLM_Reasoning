from initial_test_1 import solve

Problems = [
    ("A school is organizing a field trip. There are 4 buses, each holding 45 students. 172 students signed up, but 15 of them later cancelled. The school also needs to send 1 teacher for every 12 students who attend, rounding up. How many teachers are needed?", 14),
    ("Kushal wants two bags of milk. He can convert 30gm of protein into 1 bag of milk. One lassi contains 10gm of protein. How many lassi does Kushal need to drink?", 6),
    ("A tank is filled by pipe A at 20 liters per minute and drained by pipe B at 8 liters per minute, both running at the same time. If the tank starts empty and needs 600 liters to be full, how many minutes will it take?", 50),
    ("A shop sells pens in packs of 6 for $9, and separately as single pens for $2 each. Priya needs exactly 20 pens and wants to spend as little money as possible. How much does she spend?", 31),
    ("A train travels 60 km in the first hour, then its speed increases by 10 km/h every following hour. How far has it traveled after 4 hours in total?", 300),
]

result = {}
for i in range(len(Problems)):
    print(f"Problem {i+1}")
    print(f"Correct - {Problems[i][1] == solve(Problems[i][0])["final_answer"]}")
    print("-"*30)



