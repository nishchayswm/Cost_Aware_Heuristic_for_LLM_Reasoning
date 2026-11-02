from heuristic import R, T, C ,final

print(R({"confidence": 0.92}))

print(T(["Bob buys the hammer", "Bob is a builder"], "Is Bob the builder?"))

candidates = [
    "3 times 12 apples equals 36 apples total",
    "Sarah sells apples at the market on weekends",
    "36 minus 15 equals 21 apples remaining",
    "the answer is 21",
]
goal = "Sarah boxes apples sells apples how many apples left"

print(T(candidates, goal))
