import pickle
with open(f"lab.pickle", "wb") as file:
    pickle.dump({"bg": [1, 2, 3], "sg": [4, 5, 6]}, file)
