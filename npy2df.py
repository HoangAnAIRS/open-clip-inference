import numpy as np
import pandas as pd

# Load the NumPy array
item_id = np.load("item_id.npy")

# Create a pandas DataFrame with the column 'item_id'
df = pd.DataFrame(item_id, columns=['item_id'])

df.to_csv("000.csv", index = False)