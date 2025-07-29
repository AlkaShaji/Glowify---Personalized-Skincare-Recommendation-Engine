import numpy as np
import pandas as pd

# Load the CSV file
csv_path = r"C:\Users\ALKA\OneDrive\Desktop\ULTS\glowify\dataset\Skinpro - Skinpro (3).csv"
df = pd.read_csv(csv_path)

# Check if it's loaded
print("Data loaded. Shape:", df.shape)

# Add a 'Rating' column with random float values from 3.0 to 5.0
df["Rating"] = np.round(np.random.uniform(3.0, 5.0, size=len(df)), 1)

# Show a preview
print(df.head())

# Save to new CSV file
df.to_csv(r"C:\Users\ALKA\OneDrive\Desktop\ULTS\glowify\dataset\Skinpro_with_ratings.csv", index=False)
print("Updated CSV saved successfully.")

