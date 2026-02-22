import pandas as pd

df = pd.read_pickle("result_df_full_year_2020.pkl")

row = 0  # change this
ts = df.loc[row, "interval_15m"]
tm = df.loc[row, "tasks_matrix"]

print("interval:", ts)
print("tasks count:", len(tm))

for i, task in enumerate(tm[:10]):   # first 10 tasks
    print(i, task)