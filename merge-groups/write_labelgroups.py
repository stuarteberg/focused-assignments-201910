import sys

from tqdm import tqdm

import numpy as np
import pandas as pd

# Input file is has two columns:
# 
#   new-id,string-of-merges
#
# or, in cases where only one body exists in the right-hand column,
# it is given as a single int (not a string):
# 
#   new-id,old-id
#
# Example:
# 
#   123,"234,345,456,567"
#   321,"890,789"
#   432,"645,765,876"
#   789,234
#
input_csv = sys.argv[1]
output_csv = sys.argv[2] # e.g. label-groups.csv

input_df = pd.read_csv(input_csv, names=['new_body', 'merges_str'])

# Evaluate as Python syntax to convert from string to list
input_df['bodies'] = input_df['merges_str'].apply(eval)

# Convert any singletons to list, too.
input_df['bodies'] = input_df['bodies'].apply(lambda x: x if isinstance(x, tuple) else [x])

group_cols = []
for row in tqdm(input_df.itertuples(), total=len(input_df)):
    for body in row.bodies:
        group_cols.append((row.new_body, body))

group_df = pd.DataFrame(group_cols, columns=['group', 'label'], dtype=np.uint64)
group_df.to_csv(output_csv, header=True, index=False)
