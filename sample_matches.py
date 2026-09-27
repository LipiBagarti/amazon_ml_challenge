import polars as pl
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)

s1 = pl.read_csv('dataset/train/train_source1.tsv', separator='\t')
s2 = pl.read_csv('dataset/train/train_source2.tsv', separator='\t')
s3 = pl.read_csv('dataset/train/train_source3.tsv', separator='\t')
targets = pl.concat([s2, s3])

gt = pl.read_csv('dataset/train/train_ground_truth.tsv', separator='\t')
gt = gt.filter(pl.col('matched_entity_ids').is_not_null() & (pl.col('matched_entity_ids') != ''))

sample_s1 = gt.sample(n=10, seed=42)
sample_pairs = sample_s1.select(
    pl.col('source1_entity_id'),
    pl.col('matched_entity_ids').str.split(',').alias('target_id')
).explode('target_id')

joined = sample_pairs.join(s1, left_on='source1_entity_id', right_on='entity_id')
joined = joined.join(targets, left_on='target_id', right_on='entity_id', suffix='_target')

df = joined.to_pandas()
for i, row in df.head(15).iterrows():
    print(f"=== MATCH {i+1} ===")
    print(f"S1 [{row['source1_entity_id']}] ({row['country']}):")
    print(f"  Name:    {row['business_name']}")
    print(f"  Address: {row['business_address']}")
    print(f"Target [{row['target_id']}] ({row['country_target']}):")
    print(f"  Name:    {row['business_name_target']}")
    print(f"  Address: {row['business_address_target']}")
    print()
