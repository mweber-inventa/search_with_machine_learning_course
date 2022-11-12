import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
import re

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/fasttext/labeled_queries.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1000,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])

categories.append(root_category_id)
parents.append(root_category_id)
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
queries_df = pd.read_csv(queries_file_name)[['category', 'query']]
queries_df = queries_df[queries_df['category'].isin(categories)]

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
def normalize_query(query):
    return stemmer.stem(re.sub(r'\W+', ' ', query).lower().strip())

queries_df['normalized_query'] = queries_df['query'].apply(normalize_query)

# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.

query_counts = queries_df.groupby(['category']).agg(count=('normalized_query', 'count')).reset_index()
query_counts['is_over_threshold'] = query_counts['count'].apply(lambda x: True if x >= min_queries else False)
print(f"{(~query_counts['is_over_threshold']).sum()} categories under threshold")

max_loops = 10
safe_loop_counter = 0
while ((~query_counts['is_over_threshold']).sum() > 0):
    
    # merge with parents
    query_counts = query_counts.merge(parents_df, how = 'left', on = 'category')
    queries_df = queries_df.merge(query_counts, how = 'left', on = 'category')
    
    queries_df['category'] = queries_df.apply(lambda x: str(x['category']) if x['is_over_threshold'] else str(x['parent']), axis=1)
    queries_df.drop(columns=['parent', 'is_over_threshold', 'count'], inplace = True)

    query_counts = queries_df.groupby(['category']).agg(count=('normalized_query', 'count')).reset_index()
    query_counts['is_over_threshold'] = query_counts['count'].apply(lambda x: True if x >= min_queries else False)

    safe_loop_counter = safe_loop_counter + 1

    print(f"Iteration {safe_loop_counter} completed")
    print(f"{(~query_counts['is_over_threshold']).sum()} categories under threshold - {(query_counts['is_over_threshold']).sum()} total categories")

    if safe_loop_counter >= max_loops:
        break

print(f"NA category: {queries_df['category'].isna().sum()}")
print(f"NA normalized_query: {queries_df['normalized_query'].isna().sum()}")
print(f"NA query: {queries_df['query'].isna().sum()}")

print(f"NULL category: {queries_df['category'].isnull().sum()}")
print(f"NULL normalized_query: {queries_df['normalized_query'].isnull().sum()}")
print(f"NULL query: {queries_df['query'].isnull().sum()}")

# Create labels in fastText format.
queries_df['label'] = '__label__' + queries_df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
queries_df = queries_df[queries_df['category'].isin(categories)]
queries_df['output'] = queries_df['label'] + ' ' + queries_df['normalized_query']
queries_df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
