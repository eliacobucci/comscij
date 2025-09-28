#!/usr/bin/env python3
"""
Export priming data to CSV for statistical analysis
"""

import re
import pandas as pd

def extract_distances(file_path):
    """Extract semantic distance ratings from questionnaire file"""
    distances = {}

    with open(file_path, 'r') as f:
        content = f.read()

    # Regular expression to match distance ratings
    pattern = r'(\w+)\s+and\s+(\w+)\s+__(\d+)_'
    matches = re.findall(pattern, content)

    for term1, term2, distance in matches:
        key = f"{term1}-{term2}"
        distances[key] = int(distance)

    return distances

def get_target_pairs(distances):
    """Extract target animal-evaluative term pairs"""
    target_animals = ['PIG', 'HOG', 'BOAR', 'SWINE']
    evaluative_terms = ['BENEFICIAL', 'GOOD', 'ATTRACTIVE', 'BAD']

    target_pairs = {}

    for animal in target_animals:
        for eval_term in evaluative_terms:
            key = f"{eval_term}-{animal}"
            if key in distances:
                target_pairs[key] = distances[key]

    return target_pairs

def create_csv_export():
    """Create CSV export of all priming data"""

    # File paths
    files = {
        'Control': '/Users/josephwoelfel/pigssllm/controlllm.txt',
        'Pig_Prime': '/Users/josephwoelfel/pigssllm/pigsllm.txt',
        'Hog_Prime': '/Users/josephwoelfel/pigssllm/hogsllm.txt',
        'Boar_Prime': '/Users/josephwoelfel/pigssllm/boarllm.txt',
        'Swine_Prime': '/Users/josephwoelfel/pigssllm/swinellm.txt'
    }

    # Extract distances for all conditions
    all_distances = {}
    for condition, file_path in files.items():
        distances = extract_distances(file_path)
        target_pairs = get_target_pairs(distances)
        all_distances[condition] = target_pairs

    # Create CSV data
    csv_data = []

    target_animals = ['PIG', 'HOG', 'BOAR', 'SWINE']
    evaluative_terms = ['BENEFICIAL', 'GOOD', 'ATTRACTIVE', 'BAD']

    for eval_term in evaluative_terms:
        for animal in target_animals:
            pair = f"{eval_term}-{animal}"

            # Get control distance (baseline)
            control_dist = all_distances['Control'].get(pair, None)
            if control_dist is None:
                continue

            # Base row data
            row = {
                'Target_Animal': animal,
                'Evaluative_Term': eval_term,
                'Evaluative_Valence': 'Positive' if eval_term in ['BENEFICIAL', 'GOOD', 'ATTRACTIVE'] else 'Negative',
                'Control_Distance': control_dist
            }

            # Add primed distances and calculate effects
            for condition in ['Pig_Prime', 'Hog_Prime', 'Boar_Prime', 'Swine_Prime']:
                primed_dist = all_distances[condition].get(pair, None)
                if primed_dist is not None:
                    row[f'{condition}_Distance'] = primed_dist
                    row[f'{condition}_Effect'] = control_dist - primed_dist

                    # Determine if this is target-specific priming
                    prime_animal = condition.replace('_Prime', '').upper()
                    row[f'{condition}_Target_Specific'] = (prime_animal == animal)

            csv_data.append(row)

    # Create DataFrame and save
    df = pd.DataFrame(csv_data)

    # Reorder columns for clarity
    col_order = ['Target_Animal', 'Evaluative_Term', 'Evaluative_Valence', 'Control_Distance']
    for condition in ['Pig_Prime', 'Hog_Prime', 'Boar_Prime', 'Swine_Prime']:
        col_order.extend([f'{condition}_Distance', f'{condition}_Effect', f'{condition}_Target_Specific'])

    df = df[col_order]

    # Save CSV
    df.to_csv('/Users/josephwoelfel/asa/priming_effects_data.csv', index=False)
    print(f"CSV exported with {len(df)} rows and {len(df.columns)} columns")
    print("\nColumn names:")
    for col in df.columns:
        print(f"  {col}")

    # Display summary
    print(f"\nData Summary:")
    print(f"Target Animals: {df['Target_Animal'].unique().tolist()}")
    print(f"Evaluative Terms: {df['Evaluative_Term'].unique().tolist()}")
    print(f"Distance Range: {df['Control_Distance'].min()}-{df['Control_Distance'].max()} units")

if __name__ == "__main__":
    create_csv_export()