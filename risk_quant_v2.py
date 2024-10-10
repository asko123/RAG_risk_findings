import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Data Collection and Preparation
def collect_and_prepare_data():
    # Read the CSV files
    impact_df = pd.read_csv('impact.csv', sep='\t')
    likelihood_df = pd.read_csv('likelihood.csv', sep='\t')

    # Merge the dataframes on the 'Themes' column
    merged_df = pd.merge(
        impact_df[['Themes', 'Impact']],
        likelihood_df[['Themes', 'Likelihood']],
        on='Themes'
    )

    # Drop any rows with missing values (if any)
    merged_df.dropna(inplace=True)

    # Print the merged dataframe
    print("\nMerged Dataframe:")
    print(merged_df)

    # Extract impact and likelihood values
    impact = merged_df['Impact'].values
    likelihood = merged_df['Likelihood'].values

    # Use KDE to sample new impact values
    impact_kde = gaussian_kde(impact)
    impact_samples = impact_kde.resample(size=1000).flatten()

    # Use KDE to sample new likelihood values
    likelihood_kde = gaussian_kde(likelihood)
    likelihood_samples = likelihood_kde.resample(size=1000).flatten()

    # Combine into a data array
    data = np.column_stack((impact_samples, likelihood_samples))

    # Create a dataframe from the sampled data
    sampled_df = pd.DataFrame(data, columns=['Impact', 'Likelihood'])

    # Print the sampled dataframe
    print("\nSampled Dataframe (First 5 Rows):")
    print(sampled_df.head())

    return data

# Monte Carlo Simulation
def monte_carlo_simulation(data, num_simulations=10000):
    # Randomly sample from the data to create scenarios
    idx = np.random.randint(0, data.shape[0], num_simulations)
    scenarios = data[idx]

    # Ensure generated values are non-negative
    scenarios = np.clip(scenarios, 0, None)

    # Create a dataframe for the scenarios
    risk_scores_df = pd.DataFrame(scenarios, columns=['Impact', 'Likelihood'])

    # Calculate risk scores
    risk_scores_df['Risk Score'] = risk_scores_df['Impact'] * risk_scores_df['Likelihood']

    # Print the risk scores dataframe
    print("\nRisk Scores Dataframe (First 5 Rows):")
    print(risk_scores_df.head())

    return risk_scores_df

# Calculate Risk Scores per Theme
def calculate_risk_scores_per_theme():
    # Read and merge the original data
    impact_df = pd.read_csv('impact.csv', sep='\t')
    likelihood_df = pd.read_csv('likelihood.csv', sep='\t')

    merged_df = pd.merge(
        impact_df[['Themes', 'Impact']],
        likelihood_df[['Themes', 'Likelihood']],
        on='Themes'
    )

    # Drop any rows with missing values (if any)
    merged_df.dropna(inplace=True)

    # Calculate risk scores
    merged_df['Risk Score'] = merged_df['Impact'] * merged_df['Likelihood']

    # Print the dataframe
    print("\nRisk Scores per Theme:")
    print(merged_df)

    return merged_df

# Visualization
def visualize_risk_distribution(risk_scores_df):
    # Extract the 'Risk Score' column
    risk_scores = risk_scores_df['Risk Score']

    kde = gaussian_kde(risk_scores)
    x_range = np.linspace(risk_scores.min(), risk_scores.max(), 100)
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, kde(x_range))
    plt.title('Risk Score Distribution (Synthetic Data)')
    plt.xlabel('Risk Score')
    plt.ylabel('Density')
    plt.show()

    # Print the risk scores dataframe (First 5 Rows)
    print("\nRisk Scores Dataframe (First 5 Rows):")
    print(risk_scores_df.head())

# Main execution
def main():
    # Calculate risk scores per theme using original data
    theme_risk_scores_df = calculate_risk_scores_per_theme()

    # Visualize risk scores per theme
    plt.figure(figsize=(10, 6))
    plt.hist(theme_risk_scores_df['Risk Score'], bins=10, edgecolor='k', alpha=0.7)
    plt.title('Risk Score Distribution per Theme')
    plt.xlabel('Risk Score')
    plt.ylabel('Frequency')
    plt.show()

    # Data preparation with KDE sampling
    data = collect_and_prepare_data()

    # Monte Carlo simulation
    risk_scores_df = monte_carlo_simulation(data)

    # Visualization
    visualize_risk_distribution(risk_scores_df)

    # Basic statistics
    risk_scores = risk_scores_df['Risk Score']
    print(f"\nMean Risk Score (Synthetic Data): {risk_scores.mean():.2f}")
    print(f"Median Risk Score (Synthetic Data): {risk_scores.median():.2f}")
    print(f"95th Percentile Risk Score (Synthetic Data): {risk_scores.quantile(0.95):.2f}")

if __name__ == "__main__":
    main()
