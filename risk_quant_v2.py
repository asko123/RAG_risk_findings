import numpy as np
import pandas as pd  # Added for data handling
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
    return data

# Monte Carlo Simulation
def monte_carlo_simulation(data, num_simulations=10000):
    # Randomly sample from the data to create scenarios
    idx = np.random.randint(0, data.shape[0], num_simulations)
    scenarios = data[idx]

    # Ensure generated values are non-negative
    scenarios = np.clip(scenarios, 0, None)

    risk_scores = np.array([
        calculate_risk_score(impact, likelihood)
        for impact, likelihood in scenarios
    ])
    return risk_scores

# Risk Scoring Framework
def calculate_risk_score(impact, likelihood):
    return ((impact * likelihood)/36)

# Visualization
def visualize_risk_distribution(risk_scores):
    kde = gaussian_kde(risk_scores)
    x_range = np.linspace(min(risk_scores), max(risk_scores), 100)
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, kde(x_range))
    plt.title('Risk Score Distribution')
    plt.xlabel('Risk Score')
    plt.ylabel('Density')
    plt.show()

# Main execution
def main():
    # Data preparation with KDE sampling
    data = collect_and_prepare_data()

    # Monte Carlo simulation
    risk_scores = monte_carlo_simulation(data)

    # Visualization
    visualize_risk_distribution(risk_scores)

    # Basic statistics
    print(f"Mean Risk Score: {np.mean(risk_scores):.2f}")
    print(f"Median Risk Score: {np.median(risk_scores):.2f}")
    print(f"95th Percentile Risk Score: {np.percentile(risk_scores, 95):.2f}")

if __name__ == "__main__":
    main()
