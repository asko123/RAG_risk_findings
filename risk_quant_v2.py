import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load the CSV files into DataFrames
    likelihood_filepath = "likelihood.csv"
    impact_filepath = "impact.csv"
    
    df_likelihood = pd.read_csv(likelihood_filepath)
    df_impact = pd.read_csv(impact_filepath)
    
    # Data Cleaning
    df_likelihood = df_likelihood.fillna(0)  # Replace NaNs with 0, if any
    df_impact = df_impact.fillna(0)  # Replace NaNs with 0, if any
    df_likelihood.columns = df_likelihood.columns.str.strip()  # Strip whitespace from headers
    df_impact.columns = df_impact.columns.str.strip()  # Strip whitespace from headers

    # Merge DataFrames on Themes
    df_combined = pd.merge(df_likelihood, df_impact, on='Themes', suffixes=('_Likelihood', '_Impact'))
    
    # Summary Statistics for Likelihood and Impact
    likelihood_mean = df_combined['Likelihood'].mean()
    likelihood_median = df_combined['Likelihood'].median()
    likelihood_std = df_combined['Likelihood'].std()
    impact_mean = df_combined['Impact'].mean()
    impact_median = df_combined['Impact'].median()
    impact_std = df_combined['Impact'].std()
    
    print(f"Mean Likelihood: {likelihood_mean:.2f}")
    print(f"Median Likelihood: {likelihood_median:.2f}")
    print(f"Standard Deviation of Likelihood: {likelihood_std:.2f}")
    print(f"Mean Impact: {impact_mean:.2f}")
    print(f"Median Impact: {impact_median:.2f}")
    print(f"Standard Deviation of Impact: {impact_std:.2f}")

    # Correlation Analysis
    correlation_matrix = df_combined.corr()
    print("\nCorrelation Matrix:\n", correlation_matrix)
    
    # Visual Analysis - Bar Chart for Likelihood and Impact
    plt.figure(figsize=(10, 6))
    df_combined.plot(kind='bar', x='Themes', y=['Likelihood', 'Impact'], legend=True, color=['skyblue', 'salmon'])
    plt.xlabel("Themes")
    plt.ylabel("Values")
    plt.title("Likelihood and Impact per Theme")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Boxplot Analysis for Comparing Variables
    plt.figure(figsize=(12, 8))
    df_combined.iloc[:, 1:-2].boxplot()
    plt.title("Boxplot of Different Risk Themes Across Categories")
    plt.xlabel("Categories")
    plt.ylabel("Values")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Highlighting Themes with High Likelihood (> 3.0) and High Impact (> 5.0)
    high_risk_themes = df_combined[(df_combined['Likelihood'] > 3.0) & (df_combined['Impact'] > 5.0)]
    print("\nThemes with High Likelihood (> 3.0) and High Impact (> 5.0):\n", high_risk_themes[['Themes', 'Likelihood', 'Impact']])

if __name__ == "__main__":
    main()
