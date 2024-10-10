import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# Function to load and merge data
def load_and_merge_data():
    # Load data
    impact_df = pd.read_csv('impact.csv', sep='\t')
    likelihood_df = pd.read_csv('likelihood.csv', sep='\t')

    # Merge on 'Themes'
    merged_df = pd.merge(
        impact_df[['Themes', 'Impact']],
        likelihood_df[['Themes', 'Likelihood']],
        on='Themes'
    )

    # Drop missing values
    merged_df.dropna(inplace=True)

    # Print merged dataframe
    print("\nMerged Dataframe:")
    print(merged_df.head())

    return merged_df

# Function to encode themes using OneHotEncoder
def encode_themes(merged_df):
    themes = merged_df['Themes'].values.reshape(-1, 1)

    # One-hot encode the themes
    encoder = OneHotEncoder(sparse_output=False)
    theme_encoded = encoder.fit_transform(themes)

    theme_labels = encoder.categories_[0]

    # Print theme encoding
    print("\nTheme Encoding:")
    for idx, theme in enumerate(theme_labels):
        print(f"{idx}: {theme}")

    return theme_encoded, encoder

# Function to build the generator model
def build_generator(theme_dim, noise_dim=10):
    noise_input = layers.Input(shape=(noise_dim,))
    theme_input = layers.Input(shape=(theme_dim,))

    x = layers.Concatenate()([noise_input, theme_input])

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(2, activation='linear')(x)  # Output Impact and Likelihood

    model = models.Model([noise_input, theme_input], x)
    return model

# Function to build the discriminator model
def build_discriminator(theme_dim):
    data_input = layers.Input(shape=(2,))
    theme_input = layers.Input(shape=(theme_dim,))

    x = layers.Concatenate()([data_input, theme_input])

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(1, activation='sigmoid')(x)  # Output single probability

    model = models.Model([data_input, theme_input], x)
    return model

# Function to train the Conditional GAN
def train_conditional_gan(merged_df, theme_encoded, encoder, epochs=5000, batch_size=32):
    theme_dim = theme_encoded.shape[1]
    noise_dim = 10

    # Build and compile discriminator
    discriminator = build_discriminator(theme_dim)
    discriminator.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.Adam(0.0002),
        metrics=['accuracy']
    )

    # Build generator
    generator = build_generator(theme_dim, noise_dim=noise_dim)

    # Combined GAN model
    discriminator.trainable = False
    noise_input = layers.Input(shape=(noise_dim,))
    theme_input = layers.Input(shape=(theme_dim,))
    generated_data = generator([noise_input, theme_input])
    validity = discriminator([generated_data, theme_input])
    combined_model = models.Model([noise_input, theme_input], validity)
    combined_model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.Adam(0.0002)
    )

    # Training Loop
    real_data = merged_df[['Impact', 'Likelihood']].values
    num_samples = real_data.shape[0]

    for epoch in range(epochs):
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of real data
        idx = np.random.randint(0, num_samples, batch_size)
        real_samples = real_data[idx]
        real_themes = theme_encoded[idx]

        # Generate a batch of fake data
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        generated_samples = generator.predict([noise, real_themes], verbose=0)

        # Labels for real and fake data
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        # Train the discriminator on real data
        d_loss_real = discriminator.train_on_batch(
            [real_samples, real_themes], real_labels
        )
        # Train the discriminator on fake data
        d_loss_fake = discriminator.train_on_batch(
            [generated_samples, real_themes], fake_labels
        )

        # Compute average loss and accuracy
        d_loss_real_loss, d_loss_real_acc = d_loss_real
        d_loss_fake_loss, d_loss_fake_acc = d_loss_fake
        d_loss_avg = 0.5 * (d_loss_real_loss + d_loss_fake_loss)
        d_acc_avg = 0.5 * (d_loss_real_acc + d_loss_fake_acc)

        # ---------------------
        #  Train Generator
        # ---------------------

        # Generate new noise
        noise = np.random.normal(0, 1, (batch_size, noise_dim))

        # Random theme conditions
        sampled_theme_indices = np.random.randint(0, theme_dim, batch_size)
        sampled_themes = np.zeros((batch_size, theme_dim))
        sampled_themes[np.arange(batch_size), sampled_theme_indices] = 1

        # Train the generator
        g_loss = combined_model.train_on_batch(
            [noise, sampled_themes], np.ones((batch_size, 1))
        )

        # Print the progress
        if epoch % 500 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch}/{epochs} | D loss: {d_loss_avg:.4f}, "
                f"acc: {100 * d_acc_avg:.2f}% | G loss: {g_loss:.4f}"
            )

    return generator

# Function to generate synthetic data using the trained generator
def generate_synthetic_data(generator, encoder, num_samples):
    themes = encoder.categories_[0]
    theme_encoded = encoder.transform(themes.reshape(-1, 1))
    noise_dim = 10

    synthetic_data = []

    for theme_idx, theme in enumerate(themes):
        # Generate noise
        noise = np.random.normal(0, 1, (num_samples, noise_dim))

        # Repeat the theme encoding
        theme_condition = theme_encoded[theme_idx].reshape(1, -1)
        theme_conditions = np.repeat(theme_condition, num_samples, axis=0)

        # Generate synthetic data
        generated_samples = generator.predict([noise, theme_conditions], verbose=0)

        # Create DataFrame
        theme_df = pd.DataFrame(generated_samples, columns=['Impact', 'Likelihood'])
        theme_df['Themes'] = theme

        synthetic_data.append(theme_df)

    synthetic_df = pd.concat(synthetic_data, ignore_index=True)

    # Ensure non-negative values
    synthetic_df['Impact'] = synthetic_df['Impact'].clip(lower=0)
    synthetic_df['Likelihood'] = synthetic_df['Likelihood'].clip(lower=0)

    # Print synthetic dataframe
    print("\nSynthetic Dataframe (First 5 Rows):")
    print(synthetic_df[['Themes', 'Impact', 'Likelihood']].head())

    return synthetic_df

# Function to calculate risk scores and rank themes
def calculate_and_rank_risks(synthetic_df):
    # Calculate risk scores
    synthetic_df['Risk Score'] = synthetic_df['Impact'] * synthetic_df['Likelihood']

    # Aggregate risk scores per Theme
    theme_risk_scores = synthetic_df.groupby('Themes')['Risk Score'].mean().reset_index()

    # Rank Themes based on risk score
    theme_risk_scores['Rank'] = theme_risk_scores['Risk Score'].rank(ascending=False)

    # Sort Themes by rank
    ranked_themes = theme_risk_scores.sort_values('Rank')

    # Print ranked Themes
    print("\nRanked Themes based on Average Risk Score:")
    print(ranked_themes)

    return ranked_themes

# Function to visualize risk scores
def visualize_risk_scores(synthetic_df):
    # Histogram of Risk Scores
    plt.figure(figsize=(10, 6))
    plt.hist(synthetic_df['Risk Score'], bins=30, edgecolor='k', alpha=0.7)
    plt.title('Risk Score Distribution (Synthetic Data with Themes)')
    plt.xlabel('Risk Score')
    plt.ylabel('Frequency')
    plt.show()

    # Boxplot per Theme
    plt.figure(figsize=(12, 6))
    synthetic_df.boxplot(column='Risk Score', by='Themes', rot=90)
    plt.title('Risk Score Distribution per Theme')
    plt.xlabel('Themes')
    plt.ylabel('Risk Score')
    plt.tight_layout()
    plt.show()

# Main execution function
def main():
    # Load and merge data
    merged_df = load_and_merge_data()

    # Encode Themes
    theme_encoded, encoder = encode_themes(merged_df)

    # Train Conditional GAN
    generator = train_conditional_gan(
        merged_df, theme_encoded, encoder, epochs=5000, batch_size=32
    )

    # Generate Synthetic Data
    synthetic_df = generate_synthetic_data(generator, encoder, num_samples=100)

    # Calculate and Rank Risk Scores
    ranked_themes = calculate_and_rank_risks(synthetic_df)

    # Visualize Risk Scores
    visualize_risk_scores(synthetic_df)

    # Print Basic Statistics
    print(f"\nMean Risk Score (Synthetic Data): {synthetic_df['Risk Score'].mean():.2f}")
    print(f"Median Risk Score (Synthetic Data): {synthetic_df['Risk Score'].median():.2f}")
    print(
        f"95th Percentile Risk Score (Synthetic Data): "
        f"{synthetic_df['Risk Score'].quantile(0.95):.2f}"
    )

if __name__ == "__main__":
    main()
