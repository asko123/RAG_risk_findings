import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to load and merge data
def load_and_merge_data():
    # Load the CSV files into DataFrames
    likelihood_filepath = "likelihood.csv"
    impact_filepath = "impact.csv"
    
    df_likelihood = pd.read_csv(likelihood_filepath, sep='\t')
    df_impact = pd.read_csv(impact_filepath, sep='\t')
    
    # Data Cleaning
    df_likelihood = df_likelihood.fillna(0)  # Replace NaNs with 0, if any
    df_impact = df_impact.fillna(0)  # Replace NaNs with 0, if any
    df_likelihood.columns = df_likelihood.columns.str.strip()  # Strip whitespace from headers
    df_impact.columns = df_impact.columns.str.strip()  # Strip whitespace from headers

    # Merge DataFrames on Themes
    df_combined = pd.merge(df_likelihood, df_impact, on='Themes', suffixes=('_Likelihood', '_Impact'))
    
    return df_combined

# Function to encode themes using TF-IDF Vectorizer
def encode_themes(df_combined):
    themes = df_combined['Themes'].values

    # Use TF-IDF Vectorizer to encode the themes
    vectorizer = TfidfVectorizer()
    theme_encoded = vectorizer.fit_transform(themes).toarray()

    return theme_encoded, vectorizer

# Function to build the generator model
def build_generator(theme_dim, noise_dim=10):
    noise_input = layers.Input(shape=(noise_dim,))
    theme_input = layers.Input(shape=(theme_dim,))

    x = layers.Concatenate()([noise_input, theme_input])

    x = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(2, activation='linear')(x)  # Output Impact and Likelihood

    model = models.Model([noise_input, theme_input], x)
    return model

# Function to build the discriminator model
def build_discriminator(theme_dim):
    data_input = layers.Input(shape=(2,))
    theme_input = layers.Input(shape=(theme_dim,))

    x = layers.Concatenate()([data_input, theme_input])

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(1, activation='sigmoid')(x)  # Output single probability

    model = models.Model([data_input, theme_input], x)
    return model

# Function to train the Conditional GAN
def train_conditional_gan(df_combined, theme_encoded, encoder, epochs=5000, batch_size=32):
    theme_dim = theme_encoded.shape[1]
    noise_dim = 10

    # Scale real data to [0, 1]
    scaler = MinMaxScaler()
    scaled_real_data = scaler.fit_transform(df_combined[['Impact', 'Likelihood']])

    # Build and compile discriminator
    discriminator = build_discriminator(theme_dim)
    discriminator.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.Adam(learning_rate=0.0001),
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
        optimizer=optimizers.Adam(learning_rate=0.0001)
    )

    # Training Loop
    num_samples = scaled_real_data.shape[0]

    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, num_samples, batch_size)
        real_samples = scaled_real_data[idx]
        real_themes = theme_encoded[idx]

        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        generated_samples = generator.predict([noise, real_themes], verbose=0)

        real_labels = np.ones((batch_size, 1)) * 0.9  # Smooth real labels
        fake_labels = np.zeros((batch_size, 1)) + 0.1  # Smooth fake labels

        d_loss_real = discriminator.train_on_batch([real_samples, real_themes], real_labels)
        d_loss_fake = discriminator.train_on_batch([generated_samples, real_themes], fake_labels)

        d_loss_avg = 0.5 * (d_loss_real[0] + d_loss_fake[0])
        d_acc_avg = 0.5 * (d_loss_real[1] + d_loss_fake[1])

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        sampled_theme_indices = np.random.randint(0, theme_dim, batch_size)
        sampled_themes = np.zeros((batch_size, theme_dim))
        sampled_themes[np.arange(batch_size), sampled_theme_indices] = 1

        g_loss = combined_model.train_on_batch([noise, sampled_themes], np.ones((batch_size, 1)))

        # Assuming d_loss_avg, d_acc_avg, and g_loss could be lists, use last value for print
        d_loss_avg_value = d_loss_avg[-1] if isinstance(d_loss_avg, list) else d_loss_avg
        d_acc_avg_value = d_acc_avg[-1] if isinstance(d_acc_avg, list) else d_acc_avg
        g_loss_value = g_loss[-1] if isinstance(g_loss, list) else g_loss


        if epoch % 500 == 0 or epoch == epochs - 1:
          print(
              f"Epoch {epoch}/{epochs} | D loss: {d_loss_avg_value:.4f}, "
              f"acc: {d_acc_avg_value * 100:.2f}% | G loss: {g_loss_value:.4f}"
              )

    return generator

# Function to generate synthetic data using the trained generator
def generate_synthetic_data(generator, encoder, num_samples):
    themes = encoder.get_feature_names_out()
    theme_encoded = encoder.transform(themes).toarray()
    noise_dim = 10

    synthetic_data = []

    for theme_idx, theme in enumerate(themes):
        noise = np.random.normal(0, 1, (num_samples, noise_dim))
        theme_condition = theme_encoded[theme_idx].reshape(1, -1)
        theme_conditions = np.repeat(theme_condition, num_samples, axis=0)

        generated_samples = generator.predict([noise, theme_conditions], verbose=0)

        theme_df = pd.DataFrame(generated_samples, columns=['Impact', 'Likelihood'])
        theme_df['Themes'] = theme

        synthetic_data.append(theme_df)

    synthetic_df = pd.concat(synthetic_data, ignore_index=True)
    synthetic_df['Impact'] = synthetic_df['Impact'].clip(lower=0)
    synthetic_df['Likelihood'] = synthetic_df['Likelihood'].clip(lower=0)

    print("\nSynthetic Dataframe (First 5 Rows):")
    print(synthetic_df[['Themes', 'Impact', 'Likelihood']].head())

    return synthetic_df

# Function to calculate risk scores and rank themes
def calculate_and_rank_risks(synthetic_df):
    synthetic_df['Risk Score'] = synthetic_df['Impact'] * synthetic_df['Likelihood']
    theme_risk_scores = synthetic_df.groupby('Themes')['Risk Score'].mean().reset_index()
    theme_risk_scores['Rank'] = theme_risk_scores['Risk Score'].rank(ascending=False)
    ranked_themes = theme_risk_scores.sort_values('Rank')

    print("\nRanked Themes based on Average Risk Score:")
    print(ranked_themes)

    return ranked_themes

# Function to visualize risk scores
def visualize_risk_scores(synthetic_df):
    plt.figure(figsize=(10, 6))
    plt.hist(synthetic_df['Risk Score'], bins=30, edgecolor='k', alpha=0.7)
    plt.title('Risk Score Distribution (Synthetic Data with Themes)')
    plt.xlabel('Risk Score')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(12, 6))
    synthetic_df.boxplot(column='Risk Score', by='Themes', rot=90)
    plt.title('Risk Score Distribution per Theme')
    plt.xlabel('Themes')
    plt.ylabel('Risk Score')
    plt.tight_layout()
    plt.show()

# Main execution function
def main():
    df_combined = load_and_merge_data()
    theme_encoded, encoder = encode_themes(df_combined)
    generator = train_conditional_gan(df_combined, theme_encoded, encoder, epochs=5000, batch_size=32)
    synthetic_df = generate_synthetic_data(generator, encoder, num_samples=100)
    ranked_themes = calculate_and_rank_risks(synthetic_df)
    visualize_risk_scores(synthetic_df)

    print(f"\nMean Risk Score (Synthetic Data): {synthetic_df['Risk Score'].mean():.2f}")
    print(f"Median Risk Score (Synthetic Data): {synthetic_df['Risk Score'].median():.2f}")
    print(f"95th Percentile Risk Score (Synthetic Data): {synthetic_df['Risk Score'].quantile(0.95):.2f}")

if __name__ == "__main__":
    main()
