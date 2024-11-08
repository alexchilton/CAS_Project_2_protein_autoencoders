import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class Utils:
    @staticmethod
    def convert_columns_to_float(autoencoder_input_dfs):
        for df in autoencoder_input_dfs:
            for col in df.columns:
                if col != 'AA':
                    df[col] = df[col].astype(float)
        return autoencoder_input_dfs

    @staticmethod
    def normalize_dataframes(autoencoder_input_dfs):
        normalized_autoencoder_input_dfs = []
        for df in autoencoder_input_dfs:
            numeric_columns = df.select_dtypes(include=['number']).columns
            for col in numeric_columns:
                scaler = StandardScaler()
                df[col] = scaler.fit_transform(df[[col]])
            normalized_autoencoder_input_dfs.append(df)
        return normalized_autoencoder_input_dfs

    @staticmethod
    def create_onehot_encoder(autoencoder_input_dfs, column_name='AA'):
        # Combine all unique values from the specified column in autoencoder_input_dfs
        all_values = np.unique(np.concatenate([df[column_name].values for df in autoencoder_input_dfs]))

        # Initialize and fit the OneHotEncoder with feature names
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        onehot_encoder.fit(pd.DataFrame(all_values, columns=[column_name]))

        return onehot_encoder

    @staticmethod
    def encode_values(df,onehot_encoder, column_name='AA'):
        encoded = onehot_encoder.transform(df[[column_name]])
        encoded_df = pd.DataFrame(encoded, columns=onehot_encoder.get_feature_names_out([column_name]))
        return pd.concat([df.drop(columns=[column_name]), encoded_df], axis=1)

# Function to decode values
    def decode_values(encoded_df,onehot_encoder, original_column_name='AA'):
        encoded_columns = onehot_encoder.get_feature_names_out([original_column_name])
        print(f"Encoded columns: {encoded_columns}")
        print(f"Columns in encoded_df: {encoded_df.columns}")
        encoded_values = encoded_df[encoded_columns].values
        decoded_values = onehot_encoder.inverse_transform(encoded_values).flatten()
        decoded_df = encoded_df.drop(columns=encoded_columns)
        decoded_df[original_column_name] = decoded_values
        return decoded_df