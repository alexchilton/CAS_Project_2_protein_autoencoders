import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class Utils:

    # This is all the columns we have in the graph so i have made it a variable cos
    # we will be using it in multiple places
    columns = ['X', 'Y', 'Z', 'Avg_Mass', 'Avg_Neighbor_Dist',
               'Max_Neighbor_Dist', 'Neighbor_Count', 'AA_A', 'AA_C',
               'AA_D', 'AA_E', 'AA_F', 'AA_G', 'AA_H', 'AA_I', 'AA_K',
               'AA_L', 'AA_M', 'AA_N', 'AA_P', 'AA_Q', 'AA_R', 'AA_S',
               'AA_T', 'AA_V', 'AA_W', 'AA_Y']

    @staticmethod
    def get_columns():
        """
        Returns the list of column names used in the graph.

        Returns:
            list: A list of column names.
        """
        return Utils.columns

    @staticmethod
    def get_number_of_features():
        """
        Returns the number of columns (features) used in the graph.

        Returns:
            int: The number of columns.
        """
        return len(Utils.columns)

    @staticmethod
    def convert_columns_to_float(autoencoder_input_dfs):
        ''' This function converts all columns in the input dataframes to float except the 'AA' column'''
        for df in autoencoder_input_dfs:
            for col in df.columns:
                if col != 'AA':
                    df[col] = df[col].astype(float)
        return autoencoder_input_dfs

    @staticmethod
    def normalize_dataframes(autoencoder_input_dfs):
        ''' This function normalizes the numeric columns in the input dataframes
        NOT USED AS IF IT IS NORMALIZED THEN THE GRAPH EDGING DOESN'T WORK!!!'''
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
        ''' This function creates a OneHotEncoder object for the specified column'''
        # Combine all unique values from the specified column in autoencoder_input_dfs
        all_values = np.unique(np.concatenate([df[column_name].values for df in autoencoder_input_dfs]))

        # Initialize and fit the OneHotEncoder with feature names
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        onehot_encoder.fit(pd.DataFrame(all_values, columns=[column_name]))

        return onehot_encoder

    @staticmethod
    def encode_values(df,onehot_encoder, column_name='AA'):
        ''' This function encodes the specified column in the input dataframe'''
        encoded = onehot_encoder.transform(df[[column_name]])
        encoded_df = pd.DataFrame(encoded, columns=onehot_encoder.get_feature_names_out([column_name]))
        return pd.concat([df.drop(columns=[column_name]), encoded_df], axis=1)

# Function to decode values
    def decode_values(encoded_df,onehot_encoder, original_column_name='AA'):
        ''' This function decodes the specified column in the input dataframe'''
        encoded_columns = onehot_encoder.get_feature_names_out([original_column_name])
        print(f"Encoded columns: {encoded_columns}")
        print(f"Columns in encoded_df: {encoded_df.columns}")
        encoded_values = encoded_df[encoded_columns].values
        decoded_values = onehot_encoder.inverse_transform(encoded_values).flatten()
        decoded_df = encoded_df.drop(columns=encoded_columns)
        decoded_df[original_column_name] = decoded_values
        return decoded_df
