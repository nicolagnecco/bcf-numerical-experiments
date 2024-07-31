# %%
import argparse

import numpy as np
import pandas as pd


# %%
# Function to load .npz file, convert to DataFrame, and save as CSV
def main(npz_file_path, csv_file_path):
    # %% Load the .npz file
    npz_data = np.load(npz_file_path, allow_pickle=True)

    # %%Convert to a dictionary of arrays
    data_dict = {key: npz_data[key] for key in npz_data}

    # %% Convert dictionary to Pandas DataFrame
    df = pd.DataFrame(data_dict)

    # Save DataFrame to CSV
    df.to_csv(csv_file_path, index=False)
    print(f"Data saved to {csv_file_path}")


# %%
npz_file_path = "../results/try-extended/n_preds_27-n_conf_0-n_trainenv_27/20240731-143203/causalbench-res.npz"
csv_file_path = "../results/try-extended/n_preds_27-n_conf_0-n_trainenv_27/20240731-143203/causalbench-res.csv"

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .npz file to .csv")
    parser.add_argument(
        "--npz_file", type=str, required=True, help="Path to input .npz file"
    )
    parser.add_argument(
        "--csv_file", type=str, required=True, help="Path to output .csv file"
    )
    args = parser.parse_args()

    npz_file_path = args.npz_file
    csv_file_path = args.csv_file

    main(npz_file_path, csv_file_path)
