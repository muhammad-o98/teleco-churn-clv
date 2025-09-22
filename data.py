# data.py
import os
import shutil
import kagglehub
import pandas as pd

def init_data(target_dir=None):
    """
    Downloads the 'blastchar/telco-customer-churn' dataset using kagglehub,
    saves it to the given target_dir (default: ~/Projects/teleco/data),
    and returns the path + combined dataframe of all CSVs.
    """

    # Dataset Codes from Kaggele:
    #blastchar/telco-customer-churn
    
    # Step 1: Download dataset
    path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    print("Downloaded to:", path)

    # Step 2: If no target_dir provided, use default
    #"~/Projects/teleco/data"
    if target_dir is None:
        target_dir = os.path.expanduser("~/Projects/teleco/data")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Step 3: Copy files into target_dir
    for file in os.listdir(path):
        src = os.path.join(path, file)
        dst = os.path.join(target_dir, file)
        shutil.copy(src, dst)

    print("Dataset copied to:", target_dir)

    # Step 4: Try loading all CSVs if available and combine them
    all_dfs = []
    for file in os.listdir(target_dir):
        if file.endswith(".csv"):
            csv_path = os.path.join(target_dir, file)
            temp_df = pd.read_csv(csv_path)
            print(f"Loaded {file}, shape={temp_df.shape}")
            all_dfs.append(temp_df)

    # Combine all CSVs into one main dataframe = "df"
    #df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    #print("Combined DataFrame shape:", df.shape)

    # Step 5: Save combined dataframe as CSV in the same folder
    #combined_csv_path = os.path.join(target_dir, "main_df.csv")
    #df.to_csv(combined_csv_path, index=False)
    #print(f"Combined dataframe saved to: {combined_csv_path}")

    return target_dir, pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

#"/Users/ob/Projects/teleco/data/"
if __name__ == "__main__":
    # Example: use default
    data_path, df = init_data()

    print("Init complete. Data saved at:", data_path)
    if not df.empty:
        print("First 5 rows:\n", df.head())
        print(df.info())
