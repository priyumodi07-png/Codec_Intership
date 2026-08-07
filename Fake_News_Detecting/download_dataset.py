import os
import pandas as pd
import requests

def download_dataset():
    print("Downloading a public fake news dataset (6,335 articles)...")
    
    url = "https://raw.githubusercontent.com/docketrun/Detecting-Fake-News-with-Scikit-Learn/master/fake_or_real_news.csv"
    
    dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    file_path = os.path.join(dataset_dir, 'fake_or_real_news.csv')
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            f.write(response.content)
            
        print(f"Dataset successfully downloaded to {file_path}")
        
        # Validate and clean the dataset
        df = pd.read_csv(file_path)
        print(f"Dataset loaded. Shape: {df.shape}")
        
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
            
        if 'label' in df.columns:
            df['label'] = df['label'].str.upper()
            
        df.to_csv(file_path, index=False)
        print("Dataset formatted and ready for training!")
        
    except Exception as e:
        print(f"An error occurred while downloading the dataset: {e}")

if __name__ == "__main__":
    download_dataset()
