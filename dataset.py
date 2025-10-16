import os
import zipfile
import gdown


def download_data():
    url = 'https://drive.google.com/uc?id=1xr8tsSw-a2hUYcWYAK129BKLNcxcwjWD'
            
    gdown.download(url, 'dataset.zip', quiet=True)
    
    with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
        zip_ref.extractall('data/')
        
    os.remove('dataset.zip')
        

if __name__ == "__main__":
    download_data()