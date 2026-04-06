import gdown
import os

# FFA-Net pre-trained weights download URLs from Google Drive
# These are direct links to the model files

# ITS model - for indoor hazy scenes
its_model_id = "19_lSUPrpLDZl9AyewhHBsHidZEpTMIV5"  # Folder ID
output_dir = "models/FFA-Net/net/trained_models"
os.makedirs(output_dir, exist_ok=True)

print("Downloading FFA-Net pre-trained weights...")

# Try to download from Google Drive folder
try:
    # Download the entire folder
    gdown.download_folder(
        url=f"https://drive.google.com/drive/folders/{its_model_id}",
        output=output_dir,
        quiet=False,
        use_cookies=False
    )
    print("✅ Weights downloaded successfully!")
except Exception as e:
    print(f"⚠️ Direct download failed: {e}")
    print("Trying alternative approach...")
    
    # Alternative: Using a direct model link if available
    # You can add specific model file IDs here
    model_urls = {
        "its_train_ffa_3_19.pk": "1rljm2tcHxAp8c8LzVaHhyXZaM7qFyI89",  # Example ID
        "ots_train_ffa_3_19.pk": "1-V9F7tFhRNKp5pKKnJLBcLX8MXZdHDLg",  # Example ID
    }
    
    for model_name, model_id in model_urls.items():
        try:
            print(f"\nDownloading {model_name}...")
            gdown.download(
                url=f"https://drive.google.com/uc?id={model_id}",
                output=os.path.join(output_dir, model_name),
                quiet=False
            )
            print(f"✅ {model_name} downloaded!")
        except Exception as e2:
            print(f"Failed to download {model_name}: {e2}")

print("\nProcess complete. Check the trained_models folder.")