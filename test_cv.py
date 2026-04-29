from deepfix_sdk.client import DeepFixClient
from deepfix_sdk.zoo.datasets.foodwaste import load_train_and_val_datasets
from deepfix_sdk.data.datasets import ImageClassificationDataset
from dotenv import load_dotenv
load_dotenv()

# Initialize client
client = DeepFixClient(api_url="http://localhost:8844/v2/analyse")

# Load and wrap dataset
dataset_name = "cafetaria-foodwaste-v2"
train_data, val_data = load_train_and_val_datasets(
    image_size=448, batch_size=8, num_workers=4, pin_memory=False
)
train_data = ImageClassificationDataset(dataset_name=dataset_name, dataset=train_data)
val_data = ImageClassificationDataset(dataset_name=dataset_name, dataset=val_data)

# Diagnose dataset
result = client.get_diagnosis(
    train_data=train_data,
    test_data=val_data,
    batch_size=8,
    language="english",
)

print(result.to_text())
