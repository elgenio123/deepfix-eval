from deepfix_sdk import DeepFixClient
import os

# os.environ["DEEPFIX_API_KEY"] = "" # get your API key at https://deepfix.delcaux.com/

client = DeepFixClient(api_url="http://localhost:8844//v2/analyse")

"""# Tabular data

## Classification
"""

from deepfix_sdk.data.datasets import TabularDataset
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X,y = load_iris(as_frame=True,return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
dataset_name = "iris_classification"

label = "target"
train = X_train.copy()
train[label] = y_train
cat_features = X_train.select_dtypes(include=['object','string','category']).columns.tolist()
if len(cat_features) > 0:
    cat_features = None

test = X_test.copy()
test[label] = y_test

train_data = TabularDataset(dataset=train, dataset_name=dataset_name, label=label, cat_features=cat_features)
val_data = TabularDataset(dataset=test, dataset_name=dataset_name, label=label, cat_features=cat_features)

train_data.data.head()

# Fit model
model_name = "HistGradientBoostingClassifier"
clf = HistGradientBoostingClassifier(max_depth=3)
clf = clf.fit(train_data.X, train_data.y)

result = client.get_diagnosis(
    train_data=train_data,
    test_data=val_data,
    model_name=model_name,
    model=clf,
    language="english",
)

# Visualize results
print(result.to_text(verbose=False))
# print(result.to_markdown())





