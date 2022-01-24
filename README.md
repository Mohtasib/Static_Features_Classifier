# Static Features Classifier

This is a static features classifier for Point-Could clusters using an Attention-RNN model 




# Installation

```bash
$ mkdir Static_Features_Classifier
$ git clone >>>>>>>
$ cd Static_Features_Classifier
$ pip install -e .
```

# Example Code

You can simple run the following command to run a sample code

```bash
$ python examples/train_and_evaluate.py
```

# Usage
#### 1. Import Libraries:
```bash
import numpy as np
from sfc.util import create_dataset
from sfc.models.Attention_RNN import Attention_RNN
```

#### 2. Define the constants:
```bash
NUM_FEATURES = ...
```

#### 3. Define the directories:
```bash
DATA_PATH = ...
My_Model_Logs_DIR = ...
My_Model_Weights = My_Model_Logs_DIR + 'Best_Attention_RNN_ckpt.h5'
```

#### 4. Create the dataset:
```bash
features, labels = create_dataset(DATA_PATH, NUM_FEATURES)
```

#### 5. Create the model:
```bash
My_Model = Attention_RNN(Logs_DIR=My_Model_Logs_DIR)
```

#### 6. Load the data into the model:
```bash
My_Model.x_train = features
My_Model.y_train = labels
My_Model.x_test = ...
My_Model.y_test = ...
```

#### 7. Train the model:
```bash
My_Model.Fit()
```

#### 8. Evaluate the model:
```bash
My_Model.Evaluate()
```

#### 9. Predict using the model:
```bash
predictions = My_Model.Predict(data)
```

#### NOTE: More details will be available soon!!!# Static_Features_Classifier
