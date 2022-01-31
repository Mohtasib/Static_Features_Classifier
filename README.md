# Static Features Classifier

This is a static features classifier for Point-Could clusters using an Attention-RNN model 




# Installation

```bash
$ git clone https://github.com/Mohtasib/Static_Features_Classifier.git
$ cd Static_Features_Classifier
$ conda create -n sfc_env python==3.6.10
$ conda activate sfc_env
$ pip install -e .
```

# Example Code

You can simply run the following command to execute a sample code

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
MAX_SEQ_LEN = ...
NUM_FEATURES = ...
```

#### 3. Define the directories:
```bash
DATA_PATH = ...
My_Model_Logs_DIR = ...
My_Model_Weights = My_Model_Logs_DIR + 'Best_Attention_RNN_ckpt.h5'
```

#### 4. Create the model:
```bash
My_Model = Attention_RNN(Logs_DIR=My_Model_Logs_DIR)
```

#### 5. Create the dataset and load it into the model:
```bash
My_Model.x_train, My_Model.y_train = create_dataset(DATA_PATH + 'train/', MAX_SEQ_LEN, NUM_FEATURES)
My_Model.x_test, My_Model.y_test = create_dataset(DATA_PATH + 'test/', MAX_SEQ_LEN, NUM_FEATURES)
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

#### NOTE: More details will be available soon!!!
