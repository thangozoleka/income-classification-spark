
---

## 🔹 preprocess.py

```python
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler


def build_preprocessing_pipeline(df, label_col):
    categorical_columns = [field.name for field in df.schema.fields
                           if field.dataType.simpleString() == "string" and field.name != label_col]

    numeric_columns = [field.name for field in df.schema.fields
                       if field.dataType.simpleString() != "string" and field.name != label_col]

    indexers = [StringIndexer(inputCol=col, outputCol=col + "_indexed", handleInvalid='keep')
                for col in categorical_columns]

    encoders = [OneHotEncoder(inputCol=col + "_indexed", outputCol=col + "_encoded")
                for col in categorical_columns]

    assembler_inputs = [col + "_encoded" for col in categorical_columns] + numeric_columns
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

    return indexers, encoders, assembler
