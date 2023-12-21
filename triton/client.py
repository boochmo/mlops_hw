from functools import lru_cache

import numpy as np
import struct
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient.utils import np_to_triton_dtype

@lru_cache
def get_client():
    return InferenceServerClient(url="0.0.0.0:8500")


def call_iris_classifier(features: str):
    triton_client = get_client()
    #features = parse_input(features)

    input_features = InferInput(
        name="float_input", shape=features.shape, datatype="FP32"
    )
    input_features.set_data_from_numpy(features, binary_data=True)

    query_response = triton_client.infer(
        "iris_classifier",
        [input_features],
        outputs=[
            InferRequestedOutput("label", binary_data=True),
            InferRequestedOutput("probabilities", binary_data=True),
        ],
    )

    label = query_response.as_numpy("label")
    probas = query_response.as_numpy("probabilities")

    return label, probas

def main():
    batch = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [7, 3.2, 4.7, 1.4],
        [7.7, 2.6, 6.9, 2.3],
        [5.2, 2.7, 3.9, 1.4]

    ], dtype=np.float32)

    output = call_iris_classifier(batch)
    #тест
    output_labels = output[0].astype('U13')
    target = np.array(['setosa', 'versicolor', 'virginica', 'versicolor']).astype('U13')
    assert(np.equal(output_labels, target).any())
    print(output)
    return output

if __name__ == "__main__":
    main()
