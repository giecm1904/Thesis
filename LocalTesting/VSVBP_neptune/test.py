import requests
import pprint


input = {
    "community": "community-test",
    "namespace": "namespace-test",
    "node_names": [
        "node_a", "node_b", "node_c", "node_d", "gpu_node_e"
    ],
    "node_memories": [
        30, 30, 30, 30, 30
    ],
    "node_cores": [
        10, 10, 10, 10, 10
    ],
    "gpu_node_names": [
        "gpu_node_e"
    ],
    "gpu_node_memories": [
        100
    ],
    "function_names": [
        "ns/fn_1", "ns/fn_2", "ns/fn_3", "ns/gpu_fn_4"
    ],
    "function_memories": [
        10, 10, 10, 10
    ],
    "function_max_delays": [
        100, 100, 100, 100
    ],
    "gpu_function_names": [
        "ns/gpu_fn_4"
    ],
    "gpu_function_memories": [
        50
    ],
    "actual_cpu_allocations": {
        "ns/fn_1": {
            "node_a": True,
            # "node_b": True,
            # "node_c": True
        },
        "ns/fn_2": {
            # "node_d": True,
            "node_b": True,
            "node_c": True
        },
        "ns/fn_3": {
            # "node_a": True,
            # "node_d": True,
            # "node_c": True
        },
        "ns/gpu_fn_4": {
            # "node_a": True,
            # "node_b": True,
            # "node_d": True
        }
    },
    "actual_gpu_allocations": {
    }
}


# input = {
#     'community': 'community-worker',
#     'namespace': 'openfaas-fn',
#     'node_names': [
#         'kind-worker', 'kind-worker2', 'kind-worker3'
#     ],
#     'gpu_node_names': [],
#     'function_names': ['openfaas-fn/prime-numbers'],
#     'gpu_function_names': [],
#     'node_cores': [7200, 6900, 7300],
#     'node_memories': [13398143488, 13083570688, 13534458368],
#     'gpu_node_memories': [],
#     'function_memories': [1248576000],
#     'gpu_function_memories': [],
#     'function_max_delays': [],
#     'actual_cpu_allocations': {
#         'openfaas-fn/prime-numbers': {'kind-worker': True}
#     },
#     'actual_gpu_allocations': {
#     }
# }


response = requests.request(method='get', url="http://localhost:5000/", json=input)

pprint.pprint(response.json())