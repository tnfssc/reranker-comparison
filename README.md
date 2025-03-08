# Comparing various reranking models

This project compares the performance of various reranking models, mostly open weights models.

## Sample comparison

Note that this just one of the many possible comparisons. Please refer to the `requests.example.json` file, it has a basic example. Rerank time here is just for the first rerank. Doing several reranks consecutively has a much improved performance.

| Model                       | Doc1   | Doc2   | Doc3   | Doc4   | Doc5   | Rerank Time | Init Time |
| --------------------------- | ------ | ------ | ------ | ------ | ------ | ----------- | --------- |
| gte_base                    | 0.3982 | 0.2500 | 0.2776 | 0.7277 | 0.3444 | 0.7269      | 2.3768    |
| jina_v2_base                | 0.1885 | 0.0565 | 0.0481 | 0.8554 | 0.0628 | 0.8549      | 1.5764    |
| bge_base                    | 0.8766 | 0.3266 | 0.0124 | 0.9998 | 0.0005 | 0.5601      | 3.9060    |
| bge_large                   | 0.0013 | 0.0001 | 0.0002 | 0.9987 | 0.0001 | 1.2473      | 1.0864    |
| bge_v2_m3                   | 0.0119 | 0.0012 | 0.0000 | 0.9970 | 0.0005 | 1.3595      | 0.9870    |
| bge_v2_gemma                | 0.7710 | 0.5671 | 0.1016 | 0.9999 | 0.1790 | 5.7820      | 1.7159    |
| bge_v2_minicpm_layerwise_20 | 0.0007 | 0.0000 | 0.0000 | 0.8624 | 0.0000 | 6.2769      | 1.6956    |
| bge_v2_minicpm_layerwise_28 | 0.0007 | 0.0000 | 0.0000 | 0.9258 | 0.0000 | 9.6686      | 1.5589    |
| bge_v2_minicpm_layerwise_40 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 10.1261     | 1.6054    |
| cohere_v35                  | 0.1729 | 0.0888 | 0.0866 | 0.8743 | 0.1076 | 1.2758      | 0.0000    |

## Setup

Install `uv` from [astral.sh](https://docs.astral.sh/uv/getting-started/installation)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open the project directory in terminal and enter `uv sync` to install the dependencies

```bash
uv sync
```

Create a `.env` file and populate

```bash
cp .env.example .env
```

Create a `requests.json` file and populate

```bash
cp requests.example.json requests.json
```

## Run the project

```bash
uv run main.py
```

This will create a new `responses_<timestamp>.json` file, It will be similar in structure to the `responses.example.json` file

## Troubleshooting

### `ModuleNotFoundError: No module named 'xyz'`

Make sure that you ran `uv sync` before running the project

This error may occurs when the `libbz2-dev` package is not installed or linked

```bash
sudo apt-get install libbz2-dev -y # Install the package
```

```bash
cp /usr/lib/python3.12/lib-dynload/_bz2.cpython-312-x86_64-linux-gnu.so ./.venv/lib/python3.12/site-packages/ # Link the module
# The module name may vary depending on the python version
```

This error may also occur when the lzma module is not installed

```bash
sudo apt-get install liblzma-dev -y # Install the package
```

```bash
cp /usr/lib/python3.12/lib-dynload/_lzma.cpython-312-x86_64-linux-gnu.so ./.venv/lib/python3.12/site-packages/ # Link the module
# The module name may vary depending on the python version
```

### `nvcc not found`

This error occurs when the CUDA toolkit is not installed. Install using the instructions from [Nvidia documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#network-repo-installation-for-ubuntu)
