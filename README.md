## A three-dimensional dynamic wake prediction framework for multiple turbine operation states based on diffusion model

> Mengyang Song, Jiancai Huang, Xuqiang Shao, Shiao Zhao, Chenyu Ma, Zaishan Qi

As the paper is currently under review, only the test code, trained models, and test dataset are available. The full code, models, and data will be released upon acceptance.

To test the trained model, please download the supplementary data from [zenodo](https://zenodo.org/records/14569344) and extract it to the project root directory. After extraction, the project root directory should contain:
- dataset_sliced
- logs  
- my_LDM  
- reference_repo
- environment.yml  
- README.md  
- visualization.ipynb
- LICENSE

### Environment

Two approach is provided to install the required environment.

#### Conda

In this approach, [Miniconda](https://docs.anaconda.com/miniconda/install/)/Anaconda is required to install the necessary Python packages. In the project root directory, run:

```shell
conda env create -f environment.yml
```

If you encounter installation issues and cannot solve it, another way to install the packages is (this has a high chance to work):

1. create a new environment:

```shell
conda create -n torch240_cuda124_xformers python=3.10.14 -y
conda activate torch240_cuda124_xformers 
```

2. Run the following commands to manually install the environment:

```shell
# if you are on Linux
conda install pytorch=2.4.0 torchvision=0.19.0 torchaudio=2.4.0 pytorch-cuda=12.4 xformers=0.0.27.post2 -c pytorch -c nvidia -c xformers
# if you are on Windows
conda install pytorch=2.4.0 torchvision=0.19.0 torchaudio=2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y

conda install numpy scipy scikit-learn notebook matplotlib pandas tqdm einops pygwalker seaborn regex torchinfo accelerate kornia torchmetrics diffusers lightning -c conda-forge -y

# if you are on Linux
pip install timm lpips datasets tensorboard omegaconf
# if you are on Windows
pip install timm lpips datasets tensorboard omegaconf xformers==0.0.27.post2
```

3. As these commands were used in my early environment configuration, you may need to install additional packages to run `visualization.ipynb`. Please refer to the Python interpreter's missing package reports and install them sequentially. Use conda installations whenever possible to avoid package conflict issues. If you encounter pip errors and warnings, proceed as long as the target package installation is successful (pip is shit).

For more information, check `visualization.ipynb`.

#### uv

In this approach, [uv](https://docs.astral.sh/uv/getting-started/installation) is reqired for managing this project's environment.

1. sync project's environment with following code:

```shell
uv sync
```

2. uncomment line 12 in `pyproject.toml` shown below, and rerun `uv sync`:

```toml
    # "xformers==0.0.28.post1", # 这个包依赖写的有问题，先注释掉sync一下，再解除注释sync
```

3. run `visualization.ipynb` and follows the reported error to `.venv/lib/python3.10/site-packages/floris/core/farm.py", line 174`. Should found the code below:

```python
if is_unique_path and in_external and in_internal:
    raise ValueError(
        f"The turbine type: {t} exists in both the internal and external"
        " turbine library."
    )
```

and modify the code (uncomment and add `pass`) to:

```python
if is_unique_path and in_external and in_internal:
    # raise ValueError(
    #     f"The turbine type: {t} exists in both the internal and external"
    #     " turbine library."
    # )
    pass
```
