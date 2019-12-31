import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly.offline import plot, init_notebook_mode
init_notebook_mode(connected=True)
import seaborn as sns
from tqdm import tqdm
from lyft_dataset_sdk.lyftdataset import LyftDataset
from IPython.core.debugger import set_trace

DATA_PATH = '/home/vignesh/workspace/lyft_3d/data/proper/3d-object-detection-for-autonomous-vehicles/'
lyft_dataset = LyftDataset(data_path=DATA_PATH, json_path=DATA_PATH + 'train_data')
my_scene = lyft_dataset.scene[0]

def render_scene(index):
    my_scene = lyft_dataset.scene[index]
    my_sample_token = my_scene["first_sample_token"]
    lyft_dataset.render_sample(my_sample_token)


render_scene(0)
