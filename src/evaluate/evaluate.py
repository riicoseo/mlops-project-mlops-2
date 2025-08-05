import numpy as np
import pandas as pd

def evaluate(model, x_valid):
    return model.predict(x_valid).clip(0,10)