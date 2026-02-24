import numpy as np
import viridis_colormap as viridis_map

"""
class for mapping values to colormap,
For now it can handle HSV and VIRIDIS
"""

class CustomImg:

    # Init funtion colormap should be str HSV OR VIRIDIS
    def __init__(self):
        self.colors = viridis_map.var
    # Map colors
    def map_colors(self,img) :
        img = self.colors[img.astype(int)]
        return img
     # Prepare custome image
    def get_new_img(self,img) :
        return self.map_colors(img)
"""
Class for making power values to custom indexes
like normalization
"""

class NormalizePowerValue:
    def __init__(self,step_size=.5):
        self.step_size = step_size
    def get_normalized_values(self,img):
        # for now lets make range from -120 to -20 dbm with .5 dbm step
        img = np.clip(img,-130,-3)
        img = np.round(img/self.step_size) * self.step_size
        img = np.abs((img - (-130)) / self.step_size)
        print("AI COLOR MAP MIN MAX ")
        return img
