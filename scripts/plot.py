import sys
sys.path.append("./")
from r2_gaussian.utils.plot_utils import show_two_volume 
import numpy as np

vo1 = "output/by_experiment/aorta_syn_spiral_ntrain_ablation/r2_gaussian/syn_aorta_spiral_ntrain50/point_cloud/iteration_30000/vol_gt.npy"
vo2 = "output/by_experiment/aorta_syn_spiral_ntrain_ablation/r2_gaussian/syn_aorta_spiral_ntrain50/point_cloud/iteration_30000/vol_pred.npy"

vol1 =np.load(vo1)
vol2 = np.load(vo2)
show_two_volume(vol1,vol2)