import glob
import os 

rgb = sorted(
            glob.glob(os.path.join('/media/dataset2/ShapeNet/cars_train', "*", "rgb", "*"))
        )

print(f'list rgb {len(rgb)}')