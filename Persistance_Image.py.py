import pandas as pd
import numpy as np
import random
import os
import glob
from PIL import Image
from gtda.homology import CubicalPersistence
from gtda.diagrams import BettiCurve
from PIL import Image
from gtda.diagrams import PersistenceImage

df = pd.DataFrame({'image': all_images, 'Label': labels})

images_Datta = [df['image'][i] for i in range(784)]
data_array = []
for i in images_Datta:
    image=Image.open(i).convert('L')
    gray_image =np.array(image)
    homology_dimensions = [1]
    cubical_persistence = CubicalPersistence(homology_dimensions=homology_dimensions, coeff=3, n_jobs= -1)
    X_cubical = cubical_persistence.fit_transform([gray_image])
    persistence_image = PersistenceImage(n_bins=50,n_jobs= -1)
    X_persistence_image = persistence_image.fit_transform(X_cubical)[:,0]
    data_array.append(X_persistence_image)
data_array_images = np.array(data_array)
data_array_images
