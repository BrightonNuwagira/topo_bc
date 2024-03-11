import pandas as pd
import numpy as np
import numpy as np
import random
import os
import glob
from PIL import Image
from gtda.homology import CubicalPersistence
from gtda.diagrams import BettiCurve
from PIL import Image




Datta_final_list2 = []
for i in images_Datta1:
    img2 = Image.open(i).convert('L')
    resized_array =   img2.resize((224, 224))
    arry_image2 = np.array(resized_array)
    homology_dimensions2 = [0,1]
    cubical_persistence2 = CubicalPersistence(homology_dimensions=homology_dimensions2, coeff=3, n_jobs=-1)
    im8_cubical2 = cubical_persistence2.fit_transform(np.array( arry_image2)[None, :, :])
    Betti_Curve = BettiCurve()
    Fitted_Betti_curve2 = Betti_Curve.fit_transform(im8_cubical2)
    Reshaped_1002 = np.reshape(Fitted_Betti_curve2, 200)
    testpredict_12 = np.array(Reshaped_1002)
    Datta_final_list2.append(testpredict_12.tolist())

df_Datta_12 = pd.DataFrame(Datta_final_list2)
concatenated_df = pd.concat([df_Datta_12, df['Label']], axis=1)
