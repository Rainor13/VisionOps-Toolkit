 
from P1 import *
import numpy as np

if __name__ == '__main__':

    img = cv.imread('/home/rainor/Escritorio/VA/Entrega/EXAMEN/imageP4.png', 0)

    seeds = np.array([ [24,24] ])
    SE = np.array([ [1,1,1], [1,1,1], [1,1,1] ])

    fill(img, seeds, SE, [])


    seeds1 = np.array([ [21,44], [5,5] ])
    SE1 = []

    # fill(img, seeds1, SE1, [])