import numpy as np
import math as m
import matplotlib.pyplot as plt
import cv2 as cv
import matplotlib.image as image
import scipy.signal
import scipy
from scipy import misc

def normalizeImage(inImage):
    
    aux = (inImage - np.min(inImage)) / (np.max(inImage) - np.min(inImage))
    return aux

def adjustIntensity(inImage, inRange = [], outRange = [0, 1]):
    
    outImage = np.zeros((inImage.shape[0],inImage.shape[1]))

    if inRange == []:
        inRange = np.zeros(2)
        inRange[0] = np.amin(inImage)
        inRange[1] = np.amax(inImage)

    for i in range(inImage.shape[0]):
        for j in range(inImage.shape[1]):

            numerador = outRange[0]+((outRange[1]-outRange[0])*(inImage[i, j]-inRange[0]))
            divisor = (inRange[1]-inRange[0])

            outImage[i, j] = numerador/divisor

    plt.title("Histograma original", loc='center')
    plt.hist(inImage.flatten(), 100, [np.amin(inImage), np.amax(inImage)], color = 'g')
    plt.figure()
    plt.title("Histograma procesado", loc='center')
    plt.hist(outImage.flatten(), 100, outRange, color = 'r')
    plt.figure()
    plt.title("Imagen original", loc='center')
    plt.imshow(inImage, cmap='gray')
    plt.figure()
    plt.title("Imagen procesada", loc='center')
    plt.imshow(outImage, cmap='gray')
    plt.show()

    return outImage

def equalizeIntensity (inImage, nBins = 256):

    normimage = normalizeImage(inImage)
    # fig1 = plt.figure()
    # fig1.add_subplot(1,2,1)
    hist, bins = np.histogram(normimage.flatten(), nBins)
    # fig1.add_subplot(1,2,2)
    # plt.imshow(normimage, cmap='gray') 

    cdf = np.cumsum(hist)
    cdfnorm = cdf * hist.max()/ cdf.max()

    outImage = np.zeros(256, dtype=int)
    # luma_levels = 256
    for i in range(hist.size):
        outImage[i]= (cdfnorm[i] - cdfnorm.min())*255/(cdfnorm.max()-cdfnorm.min())


    new_image = inImage.copy()
    image_row, image_col = inImage.shape
    new_image[:] = list(map(lambda a : outImage[a], inImage[:]))
    new_image = adjustIntensity(new_image, [], [0, 1])

    cv.imwrite('equalized.png', new_image)  

    fig2 = plt.figure()
    fig2.add_subplot(1,1,1)
    plt.hist(inImage.flatten(), nBins, color = 'r')
    fig2.add_subplot(1,1,1)
    plt.plot(cdfnorm, color = 'b')
    # fig2.add_subplot(1,2,2)
    plt.figure()
    plt.imshow(normimage, cmap='gray')

    fig3 =plt.figure()
    fig3.add_subplot(1,2,1)
    hist2, bins, movida = plt.hist(new_image.flatten(), nBins, color = 'r')
    cdf2 = np.cumsum(hist2)
    cdfnorm2 = cdf2 * hist2.max()/ cdf2.max()
    fig3.add_subplot(1,2,2)
    plt.plot(cdfnorm2, color = 'b')
    # fig3.add_subplot(1,3,3)
    plt.figure()
    plt.imshow(new_image, cmap='gray') 
    plt.show()    

    return new_image

def filterImage (inImage, kernel):
    
    normImage = normalizeImage(inImage)
    outImage = np.zeros_like(inImage, dtype = "float32") #Si quieres ver la imagen mal pero guapisima en modo QR quita el float

    image_row, image_col = inImage.shape #Tamanho de la imagen normalizada

    rows, columns = kernel.shape #Tamanho del kernel de las filas y columnas

    centerR = m.trunc(rows/2) #filas
    centerC = m.trunc(columns/2) #columnas
 
    window = np.zeros(shape=(kernel.shape)) #Creamos la ventana que vamos a aplicar en el bucle aplicando pixel a pixel el kernel

    #Aqui creamos la imagen con bordes con 0's para poder aplicar el kernel en los bordes de la imagen
    
    image_padded = np.zeros((image_row + 2 * centerR, image_col + 2 * centerC))
    image_padded[centerC:(-1*centerC), centerR:(-1*centerR)] = inImage

    for i in range(image_row):
        for j in range(image_col):
            outImage[i,j] = np.sum(image_padded[i:i+rows,j:j+columns]*kernel)

    outImage = adjustIntensity(outImage, [], [0,1]) #IGUAL QUITAR LA MOVIDA

    plt.imshow(inImage, cmap='gray') 
    plt.figure()
    plt.imshow(outImage, cmap='gray')
    plt.show()  

    return outImage

def gaussKernel1D(sigma):
   size = round(2*(3*sigma)+1)
   kernel = np.zeros(size)
   mid = m.floor(size/2)
   kernel=[(1/(sigma*np.sqrt(2*np.pi)))*(1/(np.exp((i**2)/(2*sigma**2)))) for i in range(-mid,mid+1)]
   return kernel

def gaussianFilter(inImage, sigma):
     
    kernel = np.hstack(gaussKernel1D(sigma))
    tkernel = np.transpose(kernel)
    new_kernel = np.outer(kernel,tkernel)
    return filterImage(inImage, new_kernel)

def medianFilter(inImage, filter_size):

    normImage = normalizeImage(inImage)
    image_row, image_col = normImage.shape 
    outImage = np.zeros_like(normImage, dtype = "float32") #Creamos el tamanho de la imagen de salida

    window = np.zeros((filter_size, filter_size))
    rows, columns = window.shape    
    centerR = m.trunc(filter_size/2) #tanto de filas como de columnas

    image_padded = np.zeros((image_row + 2 * centerR, image_col + 2 * centerR))
    image_padded[centerR:(-1*centerR), centerR:(-1*centerR)] = inImage

    for i in range(0, image_row-1): 
        for j in range(0, image_col-1): 
            
            window = image_padded[i: i+rows, j: j+columns]
            outImage[i, j] = np.median(window)

    outImage = adjustIntensity(outImage, [], [0,1])

    plt.imshow(inImage, cmap='gray') 
    plt.figure()
    plt.imshow(outImage, cmap='gray')
    plt.show()

    return outImage

def highBoost (inImage, A, method, param):

    inImage = np.array(inImage, dtype = "float32")
    outImage = np.zeros_like(inImage, dtype = "float32")
    if method == "gaussian":
        outAuxImg = gaussianFilter(inImage, param)
        outImage = inImage*A - outAuxImg  
        print(outImage)

    elif method == "median":
        outAuxImg = medianFilter(inImage, param)
        outImage = inImage*A - outAuxImg
    
    # outImage = adjustIntensity(outImage, [], [0,255]) //CON ESTE ADJUST DA EN GRIS
    cv.imwrite('highBoost.png', outImage)
    plt.imshow(outImage, cmap='gray')
    plt.show()
    return outImage

def erode (inImage, SE, center=[]):

    plt.imshow(inImage, cmap='gray')
    rows, columns = SE.shape

    if center == []:
        center = [m.trunc(SE.shape[0]/2), m.trunc(SE.shape[1]/2)]
    
    image_row, image_col = inImage.shape 
    centerpadR = m.trunc(SE.shape[0]/2)
    centerpadC = m.trunc(SE.shape[1]/2)

    window = np.zeros((rows, columns))
    
    centerR = center[0] #Tambien lo vamos a usar para saber la distancia al centro de izquierda y asi usar una ventana distinta en los bordes
    centerC = center[1] #Lo mismo para el vertical para ajustar la ventana

    image_padded =  np.full((image_row + 4 * centerpadR, image_col + 4 * centerpadC), 255)
    image_padded[centerpadR+1:((-1*centerpadR)-1), centerpadC+1:((-1*centerpadC)-1)] = inImage 

    image_paddedAux =  np.full((image_row + 4 * centerpadR, image_col + 4 * centerpadC), 255)


    i = 0
    i += 2
    j = 0
    j += 2

    for i in range (image_row):
        for j in range (image_col):
            window = image_padded[i: i+rows, j: j+columns]
            x = i
            window = window.flatten()
            auxBool = False
            for x in range (SE.flatten().size): #tengo que poner el centro 
                if (SE.flatten()[x] == 0):
                    if (SE.flatten()[x] == window[x]):
                        auxBool = True
                    else:
                        auxBool = False
                        break
            if auxBool == True:
                image_paddedAux[i+centerR][j+centerC] = 0

    outImage = image_paddedAux[1+centerpadR:((-1*centerpadR)-1), 1+centerpadC:((-1*centerpadC)-1)] 
    
    plt.figure()
    plt.imshow(outImage, cmap='gray')
    plt.show()

    return outImage

def dilate (inImage, SE, center=[]):

    # plt.imshow(inImage, cmap='gray')
    rows, columns = SE.shape

    if center == []:
        center = [m.trunc(SE.shape[0]/2), m.trunc(SE.shape[1]/2)]
    
    image_row, image_col = inImage.shape 

    window = np.zeros((rows, columns))

    centerR = center[0] #Tambien lo vamos a usar para saber la distancia al centro de izquierda y asi usar una ventana distinta en los bordes
    centerC = center[1] #Lo mismo para el vertical para ajustar la ventana

    centerpadR = m.trunc(SE.shape[0]/2)
    centerpadC = m.trunc(SE.shape[1]/2)

    image_padded =  np.full((image_row + 4 * centerpadR, image_col + 4 * centerpadC), 255)
    image_padded[centerpadR+1:((-1*centerpadR)-1), centerpadC+1:((-1*centerpadC)-1)] = inImage 

    image_paddedAux =  np.full((image_row + 4 * centerpadR, image_col + 4 * centerpadC), 255)
    image_paddedAux[centerpadR+1:((-1*centerpadR)-1), centerpadC+1:((-1*centerpadC)-1)] = inImage 

    outImage = inImage.copy()

    i = 2 # Por el padeo de la imagen ya que metemos 2 filas de ceros de esta forma nos lo saltamos ya que siempre metemos 2 filas de 0 a los bordes
    j = 2
    for i in range (image_row):
        for j in range (image_col):
            window = image_padded[i+1: i+rows+1, j+1: j+columns+1]
            if (inImage[i][j] == 0):
                for x in range (rows): #tengo que poner el centro 
                    for y in range (columns): #AQUI PETA ESTA MIERDA
                        if (window[x][y] == 255 and SE[x][y] == 0):
                            window[x][y] = 0
                
                image_paddedAux[1+i+centerpadR-centerR:1+i+rows+centerpadR-centerR, 1+j+centerpadC-centerC:1+j+columns+centerpadC-centerC] = window
                x=0
                y=0
    
    outImage = image_paddedAux[1+centerpadR:((-1*centerpadR)-1), 1+centerpadC:((-1*centerpadC)-1)]

    # plt.figure()
    # plt.imshow(outImage, cmap='gray')
    # plt.show()
    
    return outImage

def opening (inImage, SE, center=[]):

    plt.imshow(inImage, cmap='gray')
    plt.figure()
    aux = erode(inImage, SE, center)
    outImage = dilate(aux, SE, center)
    plt.imshow(outImage, cmap='gray')
    plt.show()
    return 

def closing (inImage, SE, center=[]):

    plt.imshow(inImage, cmap='gray')
    plt.figure()
    aux = dilate(inImage, SE, center)
    outImage = erode(aux, SE, center) 
    plt.imshow(outImage, cmap='gray')
    plt.show()
    return 

def invertImg (inImage):
    auxImage = inImage.copy()
    for i in range(auxImage.shape[0]):
        for j in range(auxImage.shape[1]):
            if (auxImage[i][j] == 0):
                auxImage[i][j] = 255
            else:
                auxImage[i][j] = 0
    return auxImage

def fill (inImage, seeds, SE=[], center=[]):

    if SE == []:
        SE = np.array([[255, 0, 255],[0, 0, 0],[255, 0, 255]])
    
    if center == []:
        center = [m.trunc(SE.shape[0]/2), m.trunc(SE.shape[1]/2)]

    centerR = center[0]
    centerC = center[1]

    image_row, image_col = inImage.shape 

    auxImage =  np.full((image_row , image_col), 255)  
    auxImage_row, auxImage_col = auxImage.shape 

    imgInvert = invertImg(inImage)

    for x in range(auxImage_col):
        for y in range(auxImage_row):
            for z in range(seeds.shape[0]):
                #Estoy primero poniendo z 1 porque en los bucles recorremos primero las columans y despues las filas
                if (seeds[z][1] == x and seeds[z][0] == y) == True:
                    auxImage[x][y] = 0
             

    Xd = dilate(auxImage, SE, center)
    Xk = auxImage
    
    Xk1 = cv.bitwise_not(Xd) #Para que la primera iteracion siempre se cumpla
    auxBool = False

    while (auxBool == False):
        auxImage = dilate(auxImage, SE, center)
        Xk = cv.bitwise_or(auxImage, imgInvert)
        if (Xk.all() == Xk1.all()):
            auxBool = True
        else:
            Xk1 = Xk.copy()

    outImage = cv.bitwise_and(inImage,Xk)
    

    plt.figure()
    plt.title("Resultado final", loc='center')
    plt.imshow(outImage, cmap='gray')
    plt.show()

def gradientImage(inImage, operator):
    if (operator == "Roberts"):
        kernelx = np.array([[-1, 0], [0, 1]])
        gx = filterImage(inImage, kernelx)
        kernely = np.array([[0, -1], [1, 0]])
        gy = filterImage(inImage, kernely)
    elif (operator == "CentralDiff"):
        kernelx = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])
        gx = filterImage(inImage, kernelx)
        kernely = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])
        gy = filterImage(inImage, kernely)
    elif (operator == "Prewitt"):
        kernelx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        gx = filterImage(inImage, kernelx)
        kernely = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        gy = filterImage(inImage, kernely)
    elif (operator == "Sobel"):
        kernelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        gx = filterImage(inImage, kernelx)
        kernely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        gy = filterImage(inImage, kernely)

    outImage = np.array([gx, gy])

    # plt.imshow(inImage, cmap='gray')
    # plt.figure()
    # plt.imshow(outImage, cmap='gray')
    # plt.show()

    return [gx, gy]

if __name__ == '__main__':

    img = cv.imread('/home/rainor/Escritorio/VA/LenaMini.png', 0)
    imgBynary = cv.imread('/home/rainor/Escritorio/VA/LenaBinary.png', 0)
    # imgnoisy = cv.imread('/home/rainor/Escritorio/VA/LenaNoisy.jpg', 0)
    # imgcameraman = cv.imread('/home/rainor/Escritorio/VA/cameramannoise.jpg', 0)
    imgcelebro = cv.imread('/home/rainor/Escritorio/VA/celebro.jpeg', 0)

    imgBlackWhite7 = np.array([
        [255, 255, 255, 255 ,255 ,255 ,255 ,255 ,255 ,255, 255 ,255 ,255 ,255 ,255 ,255], 
        [255, 255, 255, 255 ,255 ,255 ,255 ,255 ,255 ,255, 255 ,255 ,255 ,255 ,255 ,255], 
        [255, 255, 255, 0 ,0 ,0 ,255 ,255 ,255 ,255, 255 ,255 ,255 ,255 ,255 ,255], 
        [255, 255, 0, 0 ,0 ,0 ,0 ,255 ,255 ,255, 255 ,255 ,255 ,255 ,255 ,255], 
        [255, 255, 0, 0 ,0 ,0 ,0 ,255 ,255 ,255, 255 ,0 ,0 ,0 ,255 ,255], 
        [255, 255, 0, 0 ,0 ,0 ,255 ,255 ,255 ,255, 0 ,0 ,0 ,0 ,255 ,255],
        [255, 255, 255, 0 ,0 ,255 ,255 ,255 ,255 ,0, 0 ,0 ,0 ,0 ,255 ,255], 
        [255, 255, 255, 255 ,255 ,255 ,255 ,255 ,0 ,0, 0 ,0 ,0 ,255 ,255 ,255],
        [255, 255, 255, 255 ,255 ,255 ,255 ,0 ,0 ,0, 0 ,0 ,255 ,255 ,255 ,255], 
        [255, 255, 255, 255 ,255 ,255 ,0 ,0 ,0 ,0, 0 ,255 ,255 ,255 ,255 ,255],
        [255, 255, 255, 255 ,255 ,0 ,0 ,0 ,0 ,0, 255 ,255 ,255 ,255 ,255 ,255], 
        [255, 255, 255, 255 ,0 ,0 ,0 ,0 ,0 ,255, 255 ,255 ,255 ,255 ,255 ,255],
        [255, 255, 255, 255 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,255 ,255 ,255 ,255], 
        [255, 255, 255, 255 ,0 ,0 ,0 ,0 ,0 ,0, 0 ,0 ,255 ,255 ,255 ,255], 
        [255, 255, 255, 255 ,255 ,0 ,0 ,0 ,0 ,0, 0 ,255 ,255 ,255 ,255 ,255], 
        [255, 255, 255, 255 ,255 ,255 ,255 ,255 ,255 ,255, 255 ,255 ,255 ,255 ,255 ,255]])

    imagCierre = np.array([
        [255, 255, 255, 255 ,255 ,255 ,255 ,255 ,255 ,255, 255 ,255 ,255 ,255 ,255 ,255], 
        [255, 255, 0, 0 ,255 ,255 ,255 ,255 ,255 ,255, 0 ,0 ,255 ,255 ,255 ,255], 
        [255, 255, 0, 0 ,255 ,255 ,255 ,255 ,255 ,0, 0 ,255 ,255 ,255 ,255 ,255], 
        [255, 255, 255, 255 ,255 ,255 ,255 ,0 ,0 ,0, 255 ,255 ,255 ,255 ,255 ,255], 
        [255, 255, 255, 255 ,255 ,255 ,255 ,255 ,0 ,0, 255 ,255 ,255 ,255 ,0 ,255], 
        [255, 255, 255, 255 ,255 ,255 ,255 ,255 ,0 ,255, 255 ,255 ,0 ,0 ,0 ,255], 
        [255, 255, 255, 0 ,0 ,0 ,0 ,255 ,0 ,255, 255 ,0 ,0 ,0 ,255 ,255], 
        [255, 255, 0, 0 ,0 ,0 ,255 ,0 ,0 ,0, 0 ,0 ,0 ,255 ,255 ,255], 
        [255, 0, 0, 0 ,0 ,255 ,255 ,255 ,0 ,255, 255 ,255 ,255 ,255 ,255 ,255], 
        [255, 0, 255, 255 ,255 ,0 ,255 ,255 ,255 ,0, 255 ,255 ,255 ,255 ,255 ,255], 
        [255, 0, 255, 255 ,255 ,255 ,0 ,255 ,0 ,0, 255 ,255 ,255 ,255 ,255 ,255], 
        [255, 0, 255, 255 ,255 ,255 ,255 ,0 ,0 ,0, 255 ,255 ,255 ,255 ,255 ,255], 
        [255, 0, 0, 255 ,255 ,255 ,255 ,0 ,0 ,0, 255 ,0 ,255 ,255 ,255 ,255], 
        [255, 0, 0, 0 ,255 ,255 ,255 ,0 ,0 ,255, 255 ,255 ,255 ,255 ,255 ,255], 
        [255, 255, 0, 0 ,0 ,0 ,0 ,0 ,255 ,255, 255 ,255 ,255 ,255 ,255 ,255], 
        [255, 255, 255, 255 ,255 ,255 ,255 ,255 ,255 ,255, 255 ,255 ,255 ,255 ,255 ,255]])

    imgErase = np.array([[0, 0, 0 ,0, 0, 0, 0, 0 ,0, 0],
                          [0, 255, 0 ,0, 0, 0, 0, 0 ,0, 0],
                          [0, 0, 0 ,0, 0, 0, 255, 0 ,0, 0],
                          [0, 0, 0 ,0, 0, 255, 0, 0 ,0, 0],
                          [0, 0, 0 ,0, 0, 0, 0, 0 ,0, 0],
                          [0, 0, 255 ,0, 0, 0, 0, 0 ,255, 0],
                          [0, 0, 0 ,0, 0, 0, 0, 0 ,0, 0],
                          [0, 0, 0 ,0, 0, 255, 0, 0 ,0, 0],
                          [0, 0, 0 ,0, 0, 0, 0, 0 ,0, 0],
                          [0, 255, 0 ,0, 0, 0, 0, 255 ,0, 0]])

    imgDilate = np.array([[255, 255, 255, 255, 255, 255, 255, 255 ,255, 255],
                          [255, 0, 255, 255, 255, 255, 255,255 ,255, 255],
                          [255, 255, 255 ,255, 255, 255, 0, 255 ,255, 255],
                          [255, 255, 255 ,255, 255, 0, 255, 255 , 255, 255],
                          [255, 255, 255, 255, 255, 255, 255, 255 ,255, 255],
                          [255, 255, 0, 255, 255, 255, 255, 255 ,0, 255],
                          [255, 255, 255, 255, 255, 255, 255, 255 ,255, 255],
                          [255, 255, 255, 255, 255, 0, 255, 255 ,255, 255],
                          [255, 255, 255, 255, 255, 255, 255, 255 ,255, 255],
                          [255, 0, 255, 255, 255, 255, 255, 0 ,255, 255]])

    SE = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    SEFILL = np.array([[255, 0, 255],[0, 0, 0],[255, 0, 255]])

    imgBlackWhite3 = np.array([[255.,255.,255.,255.,255.],
                               [255.,0.,0.,0.,255.],
                               [255.,0.,0.,0.,0.],
                               [255.,0.,0.,0.,255.],
                               [255.,255.,255.,255.,255.]])
   
    imgBlackWhite4 = np.array([[255.,255.,255.,255.,255.],
                               [255.,255.,255.,255.,255.],
                               [255.,255.,0.,255.,255.],
                               [255.,255.,255.,255.,255.],
                               [255.,255.,255.,255.,255.]])

    fill_apuntes = np.array([[255, 255, 255, 0, 0, 255, 255, 255],
                    [255, 255, 255, 0, 255, 0, 255, 255],
                    [255, 255, 0, 255, 255, 0, 255, 255],
                    [255, 255, 0, 255, 255, 0, 255, 255],
                    [255, 0, 255, 255, 255, 0, 255, 255],
                    [255, 0, 255, 255, 255, 0, 255, 255],
                    [255, 255, 0, 255, 0, 255, 255, 255],
                    [255, 255, 255, 0, 255, 255, 255, 255],])



#--------------------------------------------------------------------------------------------------------------------------------

    # adjustIntensity(img, [], [10, 50])

    # adjustIntensity(img, [], [1, 10])

#--------------------------------------------------------------------------------------------------------------------------------

    equalizeIntensity(img, nBins = 256)

#--------------------------------------------------------------------------------------------------------------------------------

    kernel0 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) #Sharpened image

    kernel1 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    kernel2 = np.array([[-1, -1, -1], [-1, 4, -1], [-1, -1, -1]]) #Edge detection

    kernel3 = np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]]) #creo que Edge detecion

    kernel4 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])/9.0 #Blur

    kernel5 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16.0 #Gaussian blur

    # filterImage(img, kernel1)

#--------------------------------------------------------------------------------------------------------------------------------

    # print(gaussKernel1D(1))
    # print(gaussKernel1D(0.5))

#--------------------------------------------------------------------------------------------------------------------------------

    # gaussianFilter(img, 3)

#--------------------------------------------------------------------------------------------------------------------------------

    # medianFilter(imgcelebro, 3)
    # medianFilter(medianFilter(imgcelebro, 3), 3)
    # medianFilter(medianFilter(medianFilter(medianFilter(imgcelebro, 3), 3), 3), 3)

#--------------------------------------------------------------------------------------------------------------------------------

    # highBoost(img, -1, "gaussian", 1)
    # highBoost(img, 1, "gaussian", 1)
    # highBoost(img, 2, "gaussian", 1)
    # highBoost(img, 3, "gaussian", 1)
    # highBoost(img, 3, "gaussian", 20)
    # highBoost(img, 1, "gaussian", 5)

    # highBoost(img, -1, "median", 3)
    # highBoost(img, 1, "median", 3)
    # highBoost(img, 2, "median", 3)
    # highBoost(img, 3, "median", 3)
    # highBoost(img, 3, "median", 20)
    # highBoost(img, 1, "median", 5)

#--------------------------------------------------------------------------------------------------------------------------------

    # erode (imgBlackWhite7, SE, [])

    # erode (imgErase, SE, [])

#--------------------------------------------------------------------------------------------------------------------------------

    # dilate (imgDilate, SE, [0,0])

    # dilate (imgDilate, SE, [2,2])

    # dilate (imgBlackWhite7, SE, [])

    # dilate (imgDilate, SEFILL, [])

#--------------------------------------------------------------------------------------------------------------------------------

    # opening(imgBlackWhite7, SE, [])

#--------------------------------------------------------------------------------------------------------------------------------

    # closing(imagCierre, SE, [])

#--------------------------------------------------------------------------------------------------------------------------------

    seeds = np.array([[3,4]])
    # fill(fill_apuntes, seeds, [], [])             

#--------------------------------------------------------------------------------------------------------------------------------

    # gradientImage(img, "Roberts")

    # gradientImage(img, "CentralDiff")                  

    # gradientImage(img, "Prewitt")

    # gradientImage(img, "Sobel")
    

#--------------------------------------------------------------------------------------------------------------------------------

    # normImage(img)
    # normalizeImage(img)
    # plt.imshow(foto, cmap='gray')
    # plt.show()

#--------------------------------------------------------------------------------------------------------------------------------

