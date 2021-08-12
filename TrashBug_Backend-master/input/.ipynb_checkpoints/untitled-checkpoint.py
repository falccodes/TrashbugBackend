from skimage import io, img_as_float
import imquality.brisque as brisque
img = img_as_float(io.imread ("C:\\Users\\Ekansh\\Pictures\\my.jpeg", as_gray=True))
scor = brisque.score(img)
print("Brisque score = ", scor)