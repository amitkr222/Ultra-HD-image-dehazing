# USAGE
# python ssim.py --first 22_outdoor_gt.jpg --second test_result/22_outdoor_4.jpg

# import the necessary packages
#from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
	help="first input image")
ap.add_argument("-s", "--second", required=True,
	help="second")
args = vars(ap.parse_args())

# load the two input images
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])

# convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM value: {}".format(score))
#cv2.imshow('Image 1', imageA)
#cv2.imshow('Image 2', imageB)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
