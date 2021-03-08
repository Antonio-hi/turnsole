import cv2
import turnsole

if __name__ == '__main__':
	img = cv2.imread('./images/sunflower.jpg')
	img = turnsole.resize_with_pad(img, side_length=512)
	cv2.imshow('image', img)
	cv2.waitKey()
