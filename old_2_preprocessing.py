# Python 3
# packages
from PIL import Image, ImageEnhance
import os
import cv2

in_path = os.getcwd() + "/input"

if os.path.exists(in_path) == True:
    in_list = os.listdir(in_path)
    if len(in_list) == 0:
        print("Note: Please add some image files for preprocessing")
    else:
        os.makedirs(os.getcwd() + '/preprocessed/', exist_ok=True)
        # output directory creation
        os.makedirs(os.getcwd() + '/cropped/', exist_ok=True)
        for img in in_list:
            with Image.open(os.getcwd() + "/input/" + img) as im:
                print("[+] Preprocessing started...")
                # convert into grayscale
                grayscaleImg = im.convert("L")

                # image contrast enhancer
                enhancer = ImageEnhance.Contrast(grayscaleImg)

                factor = 1.5  # increase contrast
                contrastImg = enhancer.enhance(factor)

                # image brightness enhancer
                enhancer = ImageEnhance.Brightness(contrastImg)

                factor = 2  # increase Brightness
                im_output = enhancer.enhance(factor)

                # Save the processed image
                im_output.save(os.getcwd() + '/preprocessed/' + img)
                print("[+] Preprocessing completed")

                im_ap = cv2.imread(os.getcwd() + '/preprocessed/' + img)

                # Convert to grayscale
                gray = cv2.cvtColor(im_ap, cv2.COLOR_BGR2GRAY)

                # Apply binary thresholding
                ret, thresh = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

                # Choosing the right kernel
                rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
                dilation = cv2.dilate(thresh, rect_kernel, iterations=10)

                # Finding contours
                contours, hierarchy = cv2.findContours(
                    dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                im_mark = im_ap.copy()
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(im_mark, (x, y), (x + w, y + h), (0, 255, 0), 2)

                print("[+] Image Cropping started...")
                img = img.split(".")
                for i, contour in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(contour)
                    roi = im_mark[y:y+h, x:x+w]
                    cv2.imwrite('./cropped/' + img[0] + '_' + str(i) + '.jpeg', roi)

                print("[+] Image Cropping completed")
else:
    print("Note: Please create the input directories with image files")
