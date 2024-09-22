# Python 3
# packages
from PIL import Image, ImageEnhance
import os
import cv2
import pytesseract  # Make sure to import pytesseract

in_path = os.getcwd() + "/input"

def perform_ocr(cropped_dir, ground_truth_dir, lang='eng'):
    """
    Performs OCR on all images in the cropped directory and saves the extracted text.
    
    :param cropped_dir: Directory containing cropped images.
    :param ground_truth_dir: Directory to save ground truth text files.
    :param lang: Language for Tesseract OCR (default is English).
    """
    os.makedirs(ground_truth_dir, exist_ok=True)
    
    merged_texts = {}

    for img_file in os.listdir(cropped_dir):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            print(f"[-] Skipping unsupported file format: {img_file}")
            continue
        
        img_path = os.path.join(cropped_dir, img_file)
        base_name = os.path.splitext(img_file)[0]  # Get base filename without extension
        
        try:
            text = pytesseract.image_to_string(Image.open(img_path), lang=lang)
            text = text.strip()
            
            # Merge the text into the dictionary
            if base_name not in merged_texts:
                merged_texts[base_name] = []
            merged_texts[base_name].append(text)
            
            print(f"[+] OCR completed for {img_file}.")
        
        except Exception as e:
            print(f"[-] OCR failed for {img_file}. Check log for details: {e}")

    # Save merged text to a single file for each original image
    for base_name, texts in merged_texts.items():
        merged_text = "\n".join(texts)  # Join all lines with a newline
        text_file_path = os.path.join(ground_truth_dir, f"{base_name}.txt")
        with open(text_file_path, 'w', encoding='utf-8') as f:
            f.write(merged_text)
        print(f"[+] Merged text saved to {text_file_path}")

if os.path.exists(in_path):
    in_list = os.listdir(in_path)
    if len(in_list) == 0:
        print("Note: Please add some image files for preprocessing")
    else:
        os.makedirs(os.getcwd() + '/preprocessed/', exist_ok=True)
        os.makedirs(os.getcwd() + '/cropped/', exist_ok=True)
        os.makedirs(os.getcwd() + '/ground_truth/', exist_ok=True)

        for img in in_list:
            with Image.open(os.path.join(in_path, img)) as im:
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
                im_output.save(os.path.join(os.getcwd(), 'preprocessed', img))
                print("[+] Preprocessing completed")

                im_ap = cv2.imread(os.path.join(os.getcwd(), 'preprocessed', img))

                # Convert to grayscale
                gray = cv2.cvtColor(im_ap, cv2.COLOR_BGR2GRAY)

                # Apply binary thresholding
                ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

                # Choosing the right kernel
                rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
                dilation = cv2.dilate(thresh, rect_kernel, iterations=10)

                # Finding contours
                contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                im_mark = im_ap.copy()
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(im_mark, (x, y), (x + w, y + h), (0, 255, 0), 2)

                print("[+] Image Cropping started...")
                img = img.split(".")
                for i, contour in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(contour)
                    roi = im_mark[y:y + h, x:x + w]
                    cv2.imwrite(os.path.join(os.getcwd(), 'cropped', f"{img[0]}_{i}.jpeg"), roi)

                print("[+] Image Cropping completed")

        print("All images have been preprocessed. Starting OCR...")
        perform_ocr(os.getcwd() + '/cropped/', os.getcwd() + '/ground_truth/', lang='eng')
        print("OCR processing completed. Check the 'ground_truth' directory for extracted text files.")
else:
    print("Note: Please create the input directories with image files")
