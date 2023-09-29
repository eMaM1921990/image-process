from flask import Flask, make_response, request, send_file, Response
from PIL import Image
from pillow_heif import register_heif_opener
import numpy as np
import cv2
import rembg
import io
import base64


final_height = 1063
final_width = 826

def test_image(img):
    error_message = []
    for test_counter in range(3):
        if test_counter == 0:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if test_brightness(img_rgb,0.2,0.88) == 0:
                error_message.append('100ERRORmauvaise luminosité')
        if test_counter == 1:
            img2 = cv2.resize(img,(650,500))
            if test_bluriness(img2,20) == 0:
                error_message.append('100ERRORtrop flou')
        if test_counter == 2:
            if detect_face(img_rgb)[0] == 0:
                error_message.append('100ERRORpas de visage')
            elif detect_face(img_rgb)[1]<=1:
                error_message.append('100ERRORyeux non visibles')
    return error_message

def test_brightness(img,min_tres, max_tres):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grayscale image
    cols, rows = gray_image.shape
    brightness = np.sum(gray_image) / (255 * cols * rows) #luminosité moyenne
    if(brightness<min_tres or brightness>max_tres):
        return 0
    else:
        return 1

def test_bluriness(img,tres):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grayscale image
    bluriness = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    if bluriness<tres:
        return 0
    else:
        return 1

def detect_face(img):
    nb_eyes = 0
    nb_smiles = 0
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grayscale image
    face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_smile.xml')
    eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
    face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=7, minSize=(int(img.shape[0]/10), int(img.shape[0]/10))
    )
    for (x, y, w, h) in face:
        roi_gray = gray_image[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # detects eyes of within the detected face area (roi)
        eyes = eye_cascade.detectMultiScale(roi_gray, minNeighbors=3)
        nb_eyes = nb_eyes + len(eyes)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.4, 50, minSize=(int(img.shape[0]/30), int(img.shape[0]/30)))
        nb_smiles = nb_smiles + len(smiles)
        
        
    return len(face), nb_eyes, nb_smiles

def bgremove2(input_image):
    model_name = "u2net_human_seg"
    session = rembg.new_session(model_name)
    output_array = rembg.remove(
        input_image, session=session, alpha_matting=True, alpha_matting_foreground_threshold= 245, alpha_matting_background_threshold= 5,bgcolor=(245,245,245,255))
    return output_array

def resize_image(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grayscale image
    face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=9, minSize=(int(img.shape[0]/10), int(img.shape[0]/10))
    )
    
    
    for (x, y, w, h) in face:
        lowerylimit = y-h/2 if y-h/2>0 else 0
        upperylimit = y+h+h/2.5 if y+h+h/2.5<img.shape[0] else img.shape[0]
        lowerxlimit = x-w/4 if x-w/4>0 else 0
        upperxlimit = x+w+w/4 if x+w+w/4<img.shape[1] else img.shape[1]
        resized_image = img[int(lowerylimit):int(upperylimit), int(lowerxlimit):int(upperxlimit)]

   
    resimage = Image.fromarray(resized_image).resize((final_width,final_height))
    resimage = resimage.convert('RGB')
    resized_image = cv2.resize(resized_image, (final_width,final_height), interpolation=cv2.INTER_AREA)
    return resized_image


register_heif_opener()

app = Flask(__name__)

@app.route('/', methods=['POST'])
def check_picture():
    image_file = request.files['image_file']
    try:
        pil_image = Image.open(image_file).convert('RGB')
        imgarray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        if len(test_image(imgarray)) == 0:
            imgnobg = resize_image(np.array(bgremove2(pil_image)))
            # final_image = Image.fromarray(imgnobg).convert('RGB')
            # final_image.save('out.jpg','jpg')
            # image_file.file.close()
            # pil_image.close()


            final_image = Image.fromarray(imgnobg).convert('RGB')
            filtered_image = io.BytesIO()
            final_image.save(filtered_image,'JPEG')
            filtered_image.seek(0)
            #bytes_image = final_image.tobytes()
            return send_file(
                filtered_image,
                mimetype='image/jpg',
            )

            #converted_string = base64.b64encode(bytes_image)
            #return bytes_image[:10]

            # _, im_bytes_np = cv2.imencode('.jpg',imgnobg)
            # bytes_str = im_bytes_np.tobytes()
            # response = make_response(bytes_str)
            # response.headers.set('Content-Type', 'image/jpg')
        else:
            return test_image(imgarray)
    except:
         raise Exception()


    



