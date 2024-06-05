#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
1.PyMuPDF is a high-performance Python library for data extraction, analysis, 
conversion & manipulation of PDF (and other) documents. PyMuPDF is hosted on GitHub and registered on PyPI.

2.Pytesseract or Python-tesseract is an OCR tool for python that also serves as a wrapper 
for the Tesseract-OCR Engine.
'''
pip install pdf2image PyMuPDF 
pip install pytesseract
pip install bert_score
pip install -U sentence-transformers
pip install openai
pip install Word2Vec


# # Extracting flowcharts from the patent pdfs

# In[ ]:


import fitz  #import PyMuPDF module
import os #module used to interact with local Operating System.

# locating the folder with the patent pdfs
files = os.listdir('/Users/abdulkareemarbaz/Documents/Responsible AI Lab/patent pdfs') 
folder_path = '/Users/abdulkareemarbaz/Documents/Responsible AI Lab/patent pdfs' # recording the path to folder

# for each file in folder get flowcharts from the pdfs
for file_name in files: 
    file_path = os.path.join(folder_path, file_name)
    pdf_document = fitz.open(file_path)
    
    # iterate through each page in pdf file
    for page_number in range(len(pdf_document)): 
        page = pdf_document.load_page(page_number)
        image_list = page.get_images(full=True)

        # iterate through the pages and extract images from them.
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref) # extracting images 
            image_data = base_image["image"]

            with open(f'/Users/abdulkareemarbaz/Documents/Responsible AI Lab/images1/page_{page_number}_img_{img_index}.jpg', 'wb') as f:
                
                # saving each image in the particular folder
                f.write(image_data) 

pdf_document.close() # Close the PDF file


# In[60]:


print(sorted(files))


# # Extracting images from the folder

# In[18]:


import os
from PIL import Image
import cv2

# extract the images from patent pdfs stored in this folder
files = os.listdir('/Users/abdulkareemarbaz/Documents/Responsible AI Lab/flowchart_images/1') 
folder_path = '/Users/abdulkareemarbaz/Documents/Responsible AI Lab/flowchart_images/1'

#store the images in the list
image_list = []
for file_name in sorted(files):
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path) and file_name.lower().endswith(('.png','.jpg')):
        
        #extracting images from the folder using imread from cv2
        image = cv2.imread(file_path)
        
        #dont images which are not None
        if type(image) is not(None):
            image_list.append(image)


# # Pre processing obtained images

# In[20]:


import cv2
image = cv2.imread("/Users/abdulkareemarbaz/Documents/Responsible AI Lab")


# In[19]:


import cv2
import numpy as np
from numpy import asarray
gray_images = []
binary_images = []

# pre processing the images to obtain a better image quality.
def preprocess(image):
    
    #converting image to grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #converting images to binary images i.e white and black 
    _, binary_image = cv2.threshold(gray, 150, 215, cv2.THRESH_BINARY)
    
    #resizing the images to get a bigger image
    image= cv2.resize(binary_image, None, fx=5, fy=5)
    
    #using kernel to go through a number of cells in an image
    kernel = np.ones((2, 2), np.uint8)
    
    #using morphology's dilation to remove noise in image
    image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    return image

#taking processed images in a list
processed_image_list = []

#iterate through
for x in image_list:
    if x is not(None):
        processed_image_list.append(preprocess(x))


# In[20]:


import re
pattern = re.compile(r'[0-9a-zA-Z!@#$%^&*()=+{}\[\]:;<>,.?/\\`]')


# # Extracting text 

# In[4]:


import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt

# Extracting text from the image data
def text_extract(image):
    
    #custom_config = r'--oem 3 --psm 11'
    text_regions = pytesseract.image_to_boxes(image)
    return text_regions
    
def text_remove(image,text_regions):
    img = image.copy()
    text_area = []
    
    # Process each text region
    for x in text_regions:
        if not x.isdigit() and x != ' ':
            if x!='\n':
                text_area.append(x)
    for region in text_regions.splitlines():
        
    # Extract the coordinates from the region string
        x, y, x2, y2 = map(int, region.split()[1:5])
        print(region[0])
        matches = pattern.findall(region[0])
        y = image.shape[0] - y
        y2 = image.shape[0] - y2
        print(x, x2,y , y2)
    # Remove the identified text region
        text_area.append([x,y,x2,y2])
        

        if x!=x2 and y!=y2 and len(matches)>0:
            img[y2:y, x:x2] = cv2.medianBlur(image[y2:y, x:x2],21)

    return img
images1 = []
images_text = []
for x in processed_image_list:
    images1.append(text_remove(x,text_extract(x)))
    images_text.append(text_extract(x))


# In[147]:


images_text = []
c = 0
for x in processed_image_list:
    c = c + 1
    print(c)
    config = r'--oem 3 --psm 3'
    #gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(x,config=config)
    images_text.append(text)


# # Importing SAM(Segment Anything Model)

# In[9]:


import os
HOME = os.getcwd()
print("HOME:", HOME)


# In[10]:


get_ipython().run_line_magic('cd', '{HOME}')

import sys
get_ipython().system("{sys.executable} -m pip install 'git+https://github.com/facebookresearch/segment-anything.git'")


# In[11]:


get_ipython().system('pip install -q jupyter_bbox_widget roboflow dataclasses-json supervision')


# In[12]:


get_ipython().run_line_magic('cd', '{HOME}')
get_ipython().system('mkdir {HOME}/weights')
get_ipython().run_line_magic('cd', '{HOME}/weights')

get_ipython().system('wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth')


# In[13]:


import os

CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))


# In[14]:


get_ipython().run_line_magic('cd', '{HOME}')
get_ipython().system('mkdir {HOME}/data')
get_ipython().run_line_magic('cd', '{HOME}/data')

get_ipython().system('wget -q https://media.roboflow.com/notebooks/examples/dog.jpeg')
get_ipython().system('wget -q https://media.roboflow.com/notebooks/examples/dog-2.jpeg')
get_ipython().system('wget -q https://media.roboflow.com/notebooks/examples/dog-3.jpeg')
get_ipython().system('wget -q https://media.roboflow.com/notebooks/examples/dog-4.jpeg')


# In[15]:


import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"


# In[16]:


from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)


# In[17]:


mask_generator = SamAutomaticMaskGenerator(sam)


# # Creating segments using SAM

# Either follow this step to obtain segments or follow the one below it to get the segments already obtained , since obtaining segments for all images would take time.

# In[ ]:


import cv2
import supervision as sv
import numpy as np

sam_result = []
    
# generate the segments using the sam model and store it's result 
for x in image_list: 
    sam_result.append(mask_generator.generate(x))


# In[88]:


import pandas as pd
from numpy import array
# Replace 'your_file.xlsx' with the path to your Excel file
file_path = '/Users/abdulkareemarbaz/Documents/Responsible AI Lab/Book2.xlsx'

# Read the Excel file into a Pandas DataFrame
df = pd.read_excel(file_path)

# Display the DataFrame the data frame consists data for description , text in images , segment , pytesseract text
# Since it would take a lot of time to get the SAM result , you can use the SAM result for the whole dataset which i 
#have obtained in the column 'segment'.

print("Extracted Data:")
print(df)
list_dicts = [] #save the segmented data in a list
for x in df['segment']:
    stri = ""
    dicts = []
    for a in x[1:]:
        if a == '\'':
            stri+='\"'
        elif a == '}':
            stri+=a
            if(stri[0] == ","):
                dicts.append(eval(stri[2:]))
            else:
                dicts.append(eval(stri))
            print(stri)
            stri = ""
        else:
            stri+=a
    list_dicts.append(dicts)


# In[169]:


import math
masks = []
boxes = []

#iterate through all the images data in sam_result
    
    # store the segmentation data onbtained from SAM model and sort it according to the point coordinates
for x in list_dicts:
    masks.append( [
        mask['segmentation']
        for mask
        in sorted(x, key=lambda x: x['point_coords'][0][1] if x['area']>10000 else False
            )
    ])
    
    # store the bounding box data onbtained from SAM model and sort it according to the point coordinates
    boxes.append( [
        mask['bbox']
        for mask
        in sorted(x, key=lambda x: x['point_coords'][0][1] if x['area']>10000 else False
                )
    ])


# In[170]:


words = []
c = 0
# iterate through each segment and store the text from those images.
for box,image in zip(boxes,image_list):
    c = c + 1
    word = []
    # going through each bounding box
    width,height,color = image.shape
    print(height,width,color)
    for a in box:
        #if x == a[0] and y == a[1]:
        #    continue
        x , y ,w , h = a
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        #print(x , y , w , h)
        if x>width:
            x = width-1
        if y>width:
            y = width-1
       # matches = pattern.findall(a)
        if x!=w and y!=h :
        
            #print(x , y , w , h)
        
            #taking segment of the image
            if(x+w > height):
                continue
            if(y+h > width):
                continue
        
            image_segment = image[y:y+h, x:x+w]
            #extracting text from the image segment
            boxes_data = pytesseract.image_to_string(image_segment)
            plt.imshow(image_segment)
            word.append(boxes_data)
    words.append(word)
    print(c)


# # Using ChatGPT to make sense of the data obtained

# # Evaluating ChatGPT Enhanced text

# In[118]:


prompts1 = []
for x in words:
    prompts1.append(str(x))


# In[120]:


prec = []
rec = []
f1_s = []


# In[121]:


from bert_score import score

# Example sentences
for x,y in zip(df['TEXT IN IMAGES'],prompts):
    reference = [x]
    candidate = [y]

# Calculate BERTScore
    P, R, F1 = score(candidate, reference, lang="en")

# Print results
    for ref, can, p, r, f1 in zip(reference, candidate, P, R, F1):
        prec.append(p)
        rec.append(r)
        f1_s.append(f1)


# In[123]:


print(sum(prec)/549)
print(sum(rec)/549)
print(sum(f1_s)/549)


# In[124]:


prec_gpt = []
rec_gpt = []
f1_s_gpt = []


# In[125]:


prompts2 = []
for x in words:
    prompts2.append("correct the text detected by pytesseract:"+str(x))


# In[126]:


import openai
openai.api_key = '30cf41f3e3cd43ccbb56b8eb3f2f89a5' 
openai.api_base = "https://arbaz.openai.azure.com/"  # Replace with your Azure endpoint URL
openai.api_type = 'azure'
openai.api_version = '2023-08-01-preview'
deployment_name = 'arbaz-model'  # Replace with your deployment name


# In[127]:


c = 0
answer_gpt_pytesseract = []
for x in prompts:
    c = c+1
    print(c)
    response = openai.ChatCompletion.create(engine=deployment_name, messages=[{"role": "system", "content": x}])
    answer_gpt_pytesseract.append(response['choices'][0]['message']['content'].replace('\n', '').replace(' .', '.').strip())


# In[128]:


from bert_score import score

# Example sentences
for x,y in zip(df['TEXT IN IMAGES'],answer_gpt_pytesseract):
    reference = [x]
    candidate = [y]

# Calculate BERTScore
    P, R, F1 = score(candidate, reference, lang="en")

# Print results
    for ref, can, p, r, f1 in zip(reference, candidate, P, R, F1):
        prec_gpt.append(p)
        rec_gpt.append(r)
        f1_s_gpt.append(f1)


# In[130]:


print(sum(prec_gpt)/549)
print(sum(rec_gpt)/549)
print(sum(f1_s_gpt)/549)


# # Evaluating description of Flowcharts:

# In[123]:


prompts = []

for x,y in zip(words,boxes):
    prompts.append("Give a detailed and descriptive interpretation of the flowchart in the form of steps using following details pytesseract text recognition data:"+str(x)+"bounding box info obtained from sam(segment anything model):"+str(y))


# In[124]:


import openai
openai.api_key = '30cf41f3e3cd43ccbb56b8eb3f2f89a5' 
openai.api_base = "https://arbaz.openai.azure.com/"  # Replace with your Azure endpoint URL
openai.api_type = 'azure'
openai.api_version = '2023-08-01-preview'
deployment_name = 'arbaz-model'  # Replace with your deployment name


# In[199]:


c = 0
answer1 = []
for x in prompts:
    c = c+1
    print(c)
    response = openai.ChatCompletion.create(engine=deployment_name, messages=[{"role": "system", "content": x}],max_tokens=500)
    answer1.append(response['choices'][0]['message']['content'].replace('\n', '').replace(' .', '.').strip())


# In[201]:


zero_shot_prec = []
zero_shot_rec = []
zero_shot_f1_s = []


# In[202]:


from bert_score import score

# Example sentences
for x,y in zip(df['DESCRIPTION'],answer1):
    reference = [x]
    candidate = [y]

# Calculate BERTScore
    P, R, F1 = score(candidate, reference, lang="en")

# Print results
    for ref, can, p, r, f1 in zip(reference, candidate, P, R, F1):
        zero_shot_prec.append(p)
        zero_shot_rec.append(r)
        zero_shot_f1_s.append(f1)


# In[258]:


print(sum(zero_shot_prec)/549)
print(sum(zero_shot_rec)/549)
print(sum(zero_shot_f1_s)/549)


# In[171]:


input1 = "Give a detailed and descriptive interpretation of the flowchart in the form of steps using following details pytesseract text recognition data: bounding box info obtained from sam(segment anything model):[[1191, 835, 51, 18], [850, 668, 60, 23], [583, 262, 52, 17], [514, 262, 62, 18], [1345, 835, 43, 18], [1230, 655, 49, 17], [475, 235, 54, 18], [1284, 656, 66, 16], [151, 237, 14, 16], [1286, 540, 14, 65], [84, 10, 264, 175], [84, 10, 264, 121], [85, 11, 262, 510], [84, 10, 263, 260], [0, 0, 1451, 949], [442, 192, 265, 132], [535, 238, 89, 15], [88, 189, 255, 139], [441, 407, 264, 131], [88, 407, 617, 131], [798, 407, 264, 132], [84, 407, 264, 131], [1155, 407, 275, 300], [1155, 407, 275, 132], [797, 611, 617, 132], [797, 612, 264, 132], [1165, 607, 256, 141], [1150, 805, 286, 131]]the text in each of those bounding boxes['', '', '', '', '', '', '', '', '', '', 'Customer places\\n\\nan order\\n', 'Yes\\n', '', 'Customer places\\nan order\\n\\nIs item still in Email customer and\\nstock? cancel order\\n\\nEmail customer with\\nHand off to carrier shipping confirmation\\nand tracking info\\n\\nDoes carrier\\n\\nNotify customer\\n\\ndeliver item?\\n\\nEmail customer with\\n\\nconfirmation of delivery\\nand return instructions\\n', 'Email customer and_\\ncancel order\\n', '', '', 'Print label\\n', 'Pack item Print label\\n', 'Hand off to carrier\\n', 'Pack item\\n', 'Does carrier\\ndeliver item?\\n', '<—No—\\n', '', '', 'Email customer with\\nconfirmation of delivery\\n\\nCoes return ——\\n']"


# In[172]:


knowledge1 = '''1.Customer places an order: When a customer initiates an order, this is the starting point of the process. 
2.Is the item still in stock?: A decision box where we check if the item is still in stock if it is pack the item else email the customer and cancel the order
3. Email customer and cancel order: If the item is no longer in stock, this step involves notifying the customer and canceling the order. 
4.Pack item: Once we pack the item , we print the label on the item.
5.Print label: Once we label is printed , we move to the next step handoff to carrier
6.Handoff to carrier: The item is handed over to the carrier for delivery 
7.Email customer with shipping confirmation and tracking info: Assuming the item is in stock, this step involves informing the customer about the shipment with tracking details. 7.Does the carrier deliver the item?: A branching point where the process checks if the carrier delivers the item, if it does, Email customer with confirmation of delivery and return instructions , else notify the customer and cancel the order.
'''


# In[173]:


input2 = "Give a detailed and descriptive interpretation of the flowchart in the form of steps using following details pytesseract text recognition data: bounding box info obtained from sam(segment anything model):[[42, 164, 17, 6], [27, 89, 32, 8], [95, 158, 14, 6], [166, 169, 17, 6], [129, 83, 4, 5], [59, 205, 10, 6], [160, 83, 19, 7], [129, 169, 33, 8], [26, 176, 28, 6], [46, 218, 10, 10], [143, 84, 3, 7], [58, 120, 14, 6], [97, 72, 11, 6], [104, 73, 4, 5], [46, 131, 10, 11], [56, 164, 3, 6], [12, 14, 3, 6], [66, 207, 3, 4], [111, 168, 11, 10], [58, 176, 17, 6], [129, 83, 17, 8], [111, 82, 10, 10], [46, 203, 10, 26], [39, 77, 23, 8], [12, 14, 22, 8], [46, 45, 10, 11], [162, 84, 4, 5], [68, 121, 4, 5], [97, 72, 5, 5], [42, 164, 4, 6], [59, 205, 5, 6], [96, 81, 25, 11], [96, 86, 14, 1], [46, 31, 10, 25], [51, 31, 0, 14], [46, 117, 10, 26], [51, 117, 0, 14], [51, 203, 0, 14], [51, 31, 0, 14], [51, 117, 0, 14], [95, 158, 5, 6], [57, 238, 19, 8], [51, 203, 0, 14], [63, 121, 9, 5], [8, 5, 86, 25], [8, 5, 86, 40], [8, 5, 86, 110], [122, 74, 64, 25], [6, 57, 89, 59], [6, 45, 105, 86], [6, 57, 180, 59], [0, 0, 191, 261], [124, 161, 64, 24], [6, 144, 89, 58], [6, 144, 89, 84], [19, 173, 78, 82], [19, 230, 64, 25], [19, 218, 64, 37]]the text in each of those bounding boxes['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '[Lamp doesnt work\\n', '', '', '', 'Lamp\\nplunged in?\\n', '', 'Replace bulb\\n', '', 'burned out?\\n\\nRepair lamp\\n', '', '']"


# In[174]:


knowledge2 = '''1.Lamp Doesn't Work: In this state , the lamp is not working and move on to check if lamp is plugged in
2. Lamp Plugged in?: A decision point. It checks whether the lamp is plugged in. If it is not then plugin lamp , else check if bulb burned out.
3.Bulb burned out :Check if bulb is burned out , if yes replace bulb else repair lamp
'''


# In[178]:


prompts = []

for x,y in zip(words,boxes):
    prompts.append("Give a detailed and descriptive interpretation of the flowchart in the form of steps using following details pytesseract text recognition data:"+str(x)+"bounding box info obtained from sam(segment anything model):"+str(y))


# In[176]:


import openai
openai.api_key = '30cf41f3e3cd43ccbb56b8eb3f2f89a5' 
openai.api_base = "https://arbaz.openai.azure.com/"  # Replace with your Azure endpoint URL
openai.api_type = 'azure'
openai.api_version = '2023-08-01-preview'
deployment_name = 'arbaz-model'  # Replace with your deployment name


# In[204]:


c = 0
answer2 = []
for x in prompts:
    c = c+1
    print(c)
    response = openai.ChatCompletion.create(engine=deployment_name, messages=[{"role": "system", "content": "You describe flowcharts."},{"role": "user", "content":input1},{"role": "assistant", "content":knowledge1},{"role": "user", "content":input2},{"role": "assistant", "content":knowledge2},{"role": "user", "content":x}],max_tokens=1000)
    answer2.append(response['choices'][0]['message']['content'].replace('\n', '').replace(' .', '.').strip())


# In[57]:


len(answer2)


# In[205]:


instruction_based_prec = []
instruction_based_rec = []
instruction_based_f1_s = []


# In[206]:


from bert_score import score

# Example sentences
for x,y in zip(df['DESCRIPTION'],answer2):
    reference = [x]
    candidate = [y]

# Calculate BERTScore
    P, R, F1 = score(candidate, reference, lang="en")

# Print results
    for ref, can, p, r, f1 in zip(reference, candidate, P, R, F1):
        instruction_based_prec.append(p)
        instruction_based_rec.append(r)
        instruction_based_f1_s.append(f1)


# In[257]:


print(tensor(0.8554))
print(tensor(0.8548))
print(tensor(0.8549))


# In[208]:


prompts = []

for x in images_text:
    prompts.append("Give a detailed and descriptive interpretation of the flowchart in the form of steps using following details pytesseract text recognition data:"+str(x))


# In[209]:


import openai
openai.api_key = '30cf41f3e3cd43ccbb56b8eb3f2f89a5' 
openai.api_base = "https://arbaz.openai.azure.com/"  # Replace with your Azure endpoint URL
openai.api_type = 'azure'
openai.api_version = '2023-08-01-preview'
deployment_name = 'arbaz-model'  # Replace with your deployment name


# In[252]:


c = 0
answer3 = []
for x in prompts:
    c = c+1
    print(c)
    response = openai.ChatCompletion.create(engine=deployment_name, messages=[{"role": "system", "content": x}],max_tokens=500)
    answer3.append(response['choices'][0]['message']['content'].replace('\n', '').replace(' .', '.').strip())


# In[254]:


zero_shot_prec_ablation = []
zero_shot_rec_ablation = []
zero_shot_f1_s_ablation = []


# In[255]:


from bert_score import score

# Example sentences
for x,y in zip(df['DESCRIPTION'],answer3):
    reference = [x]
    candidate = [y]

# Calculate BERTScore
    P, R, F1 = score(candidate, reference, lang="en")

# Print results
    for ref, can, p, r, f1 in zip(reference, candidate, P, R, F1):
        zero_shot_prec_ablation.append(p)
        zero_shot_rec_ablation.append(r)
        zero_shot_f1_s_ablation.append(f1)


# In[259]:


print(sum(zero_shot_prec_ablation)/549)
print(sum(zero_shot_rec_ablation)/549)
print(sum(zero_shot_f1_s_ablation)/549)


# In[230]:


input1_ablation = "Give a detailed and descriptive interpretation of the flowchart in the form of steps using following details pytesseract text recognition data: [Lamp doesn't work]\n\nLamp\n\nplugged in? >) Plug in lamp |\n\nBulb\nburned out?\n\n—J)\n\nReplace bulb\n\n[ Repair lamp |\n\n"


# In[231]:


knowledge1_ablation = '''1.Customer places an order: When a customer initiates an order, this is the starting point of the process. 
2.Is the item still in stock?: A decision box where we check if the item is still in stock if it is pack the item else email the customer and cancel the order
3. Email customer and cancel order: If the item is no longer in stock, this step involves notifying the customer and canceling the order. 
4.Pack item: Once we pack the item , we print the label on the item.
5.Print label: Once we label is printed , we move to the next step handoff to carrier
6.Handoff to carrier: The item is handed over to the carrier for delivery 
7.Email customer with shipping confirmation and tracking info: Assuming the item is in stock, this step involves informing the customer about the shipment with tracking details. 7.Does the carrier deliver the item?: A branching point where the process checks if the carrier delivers the item, if it does, Email customer with confirmation of delivery and return instructions , else notify the customer and cancel the order.
'''


# In[232]:


input2_ablation = "Give a detailed and descriptive interpretation of the flowchart in the form of steps using following details pytesseract text recognition data: Customer places\n\nEmail customer and\ncan\n\ncel order\n\nEmail customer with\nHand off to carrier shipping confirmation\nand tracking info\n\n’ Does carrier\nNotify customer deliver item?\n\nEmail customer with\n\nai\nconfirmation of delivery\nand return instructions\n"


# In[233]:


knowledge2_ablation = '''1.Lamp Doesn't Work: In this state , the lamp is not working and move on to check if lamp is plugged in
2. Lamp Plugged in?: A decision point. It checks whether the lamp is plugged in. If it is not then plugin lamp , else check if bulb burned out.
3.Bulb burned out :Check if bulb is burned out , if yes replace bulb else repair lamp
'''


# In[220]:


prompts = []

for x in images_text:
    prompts.append("Give a detailed and descriptive interpretation of the flowchart in the form of steps using following details pytesseract text recognition data:"+str(x))


# In[221]:


import openai
openai.api_key = '30cf41f3e3cd43ccbb56b8eb3f2f89a5' 
openai.api_base = "https://arbaz.openai.azure.com/"  # Replace with your Azure endpoint URL
openai.api_type = 'azure'
openai.api_version = '2023-08-01-preview'
deployment_name = 'arbaz-model'  # Replace with your deployment name


# In[234]:


c = 0
answer4 = []
for x in prompts:
    c = c+1
    print(c)
    response = openai.ChatCompletion.create(engine=deployment_name, messages=[{"role": "system", "content": "You describe flowcharts."},{"role": "user", "content":input1_ablation},{"role": "assistant", "content":knowledge1_ablation},{"role": "user", "content":input2_ablation},{"role": "assistant", "content":knowledge2_ablation},{"role": "user", "content":x}],max_tokens=1000)
    answer4.append(response['choices'][0]['message']['content'].replace('\n', '').replace(' .', '.').strip())


# In[235]:


instruction_based_prec_ablation = []
instruction_based_rec_ablation = []
instruction_based_f1_s_ablation = []


# In[236]:


from bert_score import score

# Example sentences
for x,y in zip(df['DESCRIPTION'],answer4):
    reference = [x]
    candidate = [y]

# Calculate BERTScore
    P, R, F1 = score(candidate, reference, lang="en")

# Print results
    for ref, can, p, r, f1 in zip(reference, candidate, P, R, F1):
        instruction_based_prec_ablation.append(p)
        instruction_based_rec_ablation.append(r)
        instruction_based_f1_s_ablation.append(f1)


# In[261]:


print(sum(instruction_based_prec_ablation)/549)
print(sum(instruction_based_rec_ablation)/549)
print(sum(instruction_based_f1_s_ablation)/549)


# In[135]:


from gensim.models import KeyedVectors
import string

# Load pre-trained Word2Vec model (example using Google's Word2Vec model)
model_path = '/Users/abdulkareemarbaz/Documents/Responsible AI Lab/word2vec-google-news-300.model'
word2vec_model = KeyedVectors.load(model_path)

# Tokenize and preprocess sentences
similarity1 = []
# Remove punctuation from tokens
for x,y in zip(df['DESCRIPTION'],df['zero_shot_prompting']):
    translator = str.maketrans("", "", string.punctuation)
    tokens1 = x.lower().translate(translator).split()
    tokens2 = y.lower().translate(translator).split()

    # Check if tokens are present in the model's vocabulary
    tokens1 = [token for token in tokens1 if token in word2vec_model.key_to_index]
    tokens2 = [token for token in tokens2 if token in word2vec_model.key_to_index]

    # Calculate similarity between two sentences
    similarity_score = word2vec_model.n_similarity(tokens1, tokens2)
    print(f"Similarity Score: {similarity_score}")
    similarity1.append(similarity_score)
    


# In[136]:


sum(similarity1)/549


# In[ ]:


from gensim.models import KeyedVectors
import string

# Load pre-trained Word2Vec model (example using Google's Word2Vec model)
model_path = '/Users/abdulkareemarbaz/Documents/Responsible AI Lab/word2vec-google-news-300.model'
word2vec_model = KeyedVectors.load(model_path)

# Tokenize and preprocess sentences
sentence1 = "This is the first sentence."
sentence2 = "My name is Arbaz"
similarity2 = []
# Remove punctuation from tokens
for x,y in zip(df['DESCRIPTION'],df['instruction_based_prompting:']):
    translator = str.maketrans("", "", string.punctuation)
    tokens1 = x.lower().translate(translator).split()
    tokens2 = y.lower().translate(translator).split()

    # Check if tokens are present in the model's vocabulary
    tokens1 = [token for token in tokens1 if token in word2vec_model.key_to_index]
    tokens2 = [token for token in tokens2 if token in word2vec_model.key_to_index]

    # Calculate similarity between two sentences
    similarity_score = word2vec_model.n_similarity(tokens1, tokens2)
    print(f"Similarity Score: {similarity_score}")
    similarity2.append(similarity_score)
    


# In[138]:


sum(similarity2)/549


# In[139]:


from gensim.models import KeyedVectors
import string

# Load pre-trained Word2Vec model (example using Google's Word2Vec model)
model_path = '/Users/abdulkareemarbaz/Documents/Responsible AI Lab/word2vec-google-news-300.model'
word2vec_model = KeyedVectors.load(model_path)

# Tokenize and preprocess sentences
sentence1 = "This is the first sentence."
sentence2 = "My name is Arbaz"
similarity3 = []
# Remove punctuation from tokens
for x,y in zip(df['DESCRIPTION'],df['zero_shot_prompting_ablation']):
    translator = str.maketrans("", "", string.punctuation)
    tokens1 = x.lower().translate(translator).split()
    tokens2 = y.lower().translate(translator).split()

    # Check if tokens are present in the model's vocabulary
    tokens1 = [token for token in tokens1 if token in word2vec_model.key_to_index]
    tokens2 = [token for token in tokens2 if token in word2vec_model.key_to_index]

    # Calculate similarity between two sentences
    similarity_score = word2vec_model.n_similarity(tokens1, tokens2)
    print(f"Similarity Score: {similarity_score}")
    similarity3.append(similarity_score)
    


# In[140]:


sum(similarity3)/549


# In[141]:


from gensim.models import KeyedVectors
import string

# Load pre-trained Word2Vec model (example using Google's Word2Vec model)
model_path = '/Users/abdulkareemarbaz/Documents/Responsible AI Lab/word2vec-google-news-300.model'
word2vec_model = KeyedVectors.load(model_path)

# Tokenize and preprocess sentences
sentence1 = "This is the first sentence."
sentence2 = "My name is Arbaz"
similarity4 = []
# Remove punctuation from tokens
for x,y in zip(df['DESCRIPTION'],df['instruction_based_prompting_ablation']):
    translator = str.maketrans("", "", string.punctuation)
    tokens1 = x.lower().translate(translator).split()
    tokens2 = y.lower().translate(translator).split()

    # Check if tokens are present in the model's vocabulary
    tokens1 = [token for token in tokens1 if token in word2vec_model.key_to_index]
    tokens2 = [token for token in tokens2 if token in word2vec_model.key_to_index]

    # Calculate similarity between two sentences
    similarity_score = word2vec_model.n_similarity(tokens1, tokens2)
    print(f"Similarity Score: {similarity_score}")
    similarity4.append(similarity_score)
    


# In[142]:


sum(similarity4)/549


# In[148]:


from sentence_transformers import SentenceTransformer, util

# Load a pre-trained SBERT model
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Define your sentences
sentence1 = "This is the first sentence."
sentence2 = "This is the first sentence."

# Encode the sentences using SBERT
s1 = []
for x,y in zip(df['DESCRIPTION'],df['zero_shot_prompting']):
    embedding1 = sbert_model.encode([x], convert_to_tensor=True)
    embedding2 = sbert_model.encode([y], convert_to_tensor=True)

    # Calculate cosine similarity between the embeddings
    cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2)[0][0].item()
    print(f"Cosine Similarity: {cosine_similarity}")
    s1.append(cosine_similarity)


# In[150]:


sum(s1)/549


# In[151]:


from sentence_transformers import SentenceTransformer, util

# Load a pre-trained SBERT model
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Define your sentences
sentence1 = "This is the first sentence."
sentence2 = "This is the first sentence."

# Encode the sentences using SBERT
s2 = []
for x,y in zip(df['DESCRIPTION'],df['instruction_based_prompting:']):
    embedding1 = sbert_model.encode([x], convert_to_tensor=True)
    embedding2 = sbert_model.encode([y], convert_to_tensor=True)

    # Calculate cosine similarity between the embeddings
    cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2)[0][0].item()
    print(f"Cosine Similarity: {cosine_similarity}")
    s2.append(cosine_similarity)


# In[152]:


sum(s2)/549


# In[153]:


from sentence_transformers import SentenceTransformer, util

# Load a pre-trained SBERT model
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Define your sentences
sentence1 = "This is the first sentence."
sentence2 = "This is the first sentence."

# Encode the sentences using SBERT
s3 = []
for x,y in zip(df['DESCRIPTION'],df['zero_shot_prompting_ablation']):
    embedding1 = sbert_model.encode([x], convert_to_tensor=True)
    embedding2 = sbert_model.encode([y], convert_to_tensor=True)

    # Calculate cosine similarity between the embeddings
    cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2)[0][0].item()
    print(f"Cosine Similarity: {cosine_similarity}")
    s3.append(cosine_similarity)


# In[154]:


sum(s3)/549


# In[155]:


from sentence_transformers import SentenceTransformer, util

# Load a pre-trained SBERT model
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Define your sentences
sentence1 = "This is the first sentence."
sentence2 = "This is the first sentence."

# Encode the sentences using SBERT
s4 = []
for x,y in zip(df['DESCRIPTION'],df['instruction_based_prompting_ablation']):
    embedding1 = sbert_model.encode([x], convert_to_tensor=True)
    embedding2 = sbert_model.encode([y], convert_to_tensor=True)

    # Calculate cosine similarity between the embeddings
    cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2)[0][0].item()
    print(f"Cosine Similarity: {cosine_similarity}")
    s4.append(cosine_similarity)


# In[156]:


sum(s4)/549


# In[19]:


from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
c = 0
instruction_based = []
for x,y in zip(df['DESCRIPTION'] ,df['instruction_based_prompting:']):
    sentences = [x,y]

#Compute embedding for both lists
    embedding_1= model.encode(sentences[0], convert_to_tensor=True)
    embedding_2 = model.encode(sentences[1], convert_to_tensor=True)

    instruction_based.append(util.pytorch_cos_sim(embedding_1, embedding_2))
    c = c+1
    print(c)
## tensor([[0.6003]])


# In[20]:


sum(instruction_based)/549


# In[23]:


from sentence_transformers import SentenceTransformer, util

zero_based_a = []
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
c = 0 
for x,y in zip(df['DESCRIPTION'] ,df['zero_shot_prompting_ablation'] ):
    sentences = [x,y]

#Compute embedding for both lists
    embedding_1= model.encode(sentences[0], convert_to_tensor=True)
    embedding_2 = model.encode(sentences[1], convert_to_tensor=True)

    zero_based_a.append(util.pytorch_cos_sim(embedding_1, embedding_2))
    c = c+1
    print(c)
## tensor([[0.6003]])


# In[24]:


sum(zero_based_a)/549


# In[25]:


from sentence_transformers import SentenceTransformer, util

instruction_based_a = []
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
c = 0
for x,y in zip(df['DESCRIPTION'] ,df['instruction_based_prompting_ablation']):
    sentences = [x,y]
#Compute embedding for both lists
    embedding_1= model.encode(sentences[0], convert_to_tensor=True)
    embedding_2 = model.encode(sentences[1], convert_to_tensor=True)

    instruction_based_a.append(util.pytorch_cos_sim(embedding_1, embedding_2))
    c = c+1
    print(c)
    
## tensor([[0.6003]])


# In[26]:


sum(instruction_based_a)/549


# In[64]:


from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_sentence_embedding(sentence, word_vectors):
    words = sentence.split()
    embeddings = [word_vectors[word] for word in words if word in word_vectors]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(word_vectors.vector_size)
# Example sentences
model_path = "/Users/abdulkareemarbaz/Documents/Responsible AI Lab/word2vec-google-news-300.model"
word_vectors = KeyedVectors.load(model_path)
l = []
c = 0
for x,y in zip(df['DESCRIPTION'] ,df['instruction_based_prompting_ablation']):
    sentence1 = x
    sentence2 = y

# Generate word embeddings for each word in the sentences

# Get sentence embeddings
    embedding1 = get_sentence_embedding(sentence1, word_vectors)
    embedding2 = get_sentence_embedding(sentence2, word_vectors)

# Calculate cosine similarity between sentence embeddings
    embedding_similarity_score = cosine_similarity([embedding1], [embedding2])[0][0]

# Calculate structural similarity score (e.g., cosine similarity between sentence vectors)
    structural_similarity_score = cosine_similarity([embedding1], [embedding2])[0][0]

# Combine the scores (you can adjust the weights based on your requirements)
    weight_embeddings = 0.5
    weight_structural_similarity = 0.5

    combined_score = (weight_embeddings * embedding_similarity_score) + (weight_structural_similarity * structural_similarity_score)
    l.append(combined_score)
    c=c+1
    print(c, combined_score)


# In[65]:


sum(l)/549


# In[48]:


from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_sentence_embedding(sentence, word_vectors):
    words = sentence.split()
    embeddings = [word_vectors[word] for word in words if word in word_vectors]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(word_vectors.vector_size)
# Example sentences
model_path = "/Users/abdulkareemarbaz/Documents/Responsible AI Lab/word2vec-google-news-300.model"
word_vectors = KeyedVectors.load(model_path)
l = []
c = 0
for x,y in zip(df['DESCRIPTION'] ,df['zero_shot_prompting_ablation'] ):
    sentence1 = x
    sentence2 = y

# Generate word embeddings for each word in the sentences

# Get sentence embeddings
    embedding1 = get_sentence_embedding(sentence1, word_vectors)
    embedding2 = get_sentence_embedding(sentence2, word_vectors)

# Calculate cosine similarity between sentence embeddings
    embedding_similarity_score = cosine_similarity([embedding1], [embedding2])[0][0]

# Calculate structural similarity score (e.g., cosine similarity between sentence vectors)
    structural_similarity_score = cosine_similarity([embedding1], [embedding2])[0][0]

# Combine the scores (you can adjust the weights based on your requirements)
    weight_embeddings = 0.6
    weight_structural_similarity = 0.4

    combined_score = (weight_embeddings * embedding_similarity_score) + (weight_structural_similarity * structural_similarity_score)
    l.append(combined_score)
    c=c+1
    print(c, combined_score)


# In[49]:


sum(l)/549


# In[62]:


from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_sentence_embedding(sentence, word_vectors):
    words = sentence.split()
    embeddings = [word_vectors[word] for word in words if word in word_vectors]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(word_vectors.vector_size)
# Example sentences
model_path = "/Users/abdulkareemarbaz/Documents/Responsible AI Lab/word2vec-google-news-300.model"
word_vectors = KeyedVectors.load(model_path)
l = []
c = 0
for x,y in zip(df['DESCRIPTION'] ,df['instruction_based_prompting:']):
    sentence1 = x
    sentence2 = y

# Generate word embeddings for each word in the sentences

# Get sentence embeddings
    embedding1 = get_sentence_embedding(sentence1, word_vectors)
    embedding2 = get_sentence_embedding(sentence2, word_vectors)

# Calculate cosine similarity between sentence embeddings
    embedding_similarity_score = cosine_similarity([embedding1], [embedding2])[0][0]

# Calculate structural similarity score (e.g., cosine similarity between sentence vectors)
    structural_similarity_score = cosine_similarity([embedding1], [embedding2])[0][0]

# Combine the scores (you can adjust the weights based on your requirements)
    weight_embeddings = 0.95
    weight_structural_similarity = 0.05

    combined_score = (weight_embeddings * embedding_similarity_score) + (weight_structural_similarity * structural_similarity_score)
    l.append(combined_score)
    c=c+1
    print(c, combined_score)


# In[63]:


sum(l)/549


# In[52]:


from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_sentence_embedding(sentence, word_vectors):
    words = sentence.split()
    embeddings = [word_vectors[word] for word in words if word in word_vectors]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(word_vectors.vector_size)
# Example sentences
model_path = "/Users/abdulkareemarbaz/Documents/Responsible AI Lab/word2vec-google-news-300.model"
word_vectors = KeyedVectors.load(model_path)
l = []
c = 0
for x,y in zip(df['DESCRIPTION'] ,df['instruction_based_prompting:']):
    sentence1 = x
    sentence2 = y

# Generate word embeddings for each word in the sentences

# Get sentence embeddings
    embedding1 = get_sentence_embedding(sentence1, word_vectors)
    embedding2 = get_sentence_embedding(sentence2, word_vectors)

# Calculate cosine similarity between sentence embeddings
    embedding_similarity_score = cosine_similarity([embedding1], [embedding2])[0][0]

# Calculate structural similarity score (e.g., cosine similarity between sentence vectors)
    structural_similarity_score = cosine_similarity([embedding1], [embedding2])[0][0]

# Combine the scores (you can adjust the weights based on your requirements)
    weight_embeddings = 0.6
    weight_structural_similarity = 0.4

    combined_score = (weight_embeddings * embedding_similarity_score) + (weight_structural_similarity * structural_similarity_score)
    l.append(combined_score)
    c=c+1
    print(c, combined_score)


# In[53]:


sum(l)/549


# In[115]:


from sentence_transformers import SentenceTransformer, util
import numpy as np
import re  # Import the regex module for sentence splitting

# Load Sentence Transformers model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Function to tokenize a sentence
def tokenize(sentence):
    return sentence.split()

# Function to generate embeddings for a sentence
def generate_sentence_embedding1(sentence, word_vectors):
    words = sentence.split()
    embeddings = [word_vectors[word] for word in words if word in word_vectors]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(word_vectors.vector_size)
def generate_sentence_embedding(sentence, model):
    return model.encode(sentence, convert_to_tensor=True)

# Function to compute similarity score between two embeddings
def compute_similarity_score(embedding1, embedding2):
    return util.pytorch_cos_sim(embedding1, embedding2).item()

# Function to combine scores using weighted average
def combine_scores(word_embedding_score, structural_similarity_score, weights):
    return (weights[0] * word_embedding_score) + (weights[1] * structural_similarity_score)

def divide_sentence(sentence):
    # Split sentence based on punctuation marks
    sentence_parts = [part.strip() for part in re.split(r'[.!?]', sentence) if part.strip()]
    return sentence_parts

def generate_structural_embeddings(sentence_parts, word_embedding_model):
    structural_embeddings = []
    for part in sentence_parts:
        # Generate embeddings for each structural part
        structural_embeddings.append(generate_sentence_embedding(part, word_embedding_model).cpu().detach().numpy())
    return structural_embeddings

x = "I took a bus first then a train"
y = "I took a train first then a bus"
model_path = "/Users/abdulkareemarbaz/Documents/Responsible AI Lab/word2vec-google-news-300.model"
word_vectors = KeyedVectors.load(model_path)
embedding_model = word_vectors
embedding1 = generate_sentence_embedding1(x, embedding_model)
embedding2 = generate_sentence_embedding1(y, embedding_model)

    # Compute structural similarity (using previous implementation)
    # You can replace this with your own structural similarity calculation
structural_similarity_score = compute_similarity_score(embedding1, embedding2)

    # Phase 2: Generate structural embeddings for each sentence
embedding_model = model
sentence_parts1 = divide_sentence(x)
sentence_parts2 = divide_sentence(y)
structural_embeddings1 = generate_structural_embeddings(sentence_parts1, embedding_model)
structural_embeddings2 = generate_structural_embeddings(sentence_parts2, embedding_model)

    # Compute word-based embeddings
embedding1 = np.mean(structural_embeddings1, axis=0)
embedding2 = np.mean(structural_embeddings2, axis=0)

    # Compute word embedding similarity (using cosine similarity)
embedding_similarity_score = compute_similarity_score(embedding1, embedding2)

    # Phase 3: Combine scores
weights = [0.40, 0.60]  # Adjust weights based on experimentation
combined_score = combine_scores(embedding_similarity_score, structural_similarity_score, weights)
#list_res.append(combined_score)
print("Combined Similarity Score:", combined_score)


# In[103]:


sum(list_res)/549


# In[96]:


from sentence_transformers import SentenceTransformer, util

zero_based_a = []
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

sentences = ["I took a bus first then a train","I took a train first then a bus"]

#Compute embedding for both lists
embedding_1= model.encode(sentences[0], convert_to_tensor=True)
embedding_2 = model.encode(sentences[1], convert_to_tensor=True)

print(util.pytorch_cos_sim(embedding_1, embedding_2))


# In[116]:


from sentence_transformers import SentenceTransformer, util

zero_based_a = []
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

sentences = ["A cat is sitting on mat","A cat is not sitting on mat"]

#Compute embedding for both lists
embedding_1= model.encode(sentences[0], convert_to_tensor=True)
embedding_2 = model.encode(sentences[1], convert_to_tensor=True)

print(util.pytorch_cos_sim(embedding_1, embedding_2))


# In[87]:



import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 20})

# Data
grouped_bar_label = ['BERT-P', 'BERT-R', 'BERT-F1', 'Cosine\nword2vec', 'Cosine\nSent. Trans.']
zero_shot_with_sam_scores = np.array([0.8333, 0.8314, 0.8468, 0.6505, 0.7111])
instruction_with_sam_scores = np.array([0.8654, 0.8648, 0.8649, 0.7283, 0.7625])
zero_shot_no_sam_scores = np.array([0.8027, 0.8249, 0.8184, 0.6438, 0.6996])
instruction_no_sam_scores = np.array([0.8320, 0.8242, 0.8278, 0.6536, 0.7031])
legend_labels = ['Zero shot (w/ SAM)', 'Instruction-based (w/ SAM)', 'Zero shot (w/o SAM)', 'Instruction-based (w/o SAM)']

# Bar width
barWidth = 0.2

# Set position of bar on X axis
r1 = np.arange(len(grouped_bar_label))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
patterns = ['/', '*', 'x', 'o']

# Make the plot
fig, ax = plt.subplots(figsize=(10, 6))
plt.bar(r1, zero_shot_with_sam_scores, color='lightsalmon',edgecolor='black',linewidth=2, width=barWidth, label=legend_labels[0], hatch=patterns[0])
plt.bar(r2, instruction_with_sam_scores, width=barWidth, color='turquoise',edgecolor='black',linewidth=2, label=legend_labels[1], hatch=patterns[1])
plt.bar(r3, zero_shot_no_sam_scores,width=barWidth, color='lightskyblue',edgecolor='black',linewidth=2, label=legend_labels[2], hatch=patterns[2])
plt.bar(r4, instruction_no_sam_scores, width=barWidth, color='sandybrown',edgecolor='black',linewidth=2, label=legend_labels[3], hatch=patterns[3])

# Add xticks on the middle of the group bars
plt.xticks([r + barWidth*1.5 for r in range(len(grouped_bar_label))], grouped_bar_label)

# Create legend & Show graphic
ax.set_ylabel('Similarity score')
ax.legend(fontsize="15", loc='lower right')

# Layout adjustments
plt.tight_layout(pad=0)
plt.grid()
plt.ylim(0.4, 0.88)

# Save plot as PDF
plt.savefig('score_visualization_new.pdf')

plt.show()

