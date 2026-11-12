from openai import OpenAI
client = OpenAI()
from flask import Flask, request, jsonify, send_from_directory
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
from flask_cors import CORS  

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = models.resnet18(weights=None) 
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)  


model.load_state_dict(torch.load('leaf_disease_model.pth', map_location=device, weights_only=True))  
model = model.to(device)
model.eval()  


preprocess = transforms.Compose([      
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image):
    img_tensor = preprocess(image).unsqueeze(0)  
    img_tensor = img_tensor.to(device)

    
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)

    
    class_names = ['Anthracnose', 'Bacterial_Blight', 'Cercospora_Leaf_Spot', 'Powdery_Mildew', 'Shot_Hole_Disease']  
    predicted_class = class_names[preds.item()]
    
    return predicted_class

@app.route('/ask-ai', methods=['POST'])
def ask_ai():
    try:
        data = request.json
        disease = data.get('disease')

        if not disease:
            return jsonify({'error': 'No disease provided'}), 400

        prompt = f"""
        Tell more about this disease and give solutions and mitigation methods to solve this issue:
        {disease}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message.content

        return jsonify({'response': answer})

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
         