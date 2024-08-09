from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import base64
import openai
import os
import requests

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Enable CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to encode the image
def encode_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Load the image
        print("loading image: ", file.filename)
        image = Image.open(io.BytesIO(await file.read()))

        # Encode the image to base64
        base64_image = encode_image(image)

        # Prepare the payload for OpenAI API
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Identify the image and provide approximate calories for the food in the image in the JSON format: {'food': 'food_name', 'calories': 'calories'}. If no food is present, plrease return {'food': 'no food found', 'calories': 'n/a'}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ]
        }

        # Make the request to OpenAI API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}",
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        # Handle the response
        if response.status_code == 200:
            result = response.json().get("choices")[0].get("message").get("content")
            print("OpenAI API Response:", result)
            
            # Parse the result to remove any surrounding ```json and ```
            result_dict = eval(result.strip("```json").strip("```").strip())

            # Return the cleaned JSON response
            print("Result:", result_dict)
            return JSONResponse(content=result_dict)
        else:
            print("Error from OpenAI API:", response.status_code, response.text)
            return JSONResponse(content={"error": response.text}, status_code=response.status_code)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
