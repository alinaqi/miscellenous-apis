import zipfile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import base64
import openai
import os
import requests
import json

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
                        {"type": "text", "text": "Identify in the image and provide approximate calories for the food in the image in the JSON format: {'food': 'food_name', 'calories': 'calories'}. If no food is present, plrease return {'food': 'no food found', 'calories': 'n/a'}"},
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

# Function to encode the image
def encode_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

@app.post("/detect-garage/")
async def detect_garage(file: UploadFile = File(...), new_garage_file: UploadFile = File(...)):
    try:
        # Load the original image
        print("loading image: ", file.filename)
        original_image = Image.open(io.BytesIO(await file.read()))

        # Encode the original image to base64
        base64_image = encode_image(original_image)

        # Prepare the payload for OpenAI API
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", 
                         "text": 
                         "Identify in the image which is a garage image: the garage door and return its properties including coordinates on the image. Return all information in the JSON format: {'found': 'true', 'garage': {'x': <x coordinate of the garage door on image from top left corner>, 'y': <y coordinate of garage door on the image from top left corner>, 'width_pixels': <width in pixels of the garage door only and not the entire image>, 'height_pixels': <height in pixels of the garage door only and not the entire image>} }. If no garage door is present, please return {'found': 'false'}"},
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
            
            # Parse the result to remove any surrounding ```json and convert it to a dictionary
            result_dict_str = result.strip("```json").strip("```").strip()
            print("Result Dict Str:", result_dict_str)
            result_dict = json.loads(result_dict_str)

            # Return the cleaned JSON response
            print("Result:", result_dict)

            return JSONResponse(content=result_dict)    

            try:
                if result_dict.get("found") == "true":
                    print("Garage door found in the image!")
                    garage_data = result_dict["garage"]

                    print("Garage Data:", garage_data)
                    # Resize new garage image to the specified width and height
                    print("Resizing new garage image...")
                    # Load the new garage door image
                    print("loading new garage image: ", new_garage_file.filename)
                    new_garage_image = Image.open(io.BytesIO(await new_garage_file.read())) 
                    new_garage_resized = new_garage_image.resize(
                        (garage_data["width_pixels"], garage_data["height_pixels"])
                    )

                    # Copy the original image to preserve it before modifying
                    modified_image = original_image.copy()

                    # Paste the new garage image onto the copied image at the specified coordinates
                    print("Pasting new garage image onto original image on coordinates:", garage_data["x"], garage_data["y"])
                    modified_image.paste(new_garage_resized, (garage_data["x"], garage_data["y"]))

                    print("Garage door replaced successfully!")
                    print("Returning original and modified images...")

                    # Save both images to separate buffers
                    original_buffer = io.BytesIO()
                    original_image.save(original_buffer, format="JPEG")
                    original_buffer.seek(0)

                    modified_buffer = io.BytesIO()
                    modified_image.save(modified_buffer, format="JPEG")
                    modified_buffer.seek(0)

                    # Create a ZIP file containing both images
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                        zip_file.writestr("original_image.jpg", original_buffer.getvalue())
                        zip_file.writestr("modified_image.jpg", modified_buffer.getvalue())
                    zip_buffer.seek(0)

                    # Return the ZIP file as a streaming response
                    return StreamingResponse(zip_buffer, media_type="application/zip", headers={
                        "Content-Disposition": "attachment; filename=garage_images.zip"
                    })
                else:
                    return JSONResponse(content={"error": "No garage door found in the image"}, status_code=400)
            except Exception as e:
                print("Error replacing garage door:", str(e))
                return JSONResponse(content={"error": str(e)}, status_code=400)
        else:
            print("Error from OpenAI API:", response.status_code, response.text)
            return JSONResponse(content={"error": response.text}, status_code=response.status_code)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post("/replace-garage/")
async def replace_garage(original_file: UploadFile = File(...), new_garage_file: UploadFile = File(...)):
    try:
        # Load the original image
        print("loading original image: ", original_file.filename)
        original_image = Image.open(io.BytesIO(await original_file.read()))

        # Load the new garage door image
        print("loading new garage image: ", new_garage_file.filename)
        new_garage_image = Image.open(io.BytesIO(await new_garage_file.read()))

        # Encode the original image to base64
        base64_image = encode_image(original_image)

        # Prepare the payload for OpenAI API
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Identify the image which contains a garage door, identify the garage door and return its properties including coordinates on the image. Return all information in the JSON format: {'found': 'true', 'garage': {'x': <x coordinate of the garage door on image>, 'y': <y coordinate of garage door on the image>, 'width_pixels': <widget in pixels of the garage door>, 'height_pixels': <height in pixes of the garage door>} }. If no garage door is present, plrease return {'found': 'false'"},
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
            
            # Parse the JSON result
            result_dict = result.strip("```json").strip("```").strip()

            print("Result:", result_dict)

            try:
                if result_dict.get("found") == "true":
                    print("Garage door found in the image!")
                    garage_data = result_dict["garage"]

                    print("Garage Data:", garage_data)
                    # Resize new garage image to the specified width and height
                    print("Resizing new garage image...")
                    new_garage_resized = new_garage_image.resize(
                        (garage_data["width_pixels"], garage_data["height_pixels"])
                    )

                    # Paste the new garage image onto the original image at the specified coordinates
                    print("Pasting new garage image onto original image on coordinates:", garage_data["x"], garage_data["y"])
                    original_image.paste(new_garage_resized, (garage_data["x"], garage_data["y"]))

                    print("Garage door replaced successfully!")
                    print("Returning modified image...")
                    # Save the modified image to a buffer
                    buffer = io.BytesIO()
                    original_image.save(buffer, format="JPEG")
                    buffer.seek(0)

                    # Return the modified image as a JPEG
                    return StreamingResponse(buffer, media_type="image/jpeg")
                else:
                    return JSONResponse(content={"error": "No garage door found in the image"}, status_code=400)
            except Exception as e:
                print("Error replacing garage door:", str(e))
                return JSONResponse(content={"error": str(e)}, status_code=400)
        else:
            print("Error from OpenAI API:", response.status_code, response.text)
            return JSONResponse(content={"error": response.text}, status_code=response.status_code)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)