import zipfile
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import base64
import openai
import os
import requests
import json
import uuid
from exa_py import Exa


# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
exa = Exa(api_key="b7621e86-ef98-4d65-b429-f0f723131651")


app = FastAPI(title="Image recognition apis", description="APIs for image recognition using OpenAI and agents", version="0.1")

# In-memory storage for task results (use a database in production)
task_results = {}

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


prompt_for_products = """
"Identify in the image the product and return it in the JSON format: {'product': 'product_name', 'product_type': 'type_of_product', 'product_color': 'color of product', 'price_category':'price category of product'}. If no product is present, please return {'product': 'no product found'}"}"
"""
@app.post("/find-similar-products/")
async def find_similar_products(file: UploadFile = File(...)):
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
                        {"type": "text", "text": prompt_for_products},
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


            result = exa.search_and_contents(
                f"Find similar products as given in the following json: {result_dict}",
                type="auto",
                num_results=3,
                text=True,
                include_domains=["https://extra.com"]
                )

            print("Result:", result)
            return result 
        else:
            print("Error from OpenAI API:", response.status_code, response.text)
            return JSONResponse(content={"error": response.text}, status_code=response.status_code)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


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

            # try:
            #     if result_dict.get("found") == "true":
            #         print("Garage door found in the image!")
            #         garage_data = result_dict["garage"]

            #         print("Garage Data:", garage_data)
            #         # Resize new garage image to the specified width and height
            #         print("Resizing new garage image...")
            #         # Load the new garage door image
            #         print("loading new garage image: ", new_garage_file.filename)
            #         new_garage_image = Image.open(io.BytesIO(await new_garage_file.read())) 
            #         new_garage_resized = new_garage_image.resize(
            #             (garage_data["width_pixels"], garage_data["height_pixels"])
            #         )

            #         # Copy the original image to preserve it before modifying
            #         modified_image = original_image.copy()

            #         # Paste the new garage image onto the copied image at the specified coordinates
            #         print("Pasting new garage image onto original image on coordinates:", garage_data["x"], garage_data["y"])
            #         modified_image.paste(new_garage_resized, (garage_data["x"], garage_data["y"]))

            #         print("Garage door replaced successfully!")
            #         print("Returning original and modified images...")

            #         # Save both images to separate buffers
            #         original_buffer = io.BytesIO()
            #         original_image.save(original_buffer, format="JPEG")
            #         original_buffer.seek(0)

            #         modified_buffer = io.BytesIO()
            #         modified_image.save(modified_buffer, format="JPEG")
            #         modified_buffer.seek(0)

            #         # Create a ZIP file containing both images
            #         zip_buffer = io.BytesIO()
            #         with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            #             zip_file.writestr("original_image.jpg", original_buffer.getvalue())
            #             zip_file.writestr("modified_image.jpg", modified_buffer.getvalue())
            #         zip_buffer.seek(0)

            #         # Return the ZIP file as a streaming response
            #         return StreamingResponse(zip_buffer, media_type="application/zip", headers={
            #             "Content-Disposition": "attachment; filename=garage_images.zip"
            #         })
            #     else:
            #         return JSONResponse(content={"error": "No garage door found in the image"}, status_code=400)
            # except Exception as e:
                # print("Error replacing garage door:", str(e))
                # return JSONResponse(content={"error": str(e)}, status_code=400)
        else:
            print("Error from OpenAI API:", response.status_code, response.text)
            return JSONResponse(content={"error": response.text}, status_code=response.status_code)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)



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
    

@app.post("/capture-receipt/")
async def capture_receipt(file: UploadFile = File(...)):
    try:
        # Load the receipt image
        print("Loading receipt image:", file.filename)
        receipt_image = Image.open(io.BytesIO(await file.read()))

        # Encode the receipt image to base64
        base64_image = encode_image(receipt_image)

        # Prepare the payload for OpenAI API
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", 
                         "text": 
                         "Extract the details from this receipt image and return it in JSON format. The JSON should include fields like 'store_name', 'date', 'items' (which should be a list of objects with 'name', 'quantity', 'price', etc.), 'total_amount', and any other relevant details."},
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

            # Return the extracted details as a JSON response
            print("Extracted Receipt Data:", result_dict)
            return JSONResponse(content=result_dict)

        else:
            print("Error from OpenAI API:", response.status_code, response.text)
            return JSONResponse(content={"error": response.text}, status_code=response.status_code)

    except Exception as e:
        print("Error processing receipt:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=400)


query_instructions = """
Given the uploaded file, process the file to answer the user's query. Use the following guideline:
if the file is .xlsx file, then directly process it as excel file. 
if the file is .csv file, then process it as csv file. 
Always assume that there is a header row in the file and use that to answer user's question. 
User query is provided as json for clarity.  Simplify it further, for example, if user says "June 24", you should use it as June 2024 i.e the entire month of June. This query simplification should be done based on the data structure of the uploaded file e.g. if it's monthly date then user query should be simplified and assumed to be monthly. 
"""


@app.post("/agent-response/")
async def agent_response(query: str = Form(...), file: UploadFile = File(...)):
    try:
        # convert the query to json
        response = openai.chat.completions.create(
            model="gpt-4o",
            response_format={ "type": "json_object" },
            messages=[
                {
                    "role": "system", "content": "Analyze user query to clarify it so there is no ambiguity and convert it to JSON format. If there are dates mentioned e.g. June 24, simplify to June 2024 etc. Return data as JSON",
                    "role": "user", "content": f"User query to be analyzed and converted to json: {query}"
                }
            ]
        )

        user_query = response.choices[0].message.content

        print("User Query:", user_query)

        # Load the file content
        print("Loading file:", file.filename)
        file_content = await file.read()

        # Upload the file to OpenAI
        uploaded_file = openai.files.create(
            file=io.BytesIO(file_content),
            purpose='assistants'
        )
        print("File uploaded with ID:", uploaded_file.id)

        # Create an Assistant with the Code Interpreter tool enabled
        assistant = openai.beta.assistants.create(
            name="File Processor Assistant",
            instructions="You are an assistant that processes the uploaded file to answer the user's query.",
            model="gpt-4o",
            tools=[{"type": "code_interpreter"}],
            tool_resources={
                "code_interpreter": {
                    "file_ids": [uploaded_file.id]
                }
            }
        )
        print("Assistant created with ID:", assistant.id)

        # Create a Thread for the conversation
        thread = openai.beta.threads.create()
        print("Thread created with ID:", thread.id)

        # Add the user's message to the Thread
        message = openai.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_query
        )
        print("Message added to thread with content:", query)

        # Run the Assistant on the Thread to get a response
        run = openai.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant.id,
            instructions=query_instructions
        )
        print("Assistant response generated")

        if run.status == 'completed': 
            messages = openai.beta.threads.messages.list(
                thread_id=thread.id
            )
            print("Messages retrieved successfully")

            # Extract the content from the messages and convert to a list of JSON objects
            message_contents = []
            first_response = None
            for msg in messages.data:
                if msg.role == "assistant":
                    content_blocks = msg.content
                    if content_blocks and content_blocks[0].type == "text":
                        first_response = content_blocks[0].text.value
                        break

            if first_response:
                return JSONResponse(content={"response": first_response})
            else:
                return JSONResponse(content={"error": "No valid response from the assistant."}, status_code=500)

        else:
            print(run.status)
            

        return JSONResponse(content={"response": messages})

    except Exception as e:
        print("Error processing request:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=400)


def process_in_background(task_id: str, query: str, file_content: bytes):
    try:
        # Upload the file to OpenAI
        uploaded_file = openai.files.create(
            file=io.BytesIO(file_content),
            purpose='assistants'
        )

        # Create an Assistant
        assistant = openai.beta.assistants.create(
            name="File Processor Assistant",
            instructions="You are an assistant that processes the uploaded file to answer the user's query.",
            model="gpt-4o",
            tools=[{"type": "code_interpreter"}],
            tool_resources={
                "code_interpreter": {
                    "file_ids": [uploaded_file.id]
                }
            }
        )

        # Create a Thread
        thread = openai.beta.threads.create()

        # Add the user's message to the Thread
        message = openai.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=query
        )

        # Run the Assistant on the Thread to get a response
        run = openai.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant.id,
            instructions="Please process the uploaded file and answer the user's query."
        )

        if run.status == 'completed': 
            messages = openai.beta.threads.messages.list(
                thread_id=thread.id
            )

            first_response = None
            for msg in messages.data:
                if msg.role == "assistant":
                    content_blocks = msg.content
                    if content_blocks and content_blocks[0].type == "text":
                        first_response = content_blocks[0].text.value
                        break

            print("Assistant response generated:", first_response)
            task_results[task_id] = {"status": "completed", "result": first_response}
        else:
            task_results[task_id] = {"status": "failed", "result": "Assistant run not completed successfully."}

    except Exception as e:
        task_results[task_id] = {"status": "failed", "result": str(e)}

@app.post("/agent-response-async/")
async def agent_response_async(query: str, file: UploadFile, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    print("Received task with ID:", task_id)
    file_content = await file.read()
    background_tasks.add_task(process_in_background, task_id, query, file_content)
    task_results[task_id] = {"status": "processing"}
    return JSONResponse(content={"status": "Processing", "task_id": task_id}, status_code=202)

@app.get("/task-result/{task_id}")
async def get_task_result(task_id: str):
    print("All tasks:", task_results)
    result = task_results.get(task_id)
    if not result:
        return JSONResponse(content={"status": "Not Found"}, status_code=404)
    return JSONResponse(content=result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

