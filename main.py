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
import csv


# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
exa = Exa(api_key=os.getenv("EXA_API_KEY"))


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
                    "role": "system", "content": "Analyze user query to clarify it so there is no ambiguity and convert it to JSON format. If there are dates mentioned e.g. June 24, simplify to June 2024 i.e explain the data in as much detail as possible to remove any ambiguity. Return data should be { query: <query text>, intent: <intent>, ... other informations}. Intent can be one or more of count, request_information, or compare_information. Return data as JSON",
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
        print ("adding... ", user_query)

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


meta_prompt = """

Generate an AI prompt/guidlines for an AI assistant for the given company information. 
Do not return any other information. Do not summarize company information. Return the JSON prompt only as per the given example below:

Example output:
    {
      {
        "section_name": "specify_personality",
        "description": "Define the personality and expertise of the AI. This includes how the AI should present itself and the areas of knowledge it should emphasize.",
        "details": {
          "description": "A brief overview of the AI's role and the company it represents.",
          "expertise": "Specific areas of knowledge and skills the AI should focus on."
        }
      },
      {
        "section_name": "response_guidelines",
        "description": "Provide detailed instructions on how the AI should craft its responses.",
        "details": {
          "language_matching": "Ensure the AI's response matches the language of the user's query.",
          "clarity_and_precision": "Guide the AI to provide clear, accurate responses and to acknowledge any limitations.",
          "expansion_and_value": "Encourage the AI to add additional relevant information when appropriate.",
          "coding_queries": "Instruct the AI to generate code with comments for software-related queries.",
          "email_request": "Direct the AI not to ask for the user's email."
        }
      },
      {
        "section_name": "interaction_etiquettes",
        "description": "Outline the expected behavior during interactions with users.",
        "details": {
          "satisfaction_check": "Instruct the AI to ask users if they are satisfied with the response.",
          "context_continuation": "Guide the AI to maintain context throughout the conversation."
        }
      },
      {
        "section_name": "formatting_guidelines",
        "description": "Specify how the AI should format its responses for clarity and readability.",
        "details": {
          "structured_format": "Encourage the use of a clear, structured format in responses.",
          "use_of_headings": "Suggest using headings to organize main points.",
          "bullet_points_and_lists": "Recommend the use of bullet points or numbered lists for subpoints.",
          "summary_in_complex_responses": "Advise including a summary for complex or lengthy responses."
        }
      },
      {
        "section_name": "scope_of_assistance",
        "description": "Define the extent to which the AI should go to assist users.",
        "details": {
          "broad_scope": "Instruct the AI to be as comprehensive as possible, leveraging all available resources to provide accurate and helpful responses."
        }
      }
    }

"""

meta_prompt2 = """

Create an AI guideline for an AI bot on how to answer and interact with user queries for the company infomration described below:

Example output:
{
  "introduction": {
    <describe what the bot is e.g. You are an AI assistant for $company providing $service>
  }
  "personality": {
    <add personality guideline here based on context>
  },
  "response_guidelines": {
    <add response guideline here based on context>
  },
  "interaction_etiquettes": {
    <add interaction guideline here based on context>
  },
  "formatting_and_presentation": {
    <add formatting guideline here based on context> 
  },
  "scope_of_assistance": {
    "broad_scope": "<add scope here> "
  }
}

Company information is 

"""
@app.get("/generate-prompt/")
async def generate_prompt(summary: str, use_case: str, scope: str):
    # convert the query to json
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", "content": f"{meta_prompt2} : {summary} \n use case: {use_case} \n scope: {scope}",
                #  "role": "user", "content": f"Company information: {text}"
            }
        ]
    )

    prompt = response.choices[0].message.content

    print("Prompt:", prompt)
    return JSONResponse(content=prompt)


input_json = {
    "Input - Type of income": [
        {
            "Type of income": "Employment",
            "Integer": 1
        },
        {
            "Type of income": "Self-Employment",
            "Integer": 2
        },
        {
            "Type of income": "Spousal",
            "Integer": 3
        },
        {
            "Type of income": "Investment Income",
            "Integer": 4
        },
        {
            "Type of income": "Rental Income",
            "Integer": 5
        }
    ],
    "Input - Profession": [
        {
            "Profession": "Doctor/Lawyer",
            "Integer": 1
        },
        {
            "Profession": "Finance/Accountant/HR/IT/Sales",
            "Integer": 2
        },
        {
            "Profession": "Operational professional",
            "Integer": 3
        },
        {
            "Profession": "Civil servant (incl. teachers)",
            "Integer": 4
        },
        {
            "Profession": "Trades",
            "Integer": 5
        },
        {
            "Profession": "Bank",
            "Integer": 6
        },
        {
            "Profession": "RE Agent or Mortgage Agent",
            "Integer": 7
        },
        {
            "Profession": "Other",
            "Integer": 8
        }
    ],
    "Input - Credit Score": [
        {
            "Credit score": "750+",
            "Integer": 1
        },
        {
            "Credit score": "700 - 749",
            "Integer": 2
        },
        {
            "Credit score": "680 -699",
            "Integer": 3
        },
        {
            "Credit score": "650 - 679",
            "Integer": 4
        },
        {
            "Credit score": "625 - 649",
            "Integer": 5
        },
        {
            "Credit score": "600 - 624",
            "Integer": 6
        },
        {
            "Credit score": "575 - 599",
            "Integer": 7
        },
        {
            "Credit score": "550 - 574",
            "Integer": 8
        },
        {
            "Credit score": "500 -549",
            "Integer": 9
        },
        {
            "Credit score": "<500",
            "Integer": 10
        }
    ],
    "Input - Kids": [
        {
            "Kids": "Yes",
            "Integer": 1
        },
        {
            "Kids": "No",
            "Integer": 2
        }
    ],
    "Input - Married": [
        {
            "Unnamed: 0": "Yes",
            "Integer": 1
        },
        {
            "Unnamed: 0": "No",
            "Integer": 2
        }
    ],
    "Input - Education": [
        {
            "Education": "Not HighSchool Grad",
            "Integer": 1
        },
        {
            "Education": "Highschool Grad",
            "Integer": 2
        },
        {
            "Education": "Undergraduate",
            "Integer": 3
        },
        {
            "Education": "Masters",
            "Integer": 4
        },
        {
            "Education": "PhD",
            "Integer": 5
        }
    ],
    "Input - Type of Financing": [
        {
            "Type of financing": "Purchase",
            "Integer": 1
        },
        {
            "Type of financing": "Switch",
            "Integer": 2
        },
        {
            "Type of financing": "Refi",
            "Integer": 3
        }
    ],
    "Input - Other Properties": [
        {
            "Other properties": "Yes",
            "Integer": 1
        },
        {
            "Other properties": "No",
            "Integer": 2
        }
    ],
    "Input - Race": [
        {
            "Race": "African Canadian",
            "Integer": 1
        },
        {
            "Race": "Asian Canadian",
            "Integer": 2
        },
        {
            "Race": "South Pacific Canadian",
            "Integer": 3
        },
        {
            "Race": "Native",
            "Integer": 4
        },
        {
            "Race": "White",
            "Integer": 5
        }
    ],
    "Input - Gender": [
        {
            "Gender": "M",
            "Integer": 1
        },
        {
            "Gender": "F",
            "Integer": 2
        },
        {
            "Gender": "T",
            "Integer": 3
        },
        {
            "Gender": "N ",
            "Integer": 4
        },
        {
            "Gender": "PNTS",
            "Integer": 5
        }
    ],
    "Input - Type of Home": [
        {
            "Type of Home": "Condo",
            "Integer": 1
        },
        {
            "Type of Home": "Town House",
            "Integer": 2
        },
        {
            "Type of Home": "Semi-Detached",
            "Integer": 3
        },
        {
            "Type of Home": "Single-Detached",
            "Integer": 4
        },
        {
            "Type of Home": "Mult-Family",
            "Integer": 5
        }
    ],
    "Input - Home Location": [
        {
            "Home Location": "Barrie",
            "Integer": 1
        },
        {
            "Home Location": "Newmarket",
            "Integer": 2
        },
        {
            "Home Location": "Aurora",
            "Integer": 3
        },
        {
            "Home Location": "Richmond Hill",
            "Integer": 4
        },
        {
            "Home Location": "Vaughan",
            "Integer": 5
        },
        {
            "Home Location": "Thornhill",
            "Integer": 6
        },
        {
            "Home Location": "North York",
            "Integer": 7
        },
        {
            "Home Location": "Mid-town Toronto",
            "Integer": 8
        },
        {
            "Home Location": "Toronto West",
            "Integer": 9
        },
        {
            "Home Location": "Down-town Toronto",
            "Integer": 10
        },
        {
            "Home Location": "East York",
            "Integer": 11
        },
        {
            "Home Location": "Etobicoke",
            "Integer": 12
        },
        {
            "Home Location": "Mississauga",
            "Integer": 13
        },
        {
            "Home Location": "Oakville",
            "Integer": 14
        },
        {
            "Home Location": "Burlington",
            "Integer": 15
        }
    ],
    "Input - Age": [
        {
            "Age": "<23",
            "Integer": 1
        },
        {
            "Age": "23-29",
            "Integer": 2
        },
        {
            "Age": "30-34",
            "Integer": 3
        },
        {
            "Age": "35-39",
            "Integer": 4
        },
        {
            "Age": "40-44",
            "Integer": 5
        },
        {
            "Age": "45-49",
            "Integer": 6
        },
        {
            "Age": "50-54",
            "Integer": 7
        },
        {
            "Age": "55-60",
            "Integer": 8
        },
        {
            "Age": "64-69",
            "Integer": 9
        },
        {
            "Age": ">69",
            "Integer": 10
        }
    ],
    "Input - Loan To Value": [
        {
            "Loan To Value": "<10%",
            "Integer": 1
        },
        {
            "Loan To Value": "10% - 20%",
            "Integer": 2
        },
        {
            "Loan To Value": "20.1% - 30%",
            "Integer": 3
        },
        {
            "Loan To Value": "30.1% - 40%",
            "Integer": 4
        },
        {
            "Loan To Value": "40.1% - 50%",
            "Integer": 5
        },
        {
            "Loan To Value": "50.1% - 60%",
            "Integer": 6
        },
        {
            "Loan To Value": "60.1% - 70%",
            "Integer": 7
        },
        {
            "Loan To Value": "70.1% - 80%",
            "Integer": 8
        },
        {
            "Loan To Value": "80.1% - 90%",
            "Integer": 9
        },
        {
            "Loan To Value": ">90%",
            "Integer": 10
        }
    ],
    "Input - Variables overview": [
        {
            "Variable": "Income level",
            "Value": 1
        },
        {
            "Variable": "Type of Income",
            "Value": 2
        },
        {
            "Variable": "Profession",
            "Value": 3
        },
        {
            "Variable": "Credit score",
            "Value": 4
        },
        {
            "Variable": "Requested Loan Value",
            "Value": 5
        },
        {
            "Variable": "Total LTV",
            "Value": 6
        },
        {
            "Variable": "Education level",
            "Value": 7
        },
        {
            "Variable": "Married",
            "Value": 8
        },
        {
            "Variable": "Kids",
            "Value": 9
        },
        {
            "Variable": "Age",
            "Value": 10
        },
        {
            "Variable": "Location",
            "Value": 11
        },
        {
            "Variable": "Type of home",
            "Value": 12
        },
        {
            "Variable": "Own other properties",
            "Value": 13
        },
        {
            "Variable": "Type of Financing",
            "Value": 14
        },
        {
            "Variable": "Gender",
            "Value": 15
        },
        {
            "Variable": "Race",
            "Value": 16
        }
    ],
    "Input - income level": [
        {
            "Income level": "<50,000",
            "Integer": 1
        },
        {
            "Income level": "50,001 - 75,000",
            "Integer": 2
        },
        {
            "Income level": "75,001 - 100,000",
            "Integer": 3
        },
        {
            "Income level": "100,001 - 125,000",
            "Integer": 4
        },
        {
            "Income level": "125,001 - 150,000",
            "Integer": 5
        },
        {
            "Income level": "150,001 - 175,000",
            "Integer": 6
        },
        {
            "Income level": "175,001 - 200,000",
            "Integer": 7
        },
        {
            "Income level": "200,001 - 225,000",
            "Integer": 8
        },
        {
            "Income level": "225,001 - 250,000",
            "Integer": 9
        },
        {
            "Income level": "250,001 - 300,000",
            "Integer": 10
        },
        {
            "Income level": "300,001 - 350,000",
            "Integer": 11
        },
        {
            "Income level": "350,001 - 400,000",
            "Integer": 12
        },
        {
            "Income level": "401,000 - 425,000",
            "Integer": 13
        },
        {
            "Income level": "425,001 - 450,000",
            "Integer": 14
        },
        {
            "Income level": "450,001 - 475,000",
            "Integer": 15
        },
        {
            "Income level": "475,001 - 500,000",
            "Integer": 16
        },
        {
            "Income level": ">500,000 ",
            "Integer": 17
        }
    ],
    "Input - Requested Loan Value": [
        {
            "Loan Value": "<100,000",
            "Integer": 1
        },
        {
            "Loan Value": "100,000 - 200,000",
            "Integer": 2
        },
        {
            "Loan Value": "200,001 - 300,000",
            "Integer": 3
        },
        {
            "Loan Value": "300,001 - 400,000",
            "Integer": 4
        },
        {
            "Loan Value": "400,001 - 500,000",
            "Integer": 5
        },
        {
            "Loan Value": "500,001 - 600,000",
            "Integer": 6
        },
        {
            "Loan Value": "600,001 - 700,000",
            "Integer": 7
        },
        {
            "Loan Value": "700,001 - 800,000",
            "Integer": 8
        },
        {
            "Loan Value": "800,001 - 900,000",
            "Integer": 9
        },
        {
            "Loan Value": "900,001 - 1,000,000",
            "Integer": 10
        },
        {
            "Loan Value": "1,000,001 - 1,100,000",
            "Integer": 11
        },
        {
            "Loan Value": "1,100,001 - 1,200,000",
            "Integer": 12
        },
        {
            "Loan Value": "1,200,001 - 1,300,000",
            "Integer": 13
        },
        {
            "Loan Value": "1,300,001 - 1,400,000",
            "Integer": 14
        },
        {
            "Loan Value": "1,400,001 - 1,500,000",
            "Integer": 15
        },
        {
            "Loan Value": ">1,500,000",
            "Integer": 16
        }
    ]
}


outcome_json = {
    "Outcome Rates 1": [
        {
            "Rates": "<4%",
            "Integer": 1
        },
        {
            "Rates": "4% - 4.5%",
            "Integer": 2
        },
        {
            "Rates": "4.51% - 5%",
            "Integer": 3
        },
        {
            "Rates": "5.01% - 5.5%",
            "Integer": 4
        },
        {
            "Rates": "5.51% - 6%",
            "Integer": 5
        },
        {
            "Rates": "6.01% - 6.5%",
            "Integer": 6
        },
        {
            "Rates": "6.51% - 7%",
            "Integer": 7
        },
        {
            "Rates": "7.01% - 7.5%",
            "Integer": 8
        },
        {
            "Rates": "7.51% - 8%",
            "Integer": 9
        },
        {
            "Rates": "8.01% - 8.5%",
            "Integer": 10
        },
        {
            "Rates": "8.51% - 9%",
            "Integer": 11
        },
        {
            "Rates": "9.01% - 9.5%",
            "Integer": 12
        },
        {
            "Rates": "9.51% - 10%",
            "Integer": 13
        },
        {
            "Rates": "10.01% - 10.5%",
            "Integer": 14
        },
        {
            "Rates": "10.51% - 11%",
            "Integer": 15
        },
        {
            "Rates": "11.1% - 12%",
            "Integer": 16
        },
        {
            "Rates": "12.1%-13%",
            "Integer": 17
        }
    ],
    "Outcome Results": [
        {
            "Success": "Financing",
            "Integer": 1.0
        },
        {
            "Success": "No Financing",
            "Integer": 2.0
        }
    ],
    "Outcome Type": [
        {
            "Success": "Type",
            "Integer": None
        },
        {
            "Success": "Prime",
            "Integer": 1.0
        },
        {
            "Success": "B Deal",
            "Integer": 2.0
        },
        {
            "Success": "Sub-Prime",
            "Integer": 3.0
        },
        {
            "Success": "Fees",
            "Integer": None
        }
    ],
    "Outcome Rates 2": [
        {
            "Success": "<1%",
            "Integer": 1.0
        },
        {
            "Success": "1%-1.99%",
            "Integer": 2.0
        },
        {
            "Success": "2%-2.99%",
            "Integer": 3.0
        },
        {
            "Success": "3%-3.99%",
            "Integer": 4.0
        },
        {
            "Success": "4%-4.99%",
            "Integer": 5.0
        },
        {
            "Success": "5%-5.99%",
            "Integer": 6.0
        }
    ]
}

outcome_mapping_json = {
  "Mapping": {
    "Inputs": {
      "Credit Score": {
        "Range": {
          "300-599": 1,
          "600-649": 2,
          "650-699": 3,
          "700-749": 4,
          "750-799": 5,
          "800-850": 6
        }
      },
      "Loan to Value (LTV)": {
        "Range": {
          "90-100%": 1,
          "80-89%": 2,
          "70-79%": 3,
          "60-69%": 4,
          "50-59%": 5,
          "<50%": 6
        }
      },
      "Type of Income": {
        "Category": {
          "Salaried": 1,
          "Self-Employed": 2,
          "Retired": 3,
          "Unemployed": 4
        }
      },
      "Age": {
        "Range": {
          "18-25": 1,
          "26-35": 2,
          "36-45": 3,
          "46-60": 4,
          "61+": 5
        }
      },
      "Type of Home": {
        "Category": {
          "Single-Family": 1,
          "Multi-Family": 2,
          "Condominium": 3,
          "Townhouse": 4
        }
      }
    },
    "Score Calculation": {
      "Formula": "Sum of input variable scores"
    },
    "Outcomes": {
      "Outcome Rates": {
        "Mapping": {
          "1-3": "<4%",
          "4-5": "4% - 4.5%",
          "6-7": "4.51% - 5%",
          "8-9": "5.01% - 5.5%",
          "10-11": "5.51% - 6%",
          "12-13": "6.01% - 6.5%",
          "14-15": "6.51% - 7%",
          "16-17": "7.01% - 7.5%",
          "18+": "7.51% - 8%"
        }
      },
      "Outcome Results": {
        "Mapping": {
          "1-7": "Financing",
          "8+": "No Financing"
        }
      },
      "Outcome Type": {
        "Mapping": {
          "1-4": "Prime",
          "5-7": "B Deal",
          "8-10": "Sub-Prime",
          "11+": "Fees"
        }
      }
    }
  }
}

@app.post("/calculate_mortage_outcome/")
async def generate_prompt(file: UploadFile = File(...)):
    # convert the query to json
    mortagage_prompt = (
        f"Given the following input data as JSON, use the scheme from {input_json} "
        f"to calculate the outcome using the scheme from {outcome_json} and the mapping "
        f"from {outcome_mapping_json}. Return the result as a summary form with the outcome, "
        f"outcome rates, results, and type."
    )

    # Read the CSV file content
    file_content = await file.read()
    file_content = file_content.decode('utf-8')

    # Parse CSV content
    reader = csv.DictReader(file_content.splitlines())
    
    outcome = []
    
    for row in reader:
        # Convert each line to JSON
        input_data = json.dumps(row)

        # Create the full prompt with the input data
        full_prompt = mortagage_prompt + f"\nInput Data: {input_data}"
    
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", "content": full_prompt
                }
            ]
        )

        result = response.choices[0].message.content
        #add to outcome
        outcome.append(result)

    print("Result:", outcome)
    return JSONResponse(content=outcome)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


