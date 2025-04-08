from supabase_client import supabase
from io import BytesIO
import httpx
import asyncio
from fastapi import FastAPI, HTTPException
from config import (
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    SUPABASE_URL,
    GEMINI_MODEL,
    ASSEMBLYAI_API_KEY
)
import mimetypes
from urllib.parse import urlparse
from datetime import datetime, timedelta
from typing import AsyncIterator
import os
from dotenv import load_dotenv
import math
from openai import OpenAI
from moviepy.editor import VideoFileClip
from datetime import datetime, timedelta
import tempfile
from typing import List
import re
import pytz
import google.generativeai as genai
import json
from pydub import AudioSegment
import os
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import time 

import aiofiles  

from google.cloud import storage

import uuid
import aiohttp

from embeddings import get_embeddings
from vader_sentiment import get_vader_sentiment
from similarity import find_similar_feedbacks
import datetime
import assemblyai as aai


# Configure the API key
genai.configure(api_key=GEMINI_API_KEY)

# Add AssemblyAI config after other configs
aai.settings.api_key = ASSEMBLYAI_API_KEY # Replace with your key

CHUNK_DURATION_MS = 1 * 60 * 1000  # 5 minutes in milliseconds


def extract_audio_from_video(video_file_path: str, temp_audio_dir: str) -> str:
    """
    Extracts audio from video file and saves it as WAV.
    
    Args:
        video_file_path: Path to the video file
        temp_audio_dir: Directory to save the audio file
        
    Returns:
        str: Path to the extracted audio file
    """
    print("Extracting audio from video")

    try:
        # Extract the full audio from the video
        with VideoFileClip(video_file_path) as video:
            audio = video.audio
            full_audio_path = os.path.join(temp_audio_dir, "full_audio.wav")
            audio.write_audiofile(full_audio_path)
            print(f"Full audio extracted: {full_audio_path}")
            
        return full_audio_path

    except Exception as e:
        print(f"An error occurred during audio extraction: {e}")
        raise
    finally:
        print("Audio extraction completed")


async def check_balance(
    workspace_id: str,
    video_duration_seconds: float):
    try:
        # Calculate credits needed based on video duration
        # Formula: (duration_in_seconds / 10) + 50, rounded up
        credits_estimate = math.ceil(video_duration_seconds / 10) + 50

        # Call the Supabase RPC to get credits and has_balance
        response = await asyncio.to_thread(lambda: supabase.rpc('get_wallet_balance', {'workspace_id_param': workspace_id}).execute())

        # Handle the case where RPC returns no results
        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=400, detail="Invalid workspace_id or no wallet information available")

        # Assuming response returns a list of dictionaries with 'has_balance' and 'credits' keys
        wallet_details = response.data[0]  # Assuming only one row is returned
        available_credits = wallet_details.get('credits_balance')

        # Compare credits_estimate with available credits
        if credits_estimate > available_credits:
            return {
                "status": "error", 
                "message": "Insufficient credits in workspace wallet.",
                "required_credits": credits_estimate,
                "available_credits": available_credits
            }
        
        # If sufficient balance exists
        return {
            "status": "success",
            "message": "Sufficient credits available",
            "required_credits": credits_estimate,
            "available_credits": available_credits
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

# Get product brief, if updated by users of a workspace
async def get_product_brief(workspace_id: str):
    product_brief = await asyncio.to_thread(
        lambda: supabase.table("products")
                        .select("*")
                        .eq("workspace_id", workspace_id)
                        .execute()
    )
    if not product_brief.data:
        return None

    return product_brief.data[0]

# Extract insights using LLM
async def infer_with_llm(transcript):
    prompt = f'''You are a helpful assistant .

    You will receive a transcript of a customer meeting that includes utterances, speaker labels, start and end times (in seconds), sentiment, and confidence levels.


        Your tasks:
        1. Summarize the meeting discussion.
        2. List key action items in bullet point format, along with action owners (if specific).
        3. Extract customer insights from the transcript and categorize them into one of these themes: Ideas, Problems, Complaints, Appreciations, Questions, Compete mentions, Pricing mentions, Customer support, Customer education, Needs triage.

        For each insight, provide the following:
        - Theme (must be one of the above, strictly)
        - Insight
        - Context (summarized context from the conversation to help understand the background details supporting the insight)
        - Product_Area (if specific)
        - Feature_Recommendation (if applicable)
        - Verbatim (actual spoken text from the customer, representative of the full context needed to understand the insight)
        - Sentiment (POSITIVE, NEGATIVE, NEUTRAL)
        - Feedback_By (if customer speaker can be inferred)
        - Start and End times of the verbatim (These will be used to extract video clips from the meeting, so ensure we keep extended times to cover understanding behind the insight, context and verbatim.)

        Transcript: {transcript}

        Provide a response in STRICT JSON format. Do not include any code block formatting or ```json tags. Ensure the structure is correct with no extra whitespace or formatting issues.

        Here is an example output format:
        {{
            "summary": "In this meeting, the customer discussed product feature requests, concerns, and pricing inquiries. Key action items include addressing issues and considering new feature requests.",
            "action_items": [
                {{
                    "action_item_id": "1",
                    "action_item_owner": "Michelle",
                    "action_item": "Michelle to close the support ticket."
                }},
                {{
                    "action_item_id": "2",
                    "action_item_owner": "Jack",
                    "action_item": "Jack to respond to ticket resolution feedback survey."
                }}
            ],
            "insights": [
                {{
                "Theme": "customer support",
                "Insight": "Michelle effectively handled Jack's complaint by offering a free service appointment, making Jack feel valued.",
                "Context": "Michelle apologized for the issue and offered Jack two options for getting his car serviced, assuring him it would be free.",
                "Product_Area": "Customer Support",
                "Feature_Recommendation": "Continue providing excellent customer service with free offerings.",
                "Verbatim": "Thank you Mr. Satilly. We would love to get that taken care of for you. We have two options right now...",
                "Sentiment": "POSITIVE",
                "Feedback_By": "Michelle",
                "Start": 91,
                "End": 120
                }},
                {{
                "Theme": "ideas",
                "Insight": "Customer suggested adding more integration options with third-party tools.",
                "Context": "The customer feels that current integration options are limited and suggests supporting third-party tools for workflow automation.",
                "Product_Area": "Integration",
                "Feature_Recommendation": "Add more third-party integration options, particularly for task automation tools.",
                "Verbatim": "Right now, I feel like the integration options are kind of limited...",
                "Sentiment": "NEUTRAL",
                "Feedback_By": "Customer",
                "Start": 205,
                "End": 220
                }}
            ]
        }}
        '''

    # Initialize the OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)

    try:
        # Using the new API structure
        chat_completion = client.chat.completions.create(  # Note: Added 'await' to the call
            messages=[{"role": "user", "content": prompt}],
            model=OPENAI_MODEL,
        )

        # Correctly accessing the response content
        llm_str = chat_completion.choices[0].message.content  # Use dot notation instead of brackets
        llm_json = json.loads(llm_str)

        # Validate JSON response
        if llm_json is None or "summary" not in llm_json or "insights" not in llm_json:
            print("Invalid response from the LLM:", llm_str)
            return None

        summary = llm_json.get("summary")
        insights = llm_json.get("insights")
        action_items = llm_json.get("action_items")
        prompt_tokens = chat_completion.usage.total_tokens  # Access using dot notation

        return {
            "summary": summary,
            "insights": insights,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": chat_completion.usage.completion_tokens,  # Access using dot notation
            "action_items": action_items
        }

    except OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None



# Insert uploaded recording conversational insights
async def insert_uploaded_recording_conversational_insights(
    workspace_id: str, 
    title: str, 
    llmInferenceInsights, 
):
    clips = []
    print(f"Starting to insert insights for workspace {workspace_id}")
    print(f"Number of insights to process: {len(llmInferenceInsights)}")
    
    for index, insight in enumerate(llmInferenceInsights):
        try:
            print(f"Processing insight {index + 1}/{len(llmInferenceInsights)}")
            
            # Convert clip_start to previous closest integer (floor) and clip_end to next closest integer (ceil)
            clip_start = insight.get("Start")
            clip_end = insight.get("End")

            print(f"Clip timing - Start: {clip_start}, End: {clip_end}")

            if clip_start is not None:
                clip_start = math.floor(float(clip_start))
            if clip_end is not None:
                clip_end = math.ceil(float(clip_end))

            data = {
                "workspace_id": workspace_id,
                "title": title,
                "embeddings": insight.get("embeddings"),
                "sentiment": insight.get("sentiment"),
                "words": insight.get("words"),
                "feedback": insight.get("Verbatim"),
                "feedback_at": datetime.datetime.now().isoformat(),
                "feedback_insight_id": insight.get("feedback_insight_id"),
                "source": insight.get("source")
            }

            print(f"Inserting feedback data: {json.dumps(data, default=str)}")

            # Insert data using Supabase client
            try:
                response = await asyncio.to_thread(
                    lambda: supabase.table("feedback")
                                    .insert(data)
                                    .execute()
                )
                print(f"Supabase response: {response}")
                
                if not response.data:
                    print(f"Warning: No data returned from Supabase insert")
                    continue

                # Check if the insert was successful and extract feedback_id
                for feedback in response.data:
                    feedback_id = feedback.get("feedback_id")
                    if feedback_id:
                        clips.append({
                            "feedback_id": feedback_id,
                            "clip_start": clip_start,
                            "clip_end": clip_end
                        })
                        print(f"Added clip for feedback_id: {feedback_id}")
                    else:
                        print(f"Warning: No feedback_id in response data: {feedback}")

            except Exception as e:
                print(f"Error inserting into Supabase: {str(e)}")
                if hasattr(e, '__traceback__'):
                    import traceback
                    print(f"Full traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
                raise

        except Exception as e:
            print(f"Error processing insight {index + 1}: {str(e)}")
            if hasattr(e, '__traceback__'):
                import traceback
                print(f"Full traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
            raise

    print(f"Successfully processed {len(clips)} clips")
    return clips

# Extract single insight clip from video recording
async def process_single_clip(clip_info, temp_file_path):
    """Extract a clip from the video file based on start and end times."""
    # Change from consolidated_insight_id to feedback_id
    feedback_id = clip_info.get("feedback_id")
    clip_start = clip_info.get("clip_start")
    clip_end = clip_info.get("clip_end")
        
    # Update validation check
    if feedback_id is None or clip_start is None or clip_end is None:
        raise ValueError(f"Clip info missing required fields. Got feedback_id: {feedback_id}, start: {clip_start}, end: {clip_end}")

    print(f"Processing clip for feedback_id: {feedback_id} from {clip_start}s to {clip_end}s")

    # Create temporary paths for the video clip and audio file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_clip_file:
        output_clip_path = temp_clip_file.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        temp_audio_path = temp_audio_file.name
        
    try:
        with VideoFileClip(temp_file_path) as video:
            # Ensure clip times are within video duration
            video_duration = video.duration
            clip_start = max(0, min(clip_start, video_duration))
            clip_end = max(clip_start, min(clip_end, video_duration))
            
            print(f"Extracting clip from {clip_start}s to {clip_end}s")
            trimmed_video = video.subclip(clip_start, clip_end)
            trimmed_video.write_videofile(
                output_clip_path, 
                codec="libx264",
                temp_audiofile=temp_audio_path,
                remove_temp=True,
                logger=None  # Suppress moviepy output
            )
            
        return output_clip_path
    except Exception as e:
        print(f"Error processing clip: {str(e)}")
        if os.path.exists(output_clip_path):
            os.remove(output_clip_path)
        raise
    finally:
        # Clean up the temporary audio file if it exists
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception as e:
                print(f"Error removing temp audio file: {e}")

# Upload conversational insight clip into supabase
async def upload_clip_to_supabase(clip_path, file_name, workspace_id, title):
    full_path = f"{workspace_id}/{title}/{file_name}"
        
    try:
        with open(clip_path, "rb") as file_stream:
            await asyncio.to_thread(
                lambda: supabase.storage.from_("conversations")
                                        .upload(file=file_stream, path=full_path, file_options={"content-type": "video/mp4"})
            )
    except Exception as e:
        raise RuntimeError(f"Error uploading clip to Supabase: {e}")

# Process conversation insight clips
async def process_clips(workspace_id: str, temp_file_path: str, clips, title: str):
    """Process and upload all clips."""
    uploaded_files = []
    max_retries = 3  
    retry_delay = 2

    for clip_info in clips:
        retry_count = 0
        while retry_count < max_retries:
            try:
                feedback_id = clip_info.get("feedback_id")
                if not feedback_id:
                    print("Missing feedback_id in clip info")
                    break

                print(f"Processing clip for feedback_id: {feedback_id}")
                
                # Process the single clip in a temp file
                processed_clip_path = await process_single_clip(clip_info, temp_file_path)

                # Generate the filename for storage
                file_name = f"clip_{feedback_id}.mp4"
                
                print(f"Uploading clip to {workspace_id}/{title}/{file_name}")
                
                # Upload the clip to Supabase
                await upload_clip_to_supabase(processed_clip_path, file_name, workspace_id, title)

                # Record the full path to the uploaded file
                uploaded_file_full_path = f"{workspace_id}/{title}/{file_name}"
                uploaded_files.append({
                    "feedback_id": feedback_id,
                    "uploaded_file_full_path": uploaded_file_full_path
                })

                # Update the database with the clip path
                await asyncio.to_thread(
                    lambda: supabase.table("feedback")
                                    .update({"clip_path": uploaded_file_full_path})
                                    .eq("feedback_id", feedback_id)
                                    .execute()
                )

                print(f"Successfully processed and uploaded clip for feedback_id: {feedback_id}")

                # Remove the temporary file after uploading
                if os.path.exists(processed_clip_path):
                    os.remove(processed_clip_path)
                break  # Exit retry loop on success
                
            except Exception as e:
                print(f"Error processing or uploading clip (attempt {retry_count + 1}): {str(e)}")
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Max retries reached for clip {clip_info.get('feedback_id')}. Skipping.")
                    break
                await asyncio.sleep(retry_delay)

    return {"uploaded_files": uploaded_files}




def transcribe_audio(audio_file_path, max_retries=3):
    """
    Transcribe the given audio file using AssemblyAI and return the transcription.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            # Initialize transcriber
            transcriber = aai.Transcriber()
            
            # Configure transcription parameters
            config = aai.TranscriptionConfig(
                speaker_labels=True,
                auto_highlights=True,
                summarization=True,
                entity_detection=True,
                sentiment_analysis=True,
                language_code="en_us",
                punctuate=True,
                format_text=True,
                content_safety=True,
                iab_categories=True,
                speakers_expected=2
            )
            
            # Start transcription with config
            transcript = transcriber.transcribe(
                audio_file_path,
                config=config
            )

            if transcript.status == aai.TranscriptStatus.error:
                print(f"Transcription error: {transcript.error}")
                raise Exception(transcript.error)

            # Convert AssemblyAI response to our expected format
            formatted_response = {
                "error": None,
                "audio_duration": transcript.audio_duration,
                "utterances": []
            }

            # Process utterances with speaker information
            for utterance in transcript.utterances:
                formatted_utterance = {
                    "text": utterance.text,
                    "speaker": f"Speaker {utterance.speaker}",
                    "start": utterance.start,  # AssemblyAI provides times in milliseconds
                    "end": utterance.end,
                    # Get sentiment from utterance if available, otherwise default to NEUTRAL
                    "sentiment": utterance.sentiment.upper() if hasattr(utterance, 'sentiment') else "NEUTRAL",
                    "confidence": utterance.confidence if hasattr(utterance, 'confidence') else 1.0
                }
                formatted_response["utterances"].append(formatted_utterance)

            return formatted_response

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {str(e)}")
            attempt += 1
            if attempt >= max_retries:
                return {
                    "error": f"Failed to transcribe audio after {max_retries} attempts: {str(e)}",
                    "audio_duration": 0,
                    "utterances": []
                }
            time.sleep(2)  # Add delay between retries



def merge_transcripts(transcripts):
    merged_data = {
        "error": None,
        "audio_duration": 0,
        "utterances": []
    }
    
    current_offset = 0  # Keep track of time offset for each chunk
    
    # Loop through each transcript JSON object and merge them
    for data in transcripts:
        # Update the overall audio_duration
        chunk_duration = data.get("audio_duration", 0)
        merged_data["audio_duration"] += chunk_duration
        
        # Get utterances from current chunk
        chunk_utterances = data.get("utterances", [])
        
        # Adjust start and end times for each utterance based on current offset
        for utterance in chunk_utterances:
            adjusted_utterance = utterance.copy()
            
            # Adjust start time if present
            if "start" in adjusted_utterance:
                adjusted_utterance["start"] = adjusted_utterance["start"] + current_offset
                
            # Adjust end time if present    
            if "end" in adjusted_utterance:
                adjusted_utterance["end"] = adjusted_utterance["end"] + current_offset
                
            merged_data["utterances"].append(adjusted_utterance)
            
        # Update offset for next chunk
        current_offset += chunk_duration

    return merged_data




async def supabase_upload(file_path, workspace_id, title):
    """
    Upload a file to Supabase storage asynchronously in the format workspace_id/title/transcript.json 
    and return the file URL.
    """
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return None

        # Read the file data into memory asynchronously
        async with aiofiles.open(file_path, 'rb') as file:
            file_data = await file.read()

        # Construct the full path using workspace_id/title/transcript.json
        full_path = f"{workspace_id}/{title}/transcript.json"

        # Use asyncio.to_thread to perform the blocking upload operation
        response = await asyncio.to_thread(
            lambda: supabase.storage.from_("transcripts").upload(path=full_path, file=file_data)
        )

        # Log the response for debugging
        print(f"Supabase upload response: {response}")

        # Check if the response status indicates success
        if response.status_code != 200:
            print(f"Error details: {response.data}")  # Print error details if upload fails
            return None
        
        # Return the file URL if successful
        return f"{SUPABASE_URL}/storage/v1/object/public/transcripts/{full_path}"

    except Exception as e:
        print(f"Error uploading file to Supabase: {e}")
        return None




async def download_video(url: str) -> str:
    """
    Downloads a video from GCS, Google Drive, or a direct URL and returns the local file path.
    
    Args:
        url (str): The URL of the video (GCS URL starting with 'gs://', Google Drive URL, or direct URL)
        
    Returns:
        str: Path to the downloaded video file
    """
    try:
        # Create temporary file
        temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video_file_path = temp_video_file.name

        if url.startswith('gs://'):
            # Handle GCS URL
            storage_client = await asyncio.to_thread(storage.Client)
            bucket_name, blob_name = url.replace("gs://", "").split("/", 1)
            
            bucket = await asyncio.to_thread(storage_client.bucket, bucket_name)
            blob = await asyncio.to_thread(bucket.blob, blob_name)
            
            # Download file asynchronously
            await asyncio.to_thread(blob.download_to_filename, temp_video_file_path)
            print(f"Downloaded video from GCS to temporary file: {temp_video_file_path}")
            
        elif 'drive.google.com' in url:
            # Handle Google Drive URL
            # Extract file ID from URL
            file_id = None
            if 'id=' in url:
                file_id = url.split('id=')[-1].split('&')[0]
            
            if not file_id:
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract Google Drive file ID from URL"
                )

            # Construct direct download URL
            download_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                async with session.get(download_url, headers=headers, allow_redirects=True) as response:
                    if response.status != 200:
                        # Try alternative download URL
                        alt_download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                        async with session.get(alt_download_url, headers=headers, allow_redirects=True) as alt_response:
                            if alt_response.status != 200:
                                raise HTTPException(
                                    status_code=alt_response.status,
                                    detail="Failed to download file from Google Drive"
                                )
                            
                            content_type = alt_response.headers.get('content-type', '')
                            if 'text/html' in content_type:
                                # Handle large file confirmation page
                                body = await alt_response.text()
                                if 'confirm=' in body:
                                    confirm_token = body.split('confirm=')[1].split('&')[0]
                                    final_url = f"{alt_download_url}&confirm={confirm_token}"
                                    
                                    async with session.get(final_url, headers=headers) as final_response:
                                        if final_response.status != 200:
                                            raise HTTPException(
                                                status_code=final_response.status,
                                                detail="Failed to download file from Google Drive after confirmation"
                                            )
                                        with open(temp_video_file_path, 'wb') as f:
                                            async for chunk in final_response.content.iter_chunked(8192):
                                                f.write(chunk)
                            else:
                                with open(temp_video_file_path, 'wb') as f:
                                    async for chunk in alt_response.content.iter_chunked(8192):
                                        f.write(chunk)
                    else:
                        with open(temp_video_file_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)
                                
                print(f"Downloaded video from Google Drive to temporary file: {temp_video_file_path}")
        else:
            # Handle direct URLs
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Failed to download video from URL. Status: {response.status}"
                        )
                    
                    # Stream the response content to file
                    with open(temp_video_file_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                    print(f"Downloaded video from URL to temporary file: {temp_video_file_path}")

        # Verify the downloaded file is a valid video
        if os.path.getsize(temp_video_file_path) == 0:
            raise HTTPException(
                status_code=400,
                detail="Downloaded file is empty"
            )

        return temp_video_file_path

    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        if os.path.exists(temp_video_file_path):
            await asyncio.to_thread(os.remove, temp_video_file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download video: {str(e)}"
        )


async def check_file_mime_type(url: str) -> str:
    """
    Check the MIME type of a file from its URL.
    
    Args:
        url (str): The URL of the file
        
    Returns:
        str: The MIME type of the file
        
    Raises:
        HTTPException: If unable to determine MIME type or if file is not accessible
    """
    try:
        # First try to get MIME type from URL path
        parsed_url = urlparse(url)
        mime_type, _ = mimetypes.guess_type(parsed_url.path)
        
        if not mime_type:
            # If can't determine from path, try HEAD request
            async with httpx.AsyncClient() as client:
                response = await client.head(url)
                mime_type = response.headers.get('content-type')

        return mime_type
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to determine file type: {str(e)}"
        )


# Main method controlling the uploaded recording insights workflow
async def analyze_uploaded_recording(
    workspace_id: str,
    title: str,
    gcs_url: str,
    source_type: str
):
    temp_video_file_path = None
    try:
        # Download video using the URL format to determine the source
        temp_video_file_path = await download_video(gcs_url)
        
        # Get video duration using moviepy
        with VideoFileClip(temp_video_file_path) as video:
            video_duration = video.duration  # Duration in seconds
            
        # Create a temporary directory for storing extracted audio file
        with tempfile.TemporaryDirectory() as temp_audio_dir:
            # Extract audio from video
            full_audio_path = extract_audio_from_video(temp_video_file_path, temp_audio_dir)
            
            print(f"Processing full audio file")
            print(f"Audio file path: {full_audio_path}")
            
            # Transcribe the entire audio file at once
            transcript = transcribe_audio(full_audio_path)
            
            if transcript.get("error"):
                print(f"Transcription error: {transcript['error']}")
                raise HTTPException(status_code=500, detail=f"Error in getting transcript: {transcript['error']}")
                
            print("Successfully transcribed audio")

            # Clean up the audio file
            try:
                os.remove(full_audio_path)
                print(f"Removed audio file: {full_audio_path}")
            except OSError as e:
                print(f"Error removing audio file {full_audio_path}: {e}")

            # Create a temporary JSON file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as json_file:
                json.dump(transcript, json_file, indent=4)
                transcript_file = json_file.name

            print(f"Transcript saved to temporary file: {transcript_file}")

            transcriptFilePath = await supabase_upload(transcript_file, workspace_id, title)

            # Delete the temporary file
            try:
                if os.path.exists(transcript_file):
                    os.remove(transcript_file)
                    print(f"Temporary file deleted: {transcript_file}")
            except Exception as e:
                print(f"Error deleting the file: {e}")

            parsed_utterances = [
                {
                    "utterance": utterance["text"],
                    "speaker": utterance["speaker"],
                    "start": math.floor(utterance["start"] / 1000) if "start" in utterance else None,
                    "end": math.ceil(utterance["end"] / 1000) if "end" in utterance else None,
                    "sentiment": utterance.get("sentiment", "NEUTRAL"),
                    "confidence": utterance.get("confidence", 1.0)
                }
                for utterance in transcript["utterances"]
            ]

            # Step 3: Infer insights from the LLM using the parsed transcript and product brief
            llm_inference_result = await infer_with_llm(parsed_utterances)
            if not llm_inference_result:
                raise HTTPException(status_code=500, detail="LLM inference failed.")

            # Extract the insights and token usage from the LLM response
            summary = llm_inference_result.get("summary")
            insights = llm_inference_result.get("insights")
            input_tokens = llm_inference_result.get("prompt_tokens")
            output_tokens = llm_inference_result.get("completion_tokens")
            action_items = json.dumps(llm_inference_result.get("action_items"))
            durationSeconds = transcript["audio_duration"]

            # Initialize list to store unique insights
            unique_insights = []

            for insight in insights:
                if verbatim := insight.get("Verbatim"):
                    embeddings = await get_embeddings(verbatim)
                    insight["embeddings"] = embeddings

                    # Get sentiment
                    sentiment_result = await get_vader_sentiment(verbatim)
                    sentiment_value = sentiment_result.get("sentiment_label")
                    insight["sentiment"] = sentiment_value

                    # Calculate word count
                    word_estimate = len(verbatim.split())
                    insight["words"] = word_estimate

                    # Find similar insights
                    similar_insights, similar_feedback_insight_id = await find_similar_feedbacks(
                        embeddings=embeddings,
                        workspace_id=workspace_id
                    )

                    if similar_feedback_insight_id:
                        # If similar insight found, just insert into feedback table
                        insight["similar_feedback_insight_id"] = similar_feedback_insight_id
                        print(f"Similar feedback insight id: {similar_feedback_insight_id}")
                        try:
                            feedback_data = {
                                "workspace_id": workspace_id,
                                "source": source_type,
                                "title": title,
                                "feedback": verbatim,
                                "words": word_estimate,
                                "sentiment": sentiment_value,
                                "embeddings": embeddings,
                                "feedback_insight_id": similar_feedback_insight_id,
                                "feedback_at": datetime.datetime.now().isoformat(),
                            }
                            
                            await asyncio.to_thread(
                                lambda: supabase.table("feedback")
                                                .insert(feedback_data)
                                                .execute()
                            )
                        except Exception as e:
                            print(f"Error inserting into feedback table: {str(e)}")
                    else:
                        # If no similar insight, insert into both tables
                        print(f"No similar insight found, inserting into feedback_insights table")
                        try:
                            # First insert into feedback_insights
                            feedback_insights_data = {
                                "workspace_id": workspace_id,
                                "theme": insight.get("Theme"),
                                "insight": insight.get("Insight"),
                                "product_area": insight.get("Product_Area"),
                                "feature_recommendation": insight.get("Feature_Recommendation"),
                                "feedback": insight.get("Verbatim"),
                                "embeddings": embeddings,
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "source": source_type
                            }
                            
                            feedback_response = await asyncio.to_thread(
                                lambda: supabase.table("feedback_insights")
                                                .insert(feedback_insights_data)
                                                .execute()
                            )
                            
                            if feedback_response.data and len(feedback_response.data) > 0:
                                feedback_insight_id = feedback_response.data[0].get("feedback_insights_id")
                                insight["feedback_insight_id"] = feedback_insight_id

                            print(f"Inserted into feedback_insights table with id: {feedback_insight_id}")
                            insight["source"] = source_type
                            unique_insights.append(insight) 
                        except Exception as e:
                            print(f"Error inserting into tables: {str(e)}")

   

            try:
                print(f"Starting to insert conversational insights for workspace: {workspace_id}")
                print(f"Number of unique insights to process: {len(unique_insights)}")
                
                clips = await insert_uploaded_recording_conversational_insights(
                    workspace_id,
                    title,
                    unique_insights,
                )
                
                print(f"Successfully inserted insights. Received {len(clips)} clips to process")
                
            except Exception as e:
                print(f"Error inserting conversational insights: {str(e)}")
                if hasattr(e, '__traceback__'):
                    import traceback
                    print(f"Full traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Error updating conversational insights: {str(e)}"
                )

            try:
                print(f"Starting to process {len(clips)} clips")
                clip_result = await process_clips(workspace_id, temp_video_file_path, clips, title)
                print(f"Successfully processed clips. Result: {clip_result}")
                
            except Exception as e:
                print(f"Error processing clips: {str(e)}")
                if hasattr(e, '__traceback__'):
                    import traceback
                    print(f"Full traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Error updating clips: {str(e)}"
                )
            finally:
                # Clean up the temporary video file
                if os.path.exists(temp_video_file_path):
                    try:
                        os.remove(temp_video_file_path)
                        print(f"Temporary video file deleted: {temp_video_file_path}")
                    except Exception as e:
                        print(f"Error deleting temporary video file: {e}")

            return {"status": "success", "message": "Conversational insights processed successfully."}

    except Exception as e:
        print(f"Error in analyze_uploaded_recording: {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            print(f"Full traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process recording: {str(e)}"
        )
    
    finally:
        # Clean up the temporary video file
        if temp_video_file_path and os.path.exists(temp_video_file_path):
            try:
                os.remove(temp_video_file_path)
                print(f"Temporary video file deleted: {temp_video_file_path}")
            except Exception as e:
                print(f"Error deleting temporary video file: {e}")


