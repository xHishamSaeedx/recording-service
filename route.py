from fastapi import APIRouter, HTTPException, Request, Depends
from service import analyze_uploaded_recording, check_file_mime_type
from fastapi.responses import JSONResponse
import httpx

router = APIRouter()

@router.post("/v1/api/process-recording")
async def process_recording(
    request: Request,
):
    try:
        # Get parameters from request body
        payload = await request.json()
        gcs_url = payload.get('gcs_url')
        workspace_id = payload.get('workspace_id')
        title = payload.get('title')
        source_type = payload.get('source_type', 'gcs')

        # Validate required parameters
        if not all([gcs_url, workspace_id, title]):
            raise HTTPException(
                status_code=400, 
                detail="Missing required parameters. Please provide gcs_url, workspace_id, and title."
            )
        
        # Check file MIME type
        # mime_type = await check_file_mime_type(gcs_url)
        # if not mime_type or not mime_type.lower().startswith('video/mp4'):
        #     raise HTTPException(
        #         status_code=400,
        #         detail=f"Invalid file type. Expected video/mp4, got {mime_type}"
        #     )
        
        process_result = await analyze_uploaded_recording(
            workspace_id=workspace_id,
            gcs_url=gcs_url,
            title=title,
            source_type=source_type
        )

        return JSONResponse(content={
            "status": "completed",
            "process_result": process_result,
        })

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
