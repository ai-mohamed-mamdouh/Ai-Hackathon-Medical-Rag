import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel
from src.helper.file_validation import validate_file
from src.indexing import DocumentProcessor
from src.helper.file_helper import FileHelper


logger = logging.getLogger(__name__)

indexing_router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

class UploadResponse(BaseModel):
    message: str

@indexing_router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_document(
    request: Request,
    file: UploadFile = File(...),
) -> UploadResponse:
    temp_path: str | None = None
    vector_store = request.app.state.vector_store

    try:
        validate_file(file)

        temp_path = FileHelper.save_temp_file(file)

        
        message = DocumentProcessor().document_processing_pipeline(
            path=temp_path,
            vector_store=vector_store
        )

        return UploadResponse(message=message)

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Failed to process document: %s",
            file.filename,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process document.",
        )

    finally:
        file.file.close()

        if temp_path:
            Path(temp_path).unlink(missing_ok=True)