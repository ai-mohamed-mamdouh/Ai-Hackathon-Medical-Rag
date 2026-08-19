import logging
import hashlib
from pathlib import Path
from typing import Any
from fastapi import APIRouter, File, HTTPException,Path as PathParam, Request, UploadFile, status
from pydantic import BaseModel
from src.helper.file_validation import validate_file
from src.indexing import DocumentProcessor
from src.helper.file_helper import FileHelper

logger = logging.getLogger(__name__)

indexing_router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

class UpdateResponse(BaseModel):
    message: str

class UploadResponse(BaseModel):
    file_name: str
    file_id: str
    file_hash_content: str

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
        path = Path(temp_path)
        file_name = path.name
        file_id = hashlib.sha256(path.name.encode("utf-8")).hexdigest()
        file_hash_content = hashlib.sha256(path.read_bytes()).hexdigest()

        
        message = DocumentProcessor().document_processing_pipeline(
            path=temp_path,
            vector_store=vector_store
        )

        return UploadResponse(file_name=file_name,
                            file_id=file_id,
                            file_hash_content=file_hash_content)

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


@indexing_router.put("/update/{file_id}", response_model=UpdateResponse)
def update_document(
    request: Request,
    file_id: str = PathParam(..., description="The ID of the file to update"),
    file: UploadFile = File(...),
) -> UpdateResponse:
    temp_path: str | None = None
    vector_store = request.app.state.vector_store
    vector_store = vector_store.get_vector_store()

    try:
        # Validate file type/size using your validation helper
        validate_file(file)

        # Save uploaded file temporarily
        temp_path = FileHelper.save_temp_file(file)

        # Execute document update pipeline
        message = DocumentProcessor().update_document(
            file_id=file_id, new_file_path=temp_path, vector_store=vector_store
        )

        return UpdateResponse(message=message)

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Failed to update document ID '%s' with file: %s",
            file_id,
            file.filename,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update document with ID {file_id}.",
        )

    finally:
        # Clean up file handle and temporary disk resource
        file.file.close()

        if temp_path:
            Path(temp_path).unlink(missing_ok=True)