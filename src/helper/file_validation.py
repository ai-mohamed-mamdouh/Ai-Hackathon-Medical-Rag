# file is_valid ???
from .file_helper import FileHelper
from src.config.settings import settings
from fastapi import HTTPException, UploadFile, status

def validate_file(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File name is required.",
        )

    extension = FileHelper.get_file_extension(file)

    if extension not in settings.ALLOWED_EXTENSIONS :
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: .{extension}",
        )