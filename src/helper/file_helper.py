import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile


class FileHelper:
    TEMP_DIR = Path("temp")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_file_extension(file: UploadFile) -> str:
        """
        Get the uploaded file extension.
        """
        extension = Path(file.filename).suffix.lower()
        return extension.lstrip(".")

    @staticmethod
    def save_temp_file(file: UploadFile) -> str:
        """
        Save uploaded file temporarily using its original filename.
        """
        filename = Path(file.filename).name

        temp_path = FileHelper.TEMP_DIR / filename

        with open(temp_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)

        return str(temp_path)