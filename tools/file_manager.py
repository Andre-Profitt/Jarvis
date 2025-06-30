"""
File Manager Tool for JARVIS
============================

Provides comprehensive file and directory management capabilities.
"""

import os
import shutil
import asyncio
import aiofiles
import json
import yaml
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import mimetypes
import hashlib
import stat

from .base import BaseTool, ToolMetadata, ToolCategory


class FileManagerTool(BaseTool):
    """
    Advanced file management tool with async operations

    Features:
    - Read/write files with various formats
    - Directory operations (create, delete, list)
    - File search and filtering
    - Metadata extraction
    - Safe operations with validation
    - Archive support (zip, tar)
    """

    def __init__(self, base_path: Optional[str] = None):
        metadata = ToolMetadata(
            name="file_manager",
            description="Manage files and directories with advanced operations",
            category=ToolCategory.FILE,
            version="1.0.0",
            tags=["file", "directory", "storage", "management"],
            required_permissions=["file_read", "file_write"],
            rate_limit=None,  # No rate limit for file operations
            timeout=60,
            examples=[
                {
                    "description": "Read a text file",
                    "params": {"operation": "read", "path": "/path/to/file.txt"},
                },
                {
                    "description": "List directory contents",
                    "params": {
                        "operation": "list",
                        "path": "/path/to/directory",
                        "recursive": True,
                    },
                },
                {
                    "description": "Create a directory",
                    "params": {
                        "operation": "create_dir",
                        "path": "/path/to/new/directory",
                    },
                },
            ],
        )
        super().__init__(metadata)

        # Set base path for operations (sandbox)
        self.base_path = Path(base_path) if base_path else Path.cwd()

        # Supported file formats for structured reading
        self.supported_formats = {
            ".json": self._read_json,
            ".yaml": self._read_yaml,
            ".yml": self._read_yaml,
            ".csv": self._read_csv,
            ".txt": self._read_text,
            ".md": self._read_text,
            ".py": self._read_text,
            ".log": self._read_text,
        }

    def validate_inputs(self, **kwargs) -> tuple[bool, Optional[str]]:
        """Validate file operation parameters"""
        operation = kwargs.get("operation")

        if not operation:
            return False, "Operation parameter is required"

        valid_operations = [
            "read",
            "write",
            "append",
            "delete",
            "copy",
            "move",
            "list",
            "create_dir",
            "delete_dir",
            "exists",
            "info",
            "search",
            "archive",
            "extract",
        ]

        if operation not in valid_operations:
            return False, f"Invalid operation: {operation}"

        path = kwargs.get("path")
        if not path:
            return False, "Path parameter is required"

        # Validate path is within base_path (security)
        try:
            resolved_path = self._resolve_path(path)
            if not self._is_safe_path(resolved_path):
                return False, "Path is outside allowed directory"
        except Exception as e:
            return False, f"Invalid path: {str(e)}"

        # Operation-specific validation
        if operation in ["write", "append"]:
            if "content" not in kwargs:
                return False, "Content parameter is required for write/append"

        if operation in ["copy", "move"]:
            if "destination" not in kwargs:
                return False, "Destination parameter is required for copy/move"

        return True, None

    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """Execute file operation"""
        operation = kwargs.get("operation")
        path = self._resolve_path(kwargs.get("path"))

        operations_map = {
            "read": self._handle_read,
            "write": self._handle_write,
            "append": self._handle_append,
            "delete": self._handle_delete,
            "copy": self._handle_copy,
            "move": self._handle_move,
            "list": self._handle_list,
            "create_dir": self._handle_create_dir,
            "delete_dir": self._handle_delete_dir,
            "exists": self._handle_exists,
            "info": self._handle_info,
            "search": self._handle_search,
            "archive": self._handle_archive,
            "extract": self._handle_extract,
        }

        handler = operations_map.get(operation)
        if not handler:
            raise ValueError(f"Unknown operation: {operation}")

        return await handler(path, **kwargs)

    async def _handle_read(self, path: Path, **kwargs) -> Dict[str, Any]:
        """Read file contents"""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        # Determine file format
        suffix = path.suffix.lower()
        reader = self.supported_formats.get(suffix, self._read_text)

        content = await reader(path)

        return {
            "operation": "read",
            "path": str(path),
            "content": content,
            "size": path.stat().st_size,
            "format": suffix,
            "encoding": kwargs.get("encoding", "utf-8"),
        }

    async def _handle_write(self, path: Path, **kwargs) -> Dict[str, Any]:
        """Write content to file"""
        content = kwargs.get("content")
        encoding = kwargs.get("encoding", "utf-8")
        create_dirs = kwargs.get("create_dirs", True)

        # Create parent directories if needed
        if create_dirs and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        # Determine format and write accordingly
        suffix = path.suffix.lower()

        if suffix == ".json" and isinstance(content, (dict, list)):
            async with aiofiles.open(path, "w", encoding=encoding) as f:
                await f.write(json.dumps(content, indent=2))
        elif suffix in [".yaml", ".yml"] and isinstance(content, (dict, list)):
            async with aiofiles.open(path, "w", encoding=encoding) as f:
                await f.write(yaml.dump(content, default_flow_style=False))
        else:
            # Write as text
            async with aiofiles.open(path, "w", encoding=encoding) as f:
                await f.write(str(content))

        return {
            "operation": "write",
            "path": str(path),
            "size": path.stat().st_size,
            "created": not path.exists(),
        }

    async def _handle_append(self, path: Path, **kwargs) -> Dict[str, Any]:
        """Append content to file"""
        content = kwargs.get("content")
        encoding = kwargs.get("encoding", "utf-8")

        if not path.exists():
            # Create file if it doesn't exist
            return await self._handle_write(path, **kwargs)

        async with aiofiles.open(path, "a", encoding=encoding) as f:
            await f.write(str(content))

        return {"operation": "append", "path": str(path), "size": path.stat().st_size}

    async def _handle_delete(self, path: Path, **kwargs) -> Dict[str, Any]:
        """Delete a file"""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        # Get file info before deletion
        size = path.stat().st_size

        path.unlink()

        return {"operation": "delete", "path": str(path), "size": size, "deleted": True}

    async def _handle_copy(self, path: Path, **kwargs) -> Dict[str, Any]:
        """Copy file or directory"""
        destination = self._resolve_path(kwargs.get("destination"))

        if not path.exists():
            raise FileNotFoundError(f"Source not found: {path}")

        if path.is_file():
            shutil.copy2(path, destination)
        else:
            shutil.copytree(path, destination)

        return {
            "operation": "copy",
            "source": str(path),
            "destination": str(destination),
            "is_directory": path.is_dir(),
        }

    async def _handle_move(self, path: Path, **kwargs) -> Dict[str, Any]:
        """Move file or directory"""
        destination = self._resolve_path(kwargs.get("destination"))

        if not path.exists():
            raise FileNotFoundError(f"Source not found: {path}")

        shutil.move(str(path), str(destination))

        return {
            "operation": "move",
            "source": str(path),
            "destination": str(destination),
            "is_directory": path.is_dir(),
        }

    async def _handle_list(self, path: Path, **kwargs) -> Dict[str, Any]:
        """List directory contents"""
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        recursive = kwargs.get("recursive", False)
        pattern = kwargs.get("pattern", "*")
        include_hidden = kwargs.get("include_hidden", False)

        files = []

        if recursive:
            for item in path.rglob(pattern):
                if include_hidden or not item.name.startswith("."):
                    files.append(self._get_file_info(item))
        else:
            for item in path.glob(pattern):
                if include_hidden or not item.name.startswith("."):
                    files.append(self._get_file_info(item))

        # Sort by type (directories first) then by name
        files.sort(key=lambda x: (not x["is_directory"], x["name"]))

        return {
            "operation": "list",
            "path": str(path),
            "count": len(files),
            "files": files,
            "recursive": recursive,
        }

    async def _handle_create_dir(self, path: Path, **kwargs) -> Dict[str, Any]:
        """Create directory"""
        parents = kwargs.get("parents", True)
        exist_ok = kwargs.get("exist_ok", True)

        path.mkdir(parents=parents, exist_ok=exist_ok)

        return {"operation": "create_dir", "path": str(path), "created": True}

    async def _handle_delete_dir(self, path: Path, **kwargs) -> Dict[str, Any]:
        """Delete directory"""
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        force = kwargs.get("force", False)

        if force or not any(path.iterdir()):
            shutil.rmtree(path)
            deleted = True
        else:
            raise ValueError("Directory is not empty. Use force=True to delete.")

        return {"operation": "delete_dir", "path": str(path), "deleted": deleted}

    async def _handle_exists(self, path: Path, **kwargs) -> Dict[str, Any]:
        """Check if path exists"""
        return {
            "operation": "exists",
            "path": str(path),
            "exists": path.exists(),
            "is_file": path.is_file() if path.exists() else None,
            "is_directory": path.is_dir() if path.exists() else None,
        }

    async def _handle_info(self, path: Path, **kwargs) -> Dict[str, Any]:
        """Get detailed file/directory information"""
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        stat_info = path.stat()

        info = {
            "operation": "info",
            "path": str(path),
            "name": path.name,
            "is_file": path.is_file(),
            "is_directory": path.is_dir(),
            "size": stat_info.st_size,
            "created": datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            "accessed": datetime.fromtimestamp(stat_info.st_atime).isoformat(),
            "permissions": oct(stat_info.st_mode),
            "owner": stat_info.st_uid,
            "group": stat_info.st_gid,
        }

        if path.is_file():
            # Add file-specific info
            info["mime_type"] = mimetypes.guess_type(str(path))[0]
            info["extension"] = path.suffix

            # Calculate hash for small files
            if stat_info.st_size < 10 * 1024 * 1024:  # 10MB
                info["md5"] = await self._calculate_hash(path)

        return info

    async def _handle_search(self, path: Path, **kwargs) -> Dict[str, Any]:
        """Search for files"""
        pattern = kwargs.get("pattern", "*")
        content_search = kwargs.get("content_search")
        max_results = kwargs.get("max_results", 100)

        if not path.is_dir():
            raise ValueError(f"Path must be a directory for search: {path}")

        results = []

        for item in path.rglob(pattern):
            if len(results) >= max_results:
                break

            if item.is_file():
                match_info = {"path": str(item), "name": item.name}

                # Content search if requested
                if content_search and item.suffix in [
                    ".txt",
                    ".py",
                    ".json",
                    ".yaml",
                    ".md",
                    ".log",
                ]:
                    try:
                        content = await self._read_text(item)
                        if content_search.lower() in content.lower():
                            # Find line numbers with matches
                            lines = content.split("\n")
                            matches = []
                            for i, line in enumerate(lines):
                                if content_search.lower() in line.lower():
                                    matches.append(
                                        {"line": i + 1, "content": line.strip()}
                                    )
                            match_info["matches"] = matches[:5]  # First 5 matches
                            results.append(match_info)
                    except:
                        pass
                else:
                    results.append(match_info)

        return {
            "operation": "search",
            "path": str(path),
            "pattern": pattern,
            "content_search": content_search,
            "count": len(results),
            "results": results,
        }

    async def _handle_archive(self, path: Path, **kwargs) -> Dict[str, Any]:
        """Create archive from files/directories"""
        output = self._resolve_path(kwargs.get("output", f"{path}.zip"))
        format = kwargs.get("format", "zip")

        if format == "zip":
            import zipfile

            with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
                if path.is_file():
                    zf.write(path, path.name)
                else:
                    for item in path.rglob("*"):
                        if item.is_file():
                            zf.write(item, item.relative_to(path.parent))
        else:
            # Use shutil for other formats
            base = output.with_suffix("")
            shutil.make_archive(str(base), format, str(path.parent), str(path.name))

        return {
            "operation": "archive",
            "source": str(path),
            "output": str(output),
            "format": format,
            "size": output.stat().st_size,
        }

    async def _handle_extract(self, path: Path, **kwargs) -> Dict[str, Any]:
        """Extract archive"""
        destination = self._resolve_path(kwargs.get("destination", path.parent))

        if path.suffix == ".zip":
            import zipfile

            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(destination)
        else:
            shutil.unpack_archive(str(path), str(destination))

        return {
            "operation": "extract",
            "archive": str(path),
            "destination": str(destination),
        }

    # Helper methods

    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """Resolve path relative to base_path"""
        p = Path(path)
        if not p.is_absolute():
            p = self.base_path / p
        return p.resolve()

    def _is_safe_path(self, path: Path) -> bool:
        """Check if path is within allowed base_path"""
        try:
            path.relative_to(self.base_path)
            return True
        except ValueError:
            return False

    def _get_file_info(self, path: Path) -> Dict[str, Any]:
        """Get basic file information"""
        stat = path.stat()
        return {
            "name": path.name,
            "path": str(path),
            "is_directory": path.is_dir(),
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }

    async def _read_text(self, path: Path) -> str:
        """Read text file"""
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            return await f.read()

    async def _read_json(self, path: Path) -> Any:
        """Read JSON file"""
        content = await self._read_text(path)
        return json.loads(content)

    async def _read_yaml(self, path: Path) -> Any:
        """Read YAML file"""
        content = await self._read_text(path)
        return yaml.safe_load(content)

    async def _read_csv(self, path: Path) -> List[Dict[str, Any]]:
        """Read CSV file"""
        rows = []
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            content = await f.read()
            reader = csv.DictReader(content.splitlines())
            for row in reader:
                rows.append(dict(row))
        return rows

    async def _calculate_hash(self, path: Path) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        async with aiofiles.open(path, "rb") as f:
            while chunk := await f.read(8192):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _get_parameter_documentation(self) -> Dict[str, Any]:
        """Document the parameters for this tool"""
        return {
            "operation": {
                "type": "string",
                "description": "The file operation to perform",
                "required": True,
                "enum": [
                    "read",
                    "write",
                    "append",
                    "delete",
                    "copy",
                    "move",
                    "list",
                    "create_dir",
                    "delete_dir",
                    "exists",
                    "info",
                    "search",
                    "archive",
                    "extract",
                ],
            },
            "path": {
                "type": "string",
                "description": "The file or directory path",
                "required": True,
            },
            "content": {
                "type": "any",
                "description": "Content for write/append operations",
                "required": False,
            },
            "destination": {
                "type": "string",
                "description": "Destination path for copy/move operations",
                "required": False,
            },
            "recursive": {
                "type": "boolean",
                "description": "Recursive operation for list/search",
                "required": False,
                "default": False,
            },
            "pattern": {
                "type": "string",
                "description": "File pattern for list/search operations",
                "required": False,
                "default": "*",
            },
        }


# Example usage
async def example_usage():
    """Example of using the FileManagerTool"""
    tool = FileManagerTool(base_path="./test_files")

    # Create a directory
    result = await tool.execute(operation="create_dir", path="test_directory")
    print(f"Created directory: {result.data}")

    # Write a file
    result = await tool.execute(
        operation="write",
        path="test_directory/test.json",
        content={
            "message": "Hello from JARVIS!",
            "timestamp": datetime.now().isoformat(),
        },
    )
    print(f"Wrote file: {result.data}")

    # List directory
    result = await tool.execute(operation="list", path="test_directory")
    print(f"Directory contents: {result.data}")


if __name__ == "__main__":
    asyncio.run(example_usage())
