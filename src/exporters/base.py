"""
내보내기 기본 클래스
"""

from abc import ABC, abstractmethod
from pathlib import Path

from ..core.builder import DocumentContent


class BaseExporter(ABC):
    """내보내기 기본 클래스"""

    @property
    @abstractmethod
    def format_name(self) -> str:
        """출력 형식 이름"""
        pass

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """파일 확장자"""
        pass

    @abstractmethod
    def export(self, content: DocumentContent, output_path: Path) -> Path:
        """문서 내보내기"""
        pass
