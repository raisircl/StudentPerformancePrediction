from pathlib import Path
from typing_extensions import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

print(f"Project Root Path: {PROJECT_ROOT}")

#d:\code\python\notes\projects\StudentPerformancePrediction