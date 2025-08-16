#!/usr/bin/env python3
"""
System Cleanup Script for Four-Brain Project
Prevents storage bloat by cleaning up temporary files, old models, and redundant artifacts.
"""

import os
import shutil
import logging
import argparse
from pathlib import Path
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemCleaner:
    """Comprehensive system cleaner for the Four-Brain project."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.total_cleaned = 0
        
    def get_directory_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        total = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total += os.path.getsize(filepath)
                    except (OSError, FileNotFoundError):
                        pass
        except (OSError, FileNotFoundError):
            pass
        return total
    
    def format_size(self, size_bytes: int) -> str:
        """Format size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}PB"
    
    def clean_directory(self, path: Path, description: str) -> int:
        """Clean a directory and return bytes freed."""
        if not path.exists():
            logger.info(f"⏭️  {description}: Directory doesn't exist")
            return 0
            
        size_before = self.get_directory_size(path)
        if size_before == 0:
            logger.info(f"⏭️  {description}: Already clean")
            return 0
            
        try:
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ {description}: Cleaned {self.format_size(size_before)}")
            return size_before
        except Exception as e:
            logger.error(f"❌ {description}: Error cleaning - {e}")
            return 0
    
    def clean_files_by_pattern(self, directory: Path, patterns: List[str], description: str) -> int:
        """Clean files matching patterns."""
        if not directory.exists():
            return 0
            
        total_freed = 0
        files_cleaned = 0
        
        for pattern in patterns:
            for file_path in directory.rglob(pattern):
                if file_path.is_file():
                    try:
                        size = file_path.stat().st_size
                        file_path.unlink()
                        total_freed += size
                        files_cleaned += 1
                    except Exception as e:
                        logger.warning(f"⚠️  Could not remove {file_path}: {e}")
        
        if files_cleaned > 0:
            logger.info(f"✅ {description}: Removed {files_cleaned} files, freed {self.format_size(total_freed)}")
        else:
            logger.info(f"⏭️  {description}: No files to clean")
            
        return total_freed
    
    def clean_temporary_files(self) -> int:
        """Clean temporary files and directories."""
        logger.info("🧹 Cleaning temporary files...")
        
        total_freed = 0
        
        # Clean tmp directory
        tmp_dir = self.project_root / "tmp"
        total_freed += self.clean_directory(tmp_dir, "tmp/ directory")
        
        # Clean cache directories
        cache_dir = self.project_root / "cache"
        total_freed += self.clean_directory(cache_dir, "cache/ directory")
        
        # Clean Python cache
        total_freed += self.clean_files_by_pattern(
            self.project_root, 
            ["__pycache__", "*.pyc", "*.pyo"], 
            "Python cache files"
        )
        
        # Clean log files (keep recent ones)
        logs_dir = self.project_root / "logs"
        if logs_dir.exists():
            total_freed += self.clean_files_by_pattern(
                logs_dir,
                ["*.log.1", "*.log.2", "*.log.3", "*.log.old"],
                "Old log files"
            )
        
        return total_freed
    
    def clean_model_artifacts(self) -> int:
        """Clean old model artifacts and build files."""
        logger.info("🤖 Cleaning model artifacts...")
        
        total_freed = 0
        
        # Clean TensorRT engines (keep only latest)
        patterns = [
            "*.plan.old",
            "*.plan.backup", 
            "*.engine.old",
            "*_old.plan",
            "*_backup.plan"
        ]
        
        models_dir = self.project_root / "models"
        if models_dir.exists():
            total_freed += self.clean_files_by_pattern(
                models_dir,
                patterns,
                "Old TensorRT engines"
            )
        
        # Clean ONNX intermediate files
        onnx_patterns = [
            "*.onnx.tmp",
            "*_intermediate.onnx",
            "*_temp.onnx"
        ]
        
        total_freed += self.clean_files_by_pattern(
            self.project_root,
            onnx_patterns,
            "Temporary ONNX files"
        )
        
        return total_freed
    
    def clean_docker_artifacts(self) -> int:
        """Clean Docker build artifacts."""
        logger.info("🐳 Cleaning Docker artifacts...")
        
        total_freed = 0
        
        # Clean Docker build context
        docker_dirs = [
            self.project_root / "containers" / "**" / ".dockerignore.tmp",
            self.project_root / "containers" / "**" / "Dockerfile.tmp"
        ]
        
        for pattern in docker_dirs:
            total_freed += self.clean_files_by_pattern(
                self.project_root,
                [str(pattern)],
                "Docker temporary files"
            )
        
        return total_freed
    
    def clean_node_modules(self, force: bool = False) -> int:
        """Clean Node.js modules (with confirmation)."""
        node_modules = self.project_root / "qwen-code" / "node_modules"
        
        if not node_modules.exists():
            return 0
            
        size = self.get_directory_size(node_modules)
        
        if not force:
            logger.info(f"📦 Found node_modules: {self.format_size(size)}")
            logger.info("   Use --force-node-modules to clean (can be reinstalled with npm install)")
            return 0
        
        return self.clean_directory(node_modules, "node_modules")
    
    def analyze_large_files(self, min_size_mb: int = 100) -> List[Tuple[Path, int]]:
        """Find large files for manual review."""
        logger.info(f"🔍 Analyzing files larger than {min_size_mb}MB...")
        
        large_files = []
        min_size_bytes = min_size_mb * 1024 * 1024
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip certain directories
            dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', '__pycache__']]
            
            for file in files:
                file_path = Path(root) / file
                try:
                    size = file_path.stat().st_size
                    if size > min_size_bytes:
                        large_files.append((file_path, size))
                except (OSError, FileNotFoundError):
                    pass
        
        # Sort by size (largest first)
        large_files.sort(key=lambda x: x[1], reverse=True)
        
        if large_files:
            logger.info(f"📊 Found {len(large_files)} large files:")
            for file_path, size in large_files[:10]:  # Show top 10
                rel_path = file_path.relative_to(self.project_root)
                logger.info(f"   {self.format_size(size)}: {rel_path}")
        else:
            logger.info("✅ No large files found")
        
        return large_files
    
    def run_cleanup(self, force_node_modules: bool = False, analyze_only: bool = False) -> None:
        """Run comprehensive cleanup."""
        logger.info("🚀 Starting Four-Brain System Cleanup")
        logger.info(f"📁 Project root: {self.project_root}")
        
        if analyze_only:
            self.analyze_large_files()
            return
        
        total_freed = 0
        
        # Clean temporary files
        total_freed += self.clean_temporary_files()
        
        # Clean model artifacts
        total_freed += self.clean_model_artifacts()
        
        # Clean Docker artifacts
        total_freed += self.clean_docker_artifacts()
        
        # Clean node_modules if requested
        total_freed += self.clean_node_modules(force_node_modules)
        
        # Analyze remaining large files
        self.analyze_large_files()
        
        logger.info(f"🎉 Cleanup Complete!")
        logger.info(f"💾 Total space freed: {self.format_size(total_freed)}")
        
        if total_freed > 1024 * 1024 * 1024:  # > 1GB
            logger.info("🚀 Significant space recovered! System should run more efficiently.")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Clean up Four-Brain system to prevent storage bloat")
    parser.add_argument("--force-node-modules", action="store_true", 
                       help="Also clean node_modules (can be reinstalled)")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze large files, don't clean")
    parser.add_argument("--project-root", default=".",
                       help="Project root directory (default: current directory)")
    
    args = parser.parse_args()
    
    cleaner = SystemCleaner(args.project_root)
    cleaner.run_cleanup(args.force_node_modules, args.analyze_only)

if __name__ == "__main__":
    main()
