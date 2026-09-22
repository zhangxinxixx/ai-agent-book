"""带目录导航和文件编辑能力的终端控制器。

实现参考 AWorld terminal-controller。所有目录与文件方法都先经过工作区边界检查；
`execute_command()` 仅固定命令起始目录，并不构成操作系统级沙箱。
"""
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Dict, Any

from config import Config


class TerminalController:
    """维护当前目录、命令历史，并提供受工作区限制的文件操作。"""
    
    def __init__(self):
        self.workspace_dir = Path(Config.WORKSPACE_DIR).resolve()
        self.current_directory = self.workspace_dir
        self.command_history = []
        self.max_history = 100
    
    def _is_safe_path(self, path: Path) -> bool:
        """判断解析后的路径是否仍位于工作区内。"""
        try:
            path.resolve().relative_to(self.workspace_dir)
            return True
        except ValueError:
            return False
    
    def _resolve_path(self, path: str) -> Path:
        """将相对路径按当前目录解析为绝对路径。"""
        path_obj = Path(path)
        if not path_obj.is_absolute():
            path_obj = self.current_directory / path_obj
        return path_obj.resolve()
    
    async def execute_command(
        self,
        command: str,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """在当前目录执行 Shell 命令，并返回 stdout、stderr 与退出码。"""
        try:
            # 添加到历史记录
            self.command_history.append(command)
            if len(self.command_history) > self.max_history:
                self.command_history.pop(0)
            
            # 教材示例使用类 Unix 的 `ls`。Windows 的 shell=True 实际调用 cmd.exe，
            # 因此把完全匹配的只读命令映射为 `dir`，其余命令保持原样。
            platform_command = command
            if sys.platform == "win32" and command.strip() == "ls":
                platform_command = "dir"

            # 真正执行命令。cwd 只是起始目录，不是文件系统隔离。
            result = subprocess.run(
                platform_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.current_directory)
            )
            
            return {
                "success": result.returncode == 0,
                "command": command,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "cwd": str(self.current_directory)
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"命令执行超过 {timeout} 秒，已超时终止",
                "command": command
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"命令执行失败：{str(e)}",
                "command": command
            }
    
    async def get_current_directory(self) -> Dict[str, Any]:
        """返回当前目录和工作区根目录。"""
        return {
            "success": True,
            "current_directory": str(self.current_directory),
            "workspace": str(self.workspace_dir)
        }
    
    async def change_directory(
        self,
        directory: str
    ) -> Dict[str, Any]:
        """切换当前目录；目标必须存在且位于工作区内。"""
        try:
            new_dir = self._resolve_path(directory)
            
            if not self._is_safe_path(new_dir):
                return {
                    "success": False,
                    "error": f"目录 {directory} 位于工作区之外"
                }
            
            if not new_dir.exists():
                return {
                    "success": False,
                    "error": f"目录 {directory} 不存在（does not exist）"
                }
            
            if not new_dir.is_dir():
                return {
                    "success": False,
                    "error": f"{directory} 不是目录"
                }
            
            self.current_directory = new_dir
            
            return {
                "success": True,
                "current_directory": str(self.current_directory),
                "message": f"已切换到 {directory}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"切换目录失败：{str(e)}"
            }
    
    async def list_directory(
        self,
        directory: str = "."
    ) -> Dict[str, Any]:
        """列出目录内容；相对路径以当前目录为基准。"""
        try:
            dir_path = self._resolve_path(directory)
            
            if not self._is_safe_path(dir_path):
                return {
                    "success": False,
                    "error": f"目录 {directory} 位于工作区之外"
                }
            
            if not dir_path.exists():
                return {
                    "success": False,
                    "error": f"目录 {directory} 不存在"
                }
            
            contents = []
            for item in sorted(dir_path.iterdir()):
                contents.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": 0 if item.is_dir() else item.stat().st_size
                })
            
            return {
                "success": True,
                "directory": str(dir_path),
                "contents": contents,
                "count": len(contents)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"列出目录失败：{str(e)}"
            }
    
    async def read_file(
        self,
        file_path: str,
        encoding: str = "utf-8"
    ) -> Dict[str, Any]:
        """读取工作区内的文本文件。"""
        try:
            resolved_path = self._resolve_path(file_path)
            
            if not self._is_safe_path(resolved_path):
                return {
                    "success": False,
                    "error": f"文件 {file_path} 位于工作区之外"
                }
            
            if not resolved_path.exists():
                return {
                    "success": False,
                    "error": f"文件 {file_path} 不存在"
                }
            
            content = resolved_path.read_text(encoding=encoding)
            
            return {
                "success": True,
                "file_path": str(resolved_path),
                "content": content,
                "size": len(content),
                "lines": len(content.splitlines())
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"读取文件失败：{str(e)}"
            }
    
    async def write_file(
        self,
        file_path: str,
        content: str,
        mode: str = "w"
    ) -> Dict[str, Any]:
        """写入工作区内的文件；`mode='a'` 表示追加，其余模式覆盖。"""
        try:
            resolved_path = self._resolve_path(file_path)
            
            if not self._is_safe_path(resolved_path):
                return {
                    "success": False,
                    "error": f"文件 {file_path} 位于工作区之外"
                }
            
            # 必要时创建父目录。
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 真正执行文件写入。
            if mode == "a":
                with open(resolved_path, 'a', encoding='utf-8') as f:
                    f.write(content)
            else:
                resolved_path.write_text(content, encoding='utf-8')
            
            return {
                "success": True,
                "file_path": str(resolved_path),
                "bytes_written": len(content),
                "mode": mode
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"写入文件失败：{str(e)}"
            }
    
    async def insert_file_content(
        self,
        file_path: str,
        content: str,
        line_number: int
    ) -> Dict[str, Any]:
        """在指定行前插入内容；`line_number` 从 1 开始计数。"""
        try:
            resolved_path = self._resolve_path(file_path)
            
            if not self._is_safe_path(resolved_path):
                return {
                    "success": False,
                    "error": f"文件 {file_path} 位于工作区之外"
                }
            
            if not resolved_path.exists():
                return {
                    "success": False,
                    "error": f"文件 {file_path} 不存在"
                }
            
            # 读取当前内容，并按行处理。
            lines = resolved_path.read_text(encoding="utf-8").splitlines()
            
            # 先校验行号，再执行插入。
            if line_number < 1 or line_number > len(lines) + 1:
                return {
                    "success": False,
                    "error": f"行号 {line_number} 超出范围（1-{len(lines)+1}）"
                }
            
            lines.insert(line_number - 1, content)
            
            # 将修改后的全部内容写回文件。
            resolved_path.write_text('\n'.join(lines) + '\n', encoding="utf-8")
            
            return {
                "success": True,
                "file_path": str(resolved_path),
                "line_number": line_number,
                "total_lines": len(lines)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"插入内容失败：{str(e)}"
            }
    
    async def delete_file_content(
        self,
        file_path: str,
        start_line: int,
        end_line: int
    ) -> Dict[str, Any]:
        """删除闭区间 `[start_line, end_line]` 内的文件行。"""
        try:
            resolved_path = self._resolve_path(file_path)
            
            if not self._is_safe_path(resolved_path):
                return {
                    "success": False,
                    "error": f"文件 {file_path} 位于工作区之外"
                }
            
            if not resolved_path.exists():
                return {
                    "success": False,
                    "error": f"文件 {file_path} 不存在"
                }
            
            # 读取全部行。
            lines = resolved_path.read_text(encoding="utf-8").splitlines()
            
            # 校验闭区间是否合法。
            if start_line < 1 or end_line > len(lines) or start_line > end_line:
                return {
                    "success": False,
                    "error": f"无效行区间：{start_line}-{end_line}（文件共 {len(lines)} 行）"
                }
            
            # Python 切片右端不包含，因此这里直接使用 end_line。
            del lines[start_line - 1:end_line]
            
            # 写回剩余内容。
            resolved_path.write_text('\n'.join(lines) + '\n' if lines else '', encoding="utf-8")
            
            return {
                "success": True,
                "file_path": str(resolved_path),
                "deleted_lines": end_line - start_line + 1,
                "remaining_lines": len(lines)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"删除内容失败：{str(e)}"
            }
    
    async def update_file_content(
        self,
        file_path: str,
        line_number: int,
        new_content: str
    ) -> Dict[str, Any]:
        """替换指定行，并在结果中同时返回修改前后的内容。"""
        try:
            resolved_path = self._resolve_path(file_path)
            
            if not self._is_safe_path(resolved_path):
                return {
                    "success": False,
                    "error": f"文件 {file_path} 位于工作区之外"
                }
            
            if not resolved_path.exists():
                return {
                    "success": False,
                    "error": f"文件 {file_path} 不存在"
                }
            
            # 读取并校验目标行。
            lines = resolved_path.read_text(encoding="utf-8").splitlines()
            
            if line_number < 1 or line_number > len(lines):
                return {
                    "success": False,
                    "error": f"行号 {line_number} 超出范围（1-{len(lines)}）"
                }
            
            # 保存旧值后替换目标行。
            old_content = lines[line_number - 1]
            lines[line_number - 1] = new_content
            
            # 将修改后的全部内容写回。
            resolved_path.write_text('\n'.join(lines) + '\n', encoding="utf-8")
            
            return {
                "success": True,
                "file_path": str(resolved_path),
                "line_number": line_number,
                "old_content": old_content,
                "new_content": new_content
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"更新内容失败：{str(e)}"
            }
    
    async def get_command_history(
        self,
        count: int = 10
    ) -> Dict[str, Any]:
        """返回最近的命令历史；`count<=0` 时返回空列表。"""
        # `history[-0:]` 会返回整个列表，因此 count<=0 必须单独处理。
        if count <= 0 or not self.command_history:
            recent = []
        else:
            recent = self.command_history[-count:]
        
        return {
            "success": True,
            "history": recent,
            "count": len(recent),
            "total": len(self.command_history)
        }
