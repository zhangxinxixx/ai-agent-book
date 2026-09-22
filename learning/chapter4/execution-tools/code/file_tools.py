"""具备多重安全防护机制的文件系统工具集。"""

import itertools
import os
from pathlib import Path
from typing import Dict, Any
from llm_helper import LLMHelper
from config import Config


class FileTools:
    """文件系统工具：包含工作区路径隔离、覆盖审批、自动语法验证与 Diff 预览。"""
    
    def __init__(self, llm_helper: LLMHelper):
        """初始化文件工具，注入 LLM 辅助类并绑定工作区路径。"""
        self.llm_helper = llm_helper
        self.workspace_dir = Config.WORKSPACE_DIR
    
    def _resolve_path(self, path: str) -> Path:
        """将文件路径解析为工作目录下的路径（支持相对路径与绝对路径）。"""
        path_obj = Path(path)
        if not path_obj.is_absolute():
            path_obj = self.workspace_dir / path_obj
        return path_obj.resolve()
    
    def _is_safe_path(self, path: Path) -> bool:
        """安全路径检查：确保目标路径严格在工作目录内，防止目录遍历攻击（Path Traversal）。"""
        try:
            path.resolve().relative_to(self.workspace_dir.resolve())
            return True
        except ValueError:
            return False
    
    async def write_file(
        self,
        path: str,
        content: str,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """带安全校验与语法检查的文件写入工具。

        参数:
            path: 文件路径（相对工作区或绝对路径）
            content: 要写入的文件内容
            overwrite: 是否允许覆盖已存在文件

        返回:
            包含执行状态、写入路径、写入字节数与校验状态的结果字典
        """
        resolved_path = self._resolve_path(path)
        
        # 1. 路径安全审查：确保操作局限在沙盒工作区内，禁止逃逸
        if not self._is_safe_path(resolved_path):
            return {
                "success": False,
                "error": f"路径 {path} 超出受限工作区范围，拒绝访问"
            }
        
        # 2. 覆盖保护机制：若文件已存在且未显式开启 overwrite，触发事前审批
        if resolved_path.exists() and not overwrite:
            if Config.REQUIRE_APPROVAL_FOR_DANGEROUS_OPS:
                approved, reason = self.llm_helper.request_approval(
                    "file_overwrite",
                    {
                        "path": str(resolved_path),
                        "existing_size": resolved_path.stat().st_size,
                        "new_content_size": len(content)
                    }
                )
                
                if not approved:
                    return {
                        "success": False,
                        "error": f"文件覆盖未获批准：{reason}"
                    }
        
        # 3. 语法自动验证（Fail-safe 机制）：如果是代码文件，写入磁盘前先行校验语法
        if Config.AUTO_VERIFY_CODE and resolved_path.suffix in ['.py', '.js', '.ts']:
            language = {'.py': 'python', '.js': 'javascript', '.ts': 'typescript'}[resolved_path.suffix]
            is_valid, error_msg = self.llm_helper.verify_code_syntax(content, language)
            
            # 若代码存在语法错误，立即拦截并把语法错误信息反馈给 Agent，以便其自我纠错
            if not is_valid:
                return {
                    "success": False,
                    "error": f"语法校验失败：{error_msg}",
                    "verification": "failed"
                }
        
        # 4. 执行写入并返回结构化结果
        try:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_path.write_text(content, encoding="utf-8")
            
            return {
                "success": True,
                "path": str(resolved_path),
                "bytes_written": len(content),
                "verification": "passed" if Config.AUTO_VERIFY_CODE else "skipped"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"文件写入异常: {str(e)}"
            }
    
    async def edit_file(
        self,
        path: str,
        search: str,
        replace: str
    ) -> Dict[str, Any]:
        """按“搜索-替换”编辑文件，附带语法校验与差异（Diff）预览。

        参数:
            path: 目标文件路径
            search: 待替换的原始文本字符串
            replace: 用于替换的新文本字符串

        返回:
            包含执行成功标志、修改路径、Diff 预览及语法校验状态的字典
        """
        resolved_path = self._resolve_path(path)
        
        # 1. 路径安全性检查
        if not self._is_safe_path(resolved_path):
            return {
                "success": False,
                "error": f"路径 {path} 超出受限工作区范围，拒绝操作"
            }
        
        # 2. 文件存在性校验
        if not resolved_path.exists():
            return {
                "success": False,
                "error": f"目标文件 {path} 不存在"
            }
        
        # 3. 读取当前文件内容
        try:
            current_content = resolved_path.read_text(encoding="utf-8")
        except Exception as e:
            return {
                "success": False,
                "error": f"读取文件失败: {str(e)}"
            }

        # 4. 空搜索文本防御：空字符串匹配任意位置，拒绝执行以防意外插入
        if search == "":
            return {
                "success": False,
                "error": "搜索文本不能为空（empty search）"
            }
        
        # 5. 校验目标匹配项是否存在
        if search not in current_content:
            return {
                "success": False,
                "error": f"文件中未找到指定的搜索文本"
            }
        
        # 6. 执行单次精准替换
        new_content = current_content.replace(search, replace, 1)
        
        # 7. 生成行级差异对比（Diff Preview），供 Agent 直观观察修改效果
        diff_preview = self._generate_diff(current_content, new_content)
        
        # 8. 修改后代码语法验证：防止修改后引入语法断裂
        if Config.AUTO_VERIFY_CODE and resolved_path.suffix in ['.py', '.js', '.ts']:
            language = {'.py': 'python', '.js': 'javascript', '.ts': 'typescript'}[resolved_path.suffix]
            is_valid, error_msg = self.llm_helper.verify_code_syntax(new_content, language)
            
            if not is_valid:
                return {
                    "success": False,
                    "error": f"编辑后语法校验失败: {error_msg}",
                    "diff_preview": diff_preview
                }
        
        # 9. 将修改后的内容写回磁盘
        try:
            resolved_path.write_text(new_content, encoding="utf-8")
            
            return {
                "success": True,
                "path": str(resolved_path),
                "diff_preview": diff_preview,
                "verification": "passed" if Config.AUTO_VERIFY_CODE else "skipped"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"保存修改内容失败: {str(e)}"
            }
    
    def _generate_diff(self, old_content: str, new_content: str) -> str:
        """生成简明直观的行级 Diff 差异对比（限制前 20 行）。"""
        old_lines = old_content.split('\n')
        new_lines = new_content.split('\n')
        
        diff_lines = []
        for i, (old, new) in enumerate(itertools.zip_longest(old_lines, new_lines, fillvalue=''), 1):
            if old != new:
                diff_lines.append(f"第 {i} 行:")
                diff_lines.append(f"  - {old}")
                diff_lines.append(f"  + {new}")
        
        return '\n'.join(diff_lines[:20])  # 上下文中最多截取 20 行差异
