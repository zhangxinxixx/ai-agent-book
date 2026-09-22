"""参考 SandboxFusion 实现的多语言代码执行器。"""

import asyncio
import subprocess
import sys
import tempfile
import os
import shutil
import time
import base64
import psutil
import shlex
from typing import Dict, Any, Optional, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)

_DOCKER_ACTIVE: Optional[bool] = None


def is_docker_active() -> bool:
    """检查 Docker 守护进程是否真正可用（不仅检查可执行文件是否存在）。"""
    global _DOCKER_ACTIVE
    if _DOCKER_ACTIVE is not None:
        return _DOCKER_ACTIVE
    if not shutil.which("docker"):
        _DOCKER_ACTIVE = False
        return False
    try:
        res = subprocess.run(["docker", "info"], capture_output=True, timeout=2)
        _DOCKER_ACTIVE = (res.returncode == 0)
    except Exception:
        _DOCKER_ACTIVE = False
    return _DOCKER_ACTIVE


def try_decode(s: bytes) -> str:
    """将子进程字节流安全解码为字符串，无法解码的字符使用替代符。"""
    try:
        return s.decode('utf-8', errors='replace')
    except Exception as e:
        return f'[解码错误] {e}'


async def get_all_output(stream) -> str:
    """持续读取流直到 EOF；应在进程退出或被终止后完成。"""
    if stream is None:
        return ""
    try:
        result = await stream.read()
        return try_decode(result)
    except Exception as e:
        logger.debug(f"读取子进程输出失败：{e}")
        return ""


def kill_process_tree(pid: int):
    """终止指定进程及其全部子进程，避免超时后残留后台任务。"""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        
        # 先终止子进程，避免父进程退出后留下孤儿进程。
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        
        # 最后终止父进程。
        try:
            parent.kill()
        except psutil.NoSuchProcess:
            pass
            
    except psutil.NoSuchProcess:
        pass
    except Exception as e:
        logger.warning(f'终止进程树失败：{e}')


class ExecutionStatus(str, Enum):
    """执行状态枚举；字符串值会直接出现在工具返回结果中。"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"


class LanguageExecutor:
    """按语言路由执行方法，并统一返回状态、输出与耗时。"""
    
    def __init__(self, workspace_dir: str = None):
        """初始化执行器并记录默认工作区。"""
        self.workspace_dir = workspace_dir or os.getcwd()
    
    async def execute_code(
        self,
        code: str,
        language: str,
        timeout: float = 30.0,
        compile_timeout: float = 10.0,
        stdin: Optional[str] = None,
        files: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """执行指定语言代码，并返回统一的结构化结果。

        `timeout` 控制运行阶段，`compile_timeout` 控制编译型语言的编译阶段；
        `files` 用于把附加文件一并写入本次临时工作目录。
        """
        if language is None:
            language = "python"
        language = language.lower()
        
        # 将语言名及常用别名映射到具体执行方法。
        executors = {
            'python': self._run_python,
            'python3': self._run_python,
            'javascript': self._run_javascript,
            'js': self._run_javascript,
            'typescript': self._run_typescript,
            'ts': self._run_typescript,
            'go': self._run_go,
            'java': self._run_java,
            'cpp': self._run_cpp,
            'c++': self._run_cpp,
            'rust': self._run_rust,
            'php': self._run_php,
            'bash': self._run_bash,
            'shell': self._run_bash,
            'sh': self._run_bash,
            'nodejs': self._run_javascript,
            'node': self._run_javascript,
        }
        
        executor = executors.get(language)
        if not executor:
            return {
                "status": ExecutionStatus.ERROR,
                "error": f"不支持的语言：{language}。支持：{', '.join(sorted(set(executors.keys())))}"
            }
        
        try:
            return await executor(code, timeout, compile_timeout, stdin, files or {})
        except Exception as e:
            logger.exception(f"执行 {language} 代码时发生异常")
            return {
                "status": ExecutionStatus.ERROR,
                "error": f"执行失败：{str(e)}"
            }
    
    async def _run_command(
        self,
        command: str,
        timeout: float,
        stdin: Optional[str] = None,
        cwd: Optional[str] = None,
        shell: bool = True
    ) -> Dict[str, Any]:
        """运行 Shell 命令，同时管理输出管道、超时和进程树。"""
        process = None
        try:
            logger.debug(f'正在执行命令：{command[:100]}...')
            
            process = await asyncio.create_subprocess_shell(
                command,
                stdin=asyncio.subprocess.PIPE if stdin else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                executable='/bin/bash' if sys.platform != 'win32' else None
            )
            
            # 如提供 stdin，则写入子进程标准输入并关闭管道。
            if stdin and process.stdin:
                try:
                    process.stdin.write(stdin.encode())
                    await process.stdin.drain()
                    process.stdin.close()
                except Exception as e:
                    logger.warning(f"写入子进程 stdin 失败：{e}")
            
            start_time = time.time()

            # 必须在等待进程的同时并发读取 stdout 与 stderr。若先 wait()、退出后再读，
            # 子进程可能写满系统管道缓冲区并被阻塞，从而制造假的超时。
            stdout_task = asyncio.ensure_future(get_all_output(process.stdout))
            stderr_task = asyncio.ensure_future(get_all_output(process.stderr))

            try:
                # 在超时边界内等待子进程结束。
                await asyncio.wait_for(process.wait(), timeout=timeout)
                execution_time = time.time() - start_time

                stdout = await stdout_task
                stderr = await stderr_task

                logger.debug(f'命令执行完成，耗时 {execution_time:.2f} 秒')
                
                return {
                    "status": ExecutionStatus.SUCCESS if process.returncode == 0 else ExecutionStatus.FAILED,
                    "returncode": process.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "execution_time": execution_time
                }
                
            except asyncio.TimeoutError:
                execution_time = time.time() - start_time

                # 超时时先终止进程树，使管道关闭，再读取残余输出。
                if psutil.pid_exists(process.pid):
                    kill_process_tree(process.pid)
                    logger.info(f'进程 {process.pid} 因超时被终止')

                stdout = await stdout_task
                stderr = await stderr_task
                
                return {
                    "status": ExecutionStatus.TIMEOUT,
                    "error": f"执行超过 {timeout} 秒，已超时终止",
                    "stdout": stdout,
                    "stderr": stderr,
                    "execution_time": execution_time
                }
                
        except Exception as e:
            logger.exception(f"运行命令时发生异常：{command[:100]}")
            return {
                "status": ExecutionStatus.ERROR,
                "error": f"命令执行失败：{str(e)}"
            }
        finally:
            # 兜底清理：确保异常路径不会遗留子进程。
            if process and psutil.pid_exists(process.pid):
                kill_process_tree(process.pid)
    
    def _write_files(self, tmp_dir: str, files: Dict[str, str]):
        """把附加文件写入本次执行的临时目录。"""
        for filename, content in files.items():
            if not content or "IGNORE_THIS_FILE" in filename:
                continue
                
            filepath = os.path.join(tmp_dir, filename)
            dirpath = os.path.dirname(filepath)
            
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            
            # 自动识别 base64 内容；否则按 UTF-8 文本写入。
            try:
                if self._is_base64(content):
                    with open(filepath, 'wb') as f:
                        f.write(base64.b64decode(content))
                else:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
            except Exception as e:
                logger.warning(f"写入附加文件 {filename} 失败：{e}")
    
    def _is_base64(self, s: str) -> bool:
        """保守判断字符串是否为合法 base64。"""
        try:
            if len(s) % 4 != 0 or not all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in s):
                return False
            base64.b64decode(s, validate=True)
            return True
        except Exception:
            return False
    
    async def _run_python(
        self,
        code: str,
        timeout: float,
        compile_timeout: float,
        stdin: Optional[str],
        files: Dict[str, str]
    ) -> Dict[str, Any]:
        """执行 Python：优先 Docker 隔离，不可用时降级为本地子进程。"""
        with tempfile.TemporaryDirectory(prefix='python_', ignore_cleanup_errors=True) as tmp_dir:
            self._write_files(tmp_dir, files)
            code_file = os.path.join(tmp_dir, 'main.py')
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # Docker 可用时，把不可信 Python 放入真正的容器边界：禁用网络、根文件系统
            # 只读、限制内存/CPU/进程数，并且只挂载一个可写的临时工作目录。
            if is_docker_active():
                mount = shlex.quote(f"{tmp_dir}:/workspace:rw")
                command = (
                    "docker run --rm --network none --memory 256m --cpus 1 "
                    "--pids-limit 64 --read-only "
                    "--tmpfs /tmp:rw,nosuid,nodev,noexec,size=16m "
                    f"-v {mount} -w /workspace python:3.11-slim "
                    "python -I -B -u main.py"
                )
                result = await self._run_command(command, timeout, stdin, tmp_dir)
                result["sandbox"] = {
                    "kind": "docker",
                    "image": "python:3.11-slim",
                    "network": "none",
                    "rootfs": "read-only",
                    "memory": "256m",
                    "cpus": 1,
                    "pids_limit": 64,
                }
            else:
                py_cmd = f'"{sys.executable}" -I -B -u "{code_file}"'
                result = await self._run_command(
                    py_cmd,
                    timeout,
                    stdin,
                    tmp_dir
                )
                result["sandbox"] = {"kind": "local-process", "degraded": True}
            result['language'] = 'python'
            return result
    
    async def _run_javascript(
        self,
        code: str,
        timeout: float,
        compile_timeout: float,
        stdin: Optional[str],
        files: Dict[str, str]
    ) -> Dict[str, Any]:
        """使用 Node.js 执行 JavaScript。"""
        with tempfile.TemporaryDirectory(prefix='js_', ignore_cleanup_errors=True) as tmp_dir:
            self._write_files(tmp_dir, files)
            
            # 未提供 package.json 时创建最小配置，以启用 ES Module。
            if 'package.json' not in files:
                package_json = {
                    "type": "module",
                    "dependencies": {}
                }
                with open(os.path.join(tmp_dir, 'package.json'), 'w') as f:
                    import json
                    json.dump(package_json, f)
            
            code_file = os.path.join(tmp_dir, 'main.js')
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            result = await self._run_command(
                f'node {code_file}',
                timeout,
                stdin,
                tmp_dir
            )
            result['language'] = 'javascript'
            return result
    
    async def _run_typescript(
        self,
        code: str,
        timeout: float,
        compile_timeout: float,
        stdin: Optional[str],
        files: Dict[str, str]
    ) -> Dict[str, Any]:
        """使用 tsx 或 ts-node 执行 TypeScript。"""
        with tempfile.TemporaryDirectory(prefix='ts_', ignore_cleanup_errors=True) as tmp_dir:
            self._write_files(tmp_dir, files)
            code_file = os.path.join(tmp_dir, 'main.ts')
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # 优先使用 tsx，不可用时回退到 ts-node。
            check_tsx = await self._run_command('which tsx 2>/dev/null', 1.0)
            cmd = 'tsx' if check_tsx['status'] == ExecutionStatus.SUCCESS else 'ts-node'
            
            result = await self._run_command(
                f'{cmd} {code_file}',
                timeout,
                stdin,
                tmp_dir
            )
            result['language'] = 'typescript'
            return result
    
    async def _run_go(
        self,
        code: str,
        timeout: float,
        compile_timeout: float,
        stdin: Optional[str],
        files: Dict[str, str]
    ) -> Dict[str, Any]:
        """先编译再执行 Go 代码。"""
        with tempfile.TemporaryDirectory(prefix='go_', ignore_cleanup_errors=True) as tmp_dir:
            self._write_files(tmp_dir, files)
            
            # 初始化 Go Module；已存在时忽略错误。
            await self._run_command('go mod init main 2>/dev/null || true', 2.0, cwd=tmp_dir)
            
            code_file = os.path.join(tmp_dir, 'main.go')
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # 编译阶段使用独立超时。
            compile_result = await self._run_command(
                'go build -o main main.go',
                compile_timeout,
                cwd=tmp_dir
            )
            
            if compile_result['status'] != ExecutionStatus.SUCCESS:
                return {
                    "status": ExecutionStatus.FAILED,
                    "language": "go",
                    "phase": "compilation",
                    "returncode": compile_result.get('returncode', 1),
                    "stdout": compile_result.get('stdout', ''),
                    "stderr": compile_result.get('stderr', ''),
                    "error": "Go 编译失败"
                }
            
            # 运行编译产物。
            result = await self._run_command(
                './main',
                timeout,
                stdin,
                tmp_dir
            )
            result['language'] = 'go'
            result['compile_stdout'] = compile_result.get('stdout', '')
            result['compile_stderr'] = compile_result.get('stderr', '')
            return result
    
    async def _run_java(
        self,
        code: str,
        timeout: float,
        compile_timeout: float,
        stdin: Optional[str],
        files: Dict[str, str]
    ) -> Dict[str, Any]:
        """提取公共类名，编译并执行 Java 代码。"""
        with tempfile.TemporaryDirectory(prefix='java_', ignore_cleanup_errors=True) as tmp_dir:
            self._write_files(tmp_dir, files)
            
            # 从 public class 声明中提取类名，默认使用 Main。
            class_name = 'Main'
            import re
            match = re.search(r'public\s+class\s+(\w+)', code)
            if match:
                class_name = match.group(1)
            
            code_file = os.path.join(tmp_dir, f'{class_name}.java')
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # 将附加 jar 加入 classpath。
            jars = [f for f in files.keys() if f.endswith('.jar')]
            classpath = '.:' + ':'.join(jars) if jars else '.'
            
            # 编译阶段使用独立超时。
            compile_result = await self._run_command(
                f'javac -cp {classpath} {class_name}.java',
                compile_timeout,
                cwd=tmp_dir
            )
            
            if compile_result['status'] != ExecutionStatus.SUCCESS:
                return {
                    "status": ExecutionStatus.FAILED,
                    "language": "java",
                    "phase": "compilation",
                    "returncode": compile_result.get('returncode', 1),
                    "stdout": compile_result.get('stdout', ''),
                    "stderr": compile_result.get('stderr', ''),
                    "error": "Java 编译失败"
                }
            
            # 启用断言后运行。
            result = await self._run_command(
                f'java -cp {classpath} -ea {class_name}',
                timeout,
                stdin,
                tmp_dir
            )
            result['language'] = 'java'
            result['compile_stdout'] = compile_result.get('stdout', '')
            result['compile_stderr'] = compile_result.get('stderr', '')
            return result
    
    async def _run_cpp(
        self,
        code: str,
        timeout: float,
        compile_timeout: float,
        stdin: Optional[str],
        files: Dict[str, str]
    ) -> Dict[str, Any]:
        """使用 C++17 编译并执行 C++ 代码。"""
        with tempfile.TemporaryDirectory(prefix='cpp_', ignore_cleanup_errors=True) as tmp_dir:
            self._write_files(tmp_dir, files)
            code_file = os.path.join(tmp_dir, 'main.cpp')
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # 使用常见编译参数，并按代码内容补充可选链接库。
            compile_flags = '-std=c++17 -O2'
            optional_libs = []
            
            # 使用线程库时追加 pthread。
            if '#include <thread>' in code or 'std::thread' in code:
                optional_libs.append('-lpthread')
            
            libs = ' '.join(optional_libs)
            compile_result = await self._run_command(
                f'g++ {compile_flags} main.cpp -o main {libs}',
                compile_timeout,
                cwd=tmp_dir
            )
            
            if compile_result['status'] != ExecutionStatus.SUCCESS:
                return {
                    "status": ExecutionStatus.FAILED,
                    "language": "cpp",
                    "phase": "compilation",
                    "returncode": compile_result.get('returncode', 1),
                    "stdout": compile_result.get('stdout', ''),
                    "stderr": compile_result.get('stderr', ''),
                    "error": "C++ 编译失败"
                }
            
            # 运行编译产物。
            result = await self._run_command(
                './main',
                timeout,
                stdin,
                tmp_dir
            )
            result['language'] = 'cpp'
            result['compile_stdout'] = compile_result.get('stdout', '')
            result['compile_stderr'] = compile_result.get('stderr', '')
            return result
    
    async def _run_rust(
        self,
        code: str,
        timeout: float,
        compile_timeout: float,
        stdin: Optional[str],
        files: Dict[str, str]
    ) -> Dict[str, Any]:
        """使用优化模式编译并执行 Rust 代码。"""
        with tempfile.TemporaryDirectory(prefix='rust_', ignore_cleanup_errors=True) as tmp_dir:
            self._write_files(tmp_dir, files)
            code_file = os.path.join(tmp_dir, 'main.rs')
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # 使用优化模式编译。
            compile_result = await self._run_command(
                'rustc -O main.rs -o main',
                compile_timeout,
                cwd=tmp_dir
            )
            
            if compile_result['status'] != ExecutionStatus.SUCCESS:
                return {
                    "status": ExecutionStatus.FAILED,
                    "language": "rust",
                    "phase": "compilation",
                    "returncode": compile_result.get('returncode', 1),
                    "stdout": compile_result.get('stdout', ''),
                    "stderr": compile_result.get('stderr', ''),
                    "error": "Rust 编译失败"
                }
            
            # 运行编译产物。
            result = await self._run_command(
                './main',
                timeout,
                stdin,
                tmp_dir
            )
            result['language'] = 'rust'
            result['compile_stdout'] = compile_result.get('stdout', '')
            result['compile_stderr'] = compile_result.get('stderr', '')
            return result
    
    async def _run_php(
        self,
        code: str,
        timeout: float,
        compile_timeout: float,
        stdin: Optional[str],
        files: Dict[str, str]
    ) -> Dict[str, Any]:
        """补齐 PHP 标签后执行 PHP 代码。"""
        with tempfile.TemporaryDirectory(prefix='php_', ignore_cleanup_errors=True) as tmp_dir:
            self._write_files(tmp_dir, files)
            
            # 用户未提供 PHP 起始标签时自动补齐。
            code_clean = code.strip()
            if not code_clean.startswith('<?php') and not code_clean.startswith('<?'):
                code = '<?php\n' + code
            
            code_file = os.path.join(tmp_dir, 'main.php')
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            result = await self._run_command(
                f'php -f {code_file}',
                timeout,
                stdin,
                tmp_dir
            )
            result['language'] = 'php'
            return result
    
    async def _run_bash(
        self,
        code: str,
        timeout: float,
        compile_timeout: float,
        stdin: Optional[str],
        files: Dict[str, str]
    ) -> Dict[str, Any]:
        """写入临时脚本并使用 Bash 执行。"""
        with tempfile.TemporaryDirectory(prefix='bash_', ignore_cleanup_errors=True) as tmp_dir:
            self._write_files(tmp_dir, files)
            code_file = os.path.join(tmp_dir, 'script.sh')
            with open(code_file, 'w', encoding='utf-8') as f:
                # 用户未提供 shebang 时自动补齐。
                if not code.startswith('#!'):
                    f.write('#!/bin/bash\n')
                f.write(code)
            
            # 增加当前用户可执行权限。
            os.chmod(code_file, 0o755)
            
            result = await self._run_command(
                f'bash {code_file}',
                timeout,
                stdin,
                tmp_dir
            )
            result['language'] = 'bash'
            return result
