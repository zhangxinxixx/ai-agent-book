"""Google Calendar 与 GitHub 外部系统集成工具。"""

import os
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

from llm_helper import LLMHelper
from config import Config

# Google Calendar 为可选依赖，缺失时只阻塞对应工具。
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

# GitHub 为可选依赖，缺失时只阻塞对应工具。
try:
    from github import Github, GithubException
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False


class ExternalTools:
    """封装会对真实外部系统产生写入的工具。"""
    
    def __init__(self, llm_helper: LLMHelper):
        """注入审批用 LLMHelper；外部客户端仍采用惰性初始化。"""
        self.llm_helper = llm_helper
        self._google_service = None
        self._github_client = None
    
    def _get_google_calendar_service(self):
        """获取或创建 Google Calendar 客户端。"""
        if not GOOGLE_AVAILABLE:
            raise ImportError("Google Calendar libraries not installed")
        
        if self._google_service:
            return self._google_service
        
        SCOPES = ['https://www.googleapis.com/auth/calendar']
        creds = None
        
        token_path = Path('token.json')
        creds_path = Path(Config.GOOGLE_CALENDAR_CREDENTIALS_FILE)
        
        # 优先读取已经存在的本地授权令牌。
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        
        # 令牌无效时刷新，无法刷新时启动新的授权流程。
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not creds_path.exists():
                    raise FileNotFoundError(f"Credentials file not found: {creds_path}")
                flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
                creds = flow.run_local_server(port=0)
            
            # 保存新令牌，供后续调用复用。
            token_path.write_text(creds.to_json(), encoding="utf-8")
        
        self._google_service = build('calendar', 'v3', credentials=creds)
        return self._google_service
    
    def _get_github_client(self):
        """获取或创建 GitHub 客户端。"""
        if not GITHUB_AVAILABLE:
            raise ImportError("GitHub library not installed")
        
        if self._github_client:
            return self._github_client
        
        if not Config.GITHUB_TOKEN:
            raise ValueError("GitHub token not configured")
        
        self._github_client = Github(Config.GITHUB_TOKEN)
        return self._github_client
    
    async def google_calendar_add(
        self,
        summary: str,
        start_time: str,
        end_time: str,
        description: Optional[str] = None,
        location: Optional[str] = None
    ) -> Dict[str, Any]:
        """校验时间、完成审批后，向 Google Calendar 创建真实事件。"""
        try:
            service = self._get_google_calendar_service()
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to initialize Google Calendar: {str(e)}"
            }
        
        # 将字符串解析为 datetime。
        try:
            start_dt = self._parse_datetime(start_time)
            end_dt = self._parse_datetime(end_time)
        except ValueError as e:
            return {
                "success": False,
                "error": f"Invalid datetime format: {str(e)}"
            }
        
        # 结束时间必须晚于开始时间。
        if end_dt <= start_dt:
            return {
                "success": False,
                "error": "End time must be after start time"
            }
        
        # 外部写入前请求审批。
        if Config.REQUIRE_APPROVAL_FOR_DANGEROUS_OPS:
            approved, reason = self.llm_helper.request_approval(
                "google_calendar_add",
                {
                    "summary": summary,
                    "start_time": start_dt.isoformat(),
                    "end_time": end_dt.isoformat(),
                    "description": description
                }
            )
            
            if not approved:
                return {
                    "success": False,
                    "error": f"Calendar event creation not approved: {reason}"
                }
        
        # 组装 Google Calendar API 所需事件对象。
        event = {
            'summary': summary,
            'start': {
                'dateTime': start_dt.isoformat(),
                'timeZone': 'UTC'
            },
            'end': {
                'dateTime': end_dt.isoformat(),
                'timeZone': 'UTC'
            }
        }
        
        if description:
            event['description'] = description
        if location:
            event['location'] = location
        
        try:
            created_event = service.events().insert(
                calendarId='primary',
                body=event
            ).execute()
            
            return {
                "success": True,
                "event_id": created_event['id'],
                "event_link": created_event.get('htmlLink'),
                "summary": summary,
                "start_time": start_dt.isoformat(),
                "end_time": end_dt.isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create calendar event: {str(e)}"
            }
    
    async def github_create_pr(
        self,
        repo_name: str,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main"
    ) -> Dict[str, Any]:
        """校验仓库与分支、完成审批后，创建或复用 GitHub PR。"""
        try:
            github = self._get_github_client()
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to initialize GitHub client: {str(e)}"
            }
        
        # 仓库名必须使用 owner/repo 格式。
        if '/' not in repo_name:
            return {
                "success": False,
                "error": "Repository name must be in format: owner/repo"
            }
        
        # GitHub 外部写入前请求审批。
        if Config.REQUIRE_APPROVAL_FOR_DANGEROUS_OPS:
            approved, reason = self.llm_helper.request_approval(
                "github_create_pr",
                {
                    "repo": repo_name,
                    "title": title,
                    "head": head_branch,
                    "base": base_branch,
                    "body_preview": body[:200]
                }
            )
            
            if not approved:
                return {
                    "success": False,
                    "error": f"PR creation not approved: {reason}"
                }
        
        try:
            # 获取目标仓库。
            repo = github.get_repo(repo_name)
            
            # 校验源分支和目标分支都存在。
            try:
                repo.get_branch(head_branch)
                repo.get_branch(base_branch)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Branch verification failed: {str(e)}"
                }

            # 写入前查询相同 head/base 的打开 PR，使重试具备幂等性，不会重复创建。
            owner = repo_name.split("/", 1)[0]
            existing = repo.get_pulls(
                state="open", head=f"{owner}:{head_branch}", base=base_branch
            )
            for pr in existing:
                if pr.head.ref == head_branch and pr.base.ref == base_branch:
                    return {
                        "success": True,
                        "pr_number": pr.number,
                        "pr_url": pr.html_url,
                        "title": pr.title,
                        "state": pr.state,
                        "created_at": pr.created_at.isoformat(),
                        "idempotent_reuse": True,
                    }
            
            # 未命中已有 PR 时才真正创建。
            pr = repo.create_pull(
                title=title,
                body=body,
                head=head_branch,
                base=base_branch
            )
            
            return {
                "success": True,
                "pr_number": pr.number,
                "pr_url": pr.html_url,
                "title": title,
                "state": pr.state,
                "created_at": pr.created_at.isoformat(),
                "idempotent_reuse": False,
            }
            
        except GithubException as e:
            return {
                "success": False,
                "error": f"GitHub API error: {e.data.get('message', str(e))}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create pull request: {str(e)}"
            }
    
    def _parse_datetime(self, time_str: str) -> datetime:
        """按多种常见格式解析日期时间字符串。"""
        # 优先尝试 ISO 8601。
        formats = [
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%Y-%m-%d'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        
        # 所有格式都失败时明确报错。
        raise ValueError(f"Could not parse datetime: {time_str}")
