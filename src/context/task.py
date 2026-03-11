"""
任务上下文管理
管理完整任务的生命周期，包括图片、中间结果、报告等
使用文件存储（JSON + Markdown）
"""
import os
import json
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from config import get

logger = logging.getLogger(__name__)


class TaskContext:
    """任务上下文"""

    def __init__(self, task_id: str, base_dir: Optional[Path] = None):
        self.task_id = task_id
        self.base_dir = base_dir or Path(get('context.task_dir', './tasks'))
        self.task_dir = self.base_dir / task_id
        self.meta_file = self.task_dir / 'meta.json'
        self.history_file = self.task_dir / 'history.md'
        self.images_dir = self.task_dir / 'images'
        self.results_dir = self.task_dir / 'results'

    def exists(self) -> bool:
        """检查任务是否存在"""
        return self.task_dir.exists()

    def create(self, metadata: Dict = None) -> 'TaskContext':
        """创建新任务"""
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

        # 创建元数据
        meta = {
            'task_id': self.task_id,
            'created_at': datetime.now().isoformat(),
            'status': 'created',
            'metadata': metadata or {},
            'steps': []
        }
        self._save_meta(meta)

        # 创建历史文件
        self.history_file.write_text(f"# Task {self.task_id}\n\n", encoding='utf-8')

        logger.info(f"创建任务: {self.task_id}")
        return self

    def load(self) -> Optional[Dict]:
        """加载任务"""
        if not self.exists():
            return None
        return self._load_meta()

    def _load_meta(self) -> Dict:
        """加载元数据"""
        if self.meta_file.exists():
            return json.loads(self.meta_file.read_text(encoding='utf-8'))
        return {}

    def _save_meta(self, meta: Dict):
        """保存元数据"""
        self.meta_file.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )

    def update_status(self, status: str):
        """更新任务状态"""
        meta = self._load_meta()
        meta['status'] = status
        meta['updated_at'] = datetime.now().isoformat()
        self._save_meta(meta)

    def add_step(self, step_name: str, step_data: Dict):
        """添加步骤"""
        meta = self._load_meta()
        step = {
            'name': step_name,
            'data': step_data,
            'timestamp': datetime.now().isoformat()
        }
        meta.setdefault('steps', []).append(step)
        meta['updated_at'] = datetime.now().isoformat()
        self._save_meta(meta)

        # 追加到历史
        self._append_history(f"## Step: {step_name}\n\n{json.dumps(step_data, ensure_ascii=False, indent=2)}\n\n")

    def add_image(self, image_name: str, image_data: bytes) -> str:
        """
        添加图片

        Args:
            image_name: 图片文件名
            image_data: 图片二进制数据

        Returns:
            图片保存路径
        """
        image_path = self.images_dir / image_name
        image_path.write_bytes(image_data)
        logger.debug(f"保存图片: {self.task_id}/{image_name}")
        return str(image_path)

    def add_result(self, result_name: str, content: str):
        """添加结果文件"""
        result_path = self.results_dir / result_name
        result_path.write_text(content, encoding='utf-8')
        logger.debug(f"保存结果: {self.task_id}/{result_name}")

    def append_history(self, content: str):
        """追加历史记录"""
        self._append_history(content)

    def _append_history(self, content: str):
        """追加历史（Markdown 格式）"""
        if self.history_file.exists():
            existing = self.history_file.read_text(encoding='utf-8')
        else:
            existing = f"# Task {self.task_id}\n\n"

        self.history_file.write_text(existing + content, encoding='utf-8')

    def generate_report(self) -> str:
        """生成最终报告"""
        meta = self._load_meta()

        report = f"""# 病理分析报告

## 任务信息
- 任务ID: {meta.get('task_id', 'N/A')}
- 创建时间: {meta.get('created_at', 'N/A')}
- 状态: {meta.get('status', 'N/A')}
- 更新时间: {meta.get('updated_at', 'N/A')}

## 元数据
{json.dumps(meta.get('metadata', {}), ensure_ascii=False, indent=2)}

## 执行步骤

"""
        for i, step in enumerate(meta.get('steps', []), 1):
            report += f"### {i}. {step['name']}\n"
            report += f"时间: {step['timestamp']}\n\n"
            report += f"```json\n{json.dumps(step['data'], ensure_ascii=False, indent=2)}\n```\n\n"

        # 附加完整历史
        if self.history_file.exists():
            report += "\n## 详细日志\n\n"
            report += self.history_file.read_text(encoding='utf-8')

        # 保存报告
        report_path = self.results_dir / 'report.md'
        report_path.write_text(report, encoding='utf-8')

        logger.info(f"生成报告: {self.task_id}/report.md")
        return str(report_path)

    def delete(self):
        """删除任务"""
        if self.task_dir.exists():
            shutil.rmtree(self.task_dir)
            logger.info(f"删除任务: {self.task_id}")

    def get_images(self) -> List[str]:
        """获取所有图片"""
        if not self.images_dir.exists():
            return []
        return [f.name for f in self.images_dir.iterdir() if f.is_file()]

    def get_results(self) -> List[str]:
        """获取所有结果文件"""
        if not self.results_dir.exists():
            return []
        return [f.name for f in self.results_dir.iterdir() if f.is_file()]


class TaskManager:
    """任务管理器"""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(get('context.task_dir', './tasks'))
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_task(self, metadata: Dict = None) -> TaskContext:
        """创建新任务"""
        task_id = str(uuid.uuid4())[:12]  # 短 ID
        return TaskContext(task_id, self.base_dir).create(metadata)

    def get_task(self, task_id: str) -> Optional[TaskContext]:
        """获取任务"""
        task = TaskContext(task_id, self.base_dir)
        return task if task.exists() else None

    def list_tasks(self, status: Optional[str] = None) -> List[Dict]:
        """列出所有任务"""
        tasks = []
        for task_dir in self.base_dir.iterdir():
            if not task_dir.is_dir():
                continue

            meta_file = task_dir / 'meta.json'
            if not meta_file.exists():
                continue

            meta = json.loads(meta_file.read_text(encoding='utf-8'))

            if status and meta.get('status') != status:
                continue

            tasks.append({
                'task_id': task_dir.name,
                'status': meta.get('status'),
                'created_at': meta.get('created_at'),
                'updated_at': meta.get('updated_at')
            })

        return sorted(tasks, key=lambda x: x.get('created_at', ''), reverse=True)

    def cleanup_old_tasks(self, days: int = 30) -> int:
        """清理旧任务"""
        import time
        cutoff = time.time() - (days * 86400)
        count = 0

        for task_dir in self.base_dir.iterdir():
            if not task_dir.is_dir():
                continue

            if task_dir.stat().st_mtime < cutoff:
                shutil.rmtree(task_dir)
                count += 1

        if count > 0:
            logger.info(f"清理了 {count} 个旧任务")

        return count


# 全局任务管理器
_task_manager = None


def get_task_manager() -> TaskManager:
    """获取任务管理器实例"""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager
