#!/usr/bin/env python3
"""
一键业务初始化脚本

用法:
    python template/init_business.py --name 产品问答 --folder ./data/product_faq --config ./product_config.yaml

功能:
    1. 创建 Milvus 业务集合
    2. 加载配置文件覆盖默认参数
    3. 批量导入业务知识库文档
    4. 创建独立会话存储空间
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from template.config_loader import load_config, apply_config
from core.pipeline import process_folder
from core.vector_store import milvus_store
from utils.logger import logger


def init_business(
    business_name: str,
    folder_path: str,
    config_path: str = "",
):
    logger.info(f"=== 开始初始化业务: {business_name} ===")

    if config_path:
        config = load_config(config_path)
        apply_config(config)
        logger.info(f"已加载配置文件: {config_path}")
    else:
        logger.info("未指定配置文件, 使用默认参数")

    logger.info(f"检查/创建 Milvus 集合: {milvus_store.collection_name}")
    count_before = milvus_store.count(expr=f'business_type == "{business_name}"')
    logger.info(f"当前业务知识库条目: {count_before}")

    if folder_path:
        folder = Path(folder_path)
        if not folder.exists():
            logger.error(f"知识库文件夹不存在: {folder_path}")
            return False

        logger.info(f"开始导入知识库: {folder_path}")
        stats = process_folder(
            folder_path=str(folder),
            business_type=business_name,
        )
        logger.info(
            f"导入完成: {stats['total_files']} 文件, "
            f"{stats['total_chunks']} 分块"
        )

        failed = [
            f for f in stats["files"] if f["status"] != "success"
        ]
        if failed:
            logger.warning(f"失败的文档: {len(failed)} 个")
            for f in failed:
                logger.warning(f"  - {f['file']}: {f['status']}")

    count_after = milvus_store.count(expr=f'business_type == "{business_name}"')
    logger.info(f"入库后业务知识库条目: {count_after}")

    logger.info(f"=== 业务初始化完成: {business_name} ===")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="一键初始化新业务知识库"
    )
    parser.add_argument(
        "--name", type=str, required=True,
        help="业务名称 (如: 产品问答、售后支持)"
    )
    parser.add_argument(
        "--folder", type=str, default="",
        help="知识库文件夹路径 (支持 PDF/DOCX/MD/TXT)"
    )
    parser.add_argument(
        "--config", type=str, default="",
        help="业务 YAML 配置文件路径"
    )

    args = parser.parse_args()

    success = init_business(
        business_name=args.name,
        folder_path=args.folder,
        config_path=args.config,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
