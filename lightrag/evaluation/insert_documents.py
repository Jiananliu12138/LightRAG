#!/usr/bin/env python3
"""
Insert sample documents into LightRAG for evaluation
将样本文档插入到 LightRAG 知识库中，用于评估测试

Usage:
    python insert_documents.py
    python insert_documents.py --api-url http://localhost:9621
    python insert_documents.py --docs-dir ./sample_documents
"""

import argparse
import sys
from pathlib import Path

import httpx
from tqdm import tqdm


def insert_documents(api_url: str, docs_dir: Path, timeout: float = 300.0):
    """
    Insert all markdown documents from a directory into LightRAG
    
    Args:
        api_url: LightRAG API base URL
        docs_dir: Directory containing document files
        timeout: Request timeout in seconds
    """
    # 获取所有 markdown 文件（排除 README）
    md_files = sorted(docs_dir.glob("*.md"))
    md_files = [f for f in md_files if f.name.lower() != "readme.md"]
    
    if not md_files:
        print(f"❌ 在 {docs_dir} 中没有找到 markdown 文档")
        return False
    
    print("=" * 70)
    print(f"📂 找到 {len(md_files)} 个文档待插入")
    print(f"🔗 API 地址: {api_url}")
    print(f"📁 文档目录: {docs_dir}")
    print("=" * 70)
    
    success_count = 0
    failed_count = 0
    
    # 使用 tqdm 显示进度条
    for md_file in tqdm(md_files, desc="插入文档", unit="doc"):
        try:
            # 读取文档内容
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 调用插入 API（正确的端点是 /documents/text）
            response = httpx.post(
                f"{api_url}/documents/text",
                json={"text": content, "file_source": md_file.name},
                timeout=timeout
            )
            
            if response.status_code == 200:
                success_count += 1
                tqdm.write(f"   ✅ {md_file.name} - 插入成功")
            else:
                failed_count += 1
                tqdm.write(f"   ❌ {md_file.name} - 失败 (HTTP {response.status_code})")
                if response.text:
                    tqdm.write(f"      错误信息: {response.text[:200]}")
        
        except httpx.TimeoutException:
            failed_count += 1
            tqdm.write(f"   ❌ {md_file.name} - 超时（{timeout}秒）")
        
        except httpx.ConnectError:
            failed_count += 1
            tqdm.write(f"   ❌ {md_file.name} - 无法连接到服务器")
            tqdm.write(f"      请确认 LightRAG 服务器正在运行: {api_url}")
            break
        
        except Exception as e:
            failed_count += 1
            tqdm.write(f"   ❌ {md_file.name} - 错误: {e}")
    
    # 打印总结
    print("\n" + "=" * 70)
    print("📊 插入完成统计")
    print("=" * 70)
    print(f"✅ 成功: {success_count} 个文档")
    print(f"❌ 失败: {failed_count} 个文档")
    print(f"📝 总计: {len(md_files)} 个文档")
    print("=" * 70)
    
    if success_count > 0:
        print("\n💡 提示：现在可以运行评估脚本了")
        print("   python eval_rag_quality.py")
    
    return failed_count == 0


def main():
    parser = argparse.ArgumentParser(
        description="Insert sample documents into LightRAG for evaluation"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:9621",
        help="LightRAG API base URL (default: http://localhost:9621)"
    )
    parser.add_argument(
        "--docs-dir",
        default="./sample_documents",
        help="Directory containing document files (default: ./sample_documents)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Request timeout in seconds (default: 300)"
    )
    
    args = parser.parse_args()
    
    # 转换为 Path 对象
    docs_dir = Path(args.docs_dir)
    
    # 检查目录是否存在
    if not docs_dir.exists():
        print(f"❌ 错误：文档目录不存在: {docs_dir}")
        return 1
    
    if not docs_dir.is_dir():
        print(f"❌ 错误：{docs_dir} 不是一个目录")
        return 1
    
    # 插入文档
    try:
        success = insert_documents(args.api_url, docs_dir, args.timeout)
        return 0 if success else 1
    
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断操作")
        return 130
    
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
