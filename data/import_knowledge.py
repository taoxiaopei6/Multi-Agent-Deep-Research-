"""导入通用技术文档到 deep_research 的 Milvus 知识库。

使用方法：
    cd F:\agent\deep_research\deep_research\data
    F:\deep_research_env\python.exe import_knowledge.py

导入前确保 Docker 中 Milvus 正在运行。
导入的文档放在 knowledge/ 目录下，可自行添加更多 .md 文件。
"""
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "app"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("import_knowledge")

from mult_agents.rag.core import RAGSystem, RAGConfig


def main():
    rag_cfg = RAGConfig(
        milvus_host="localhost",
        milvus_port=19530,
        collection_name="mult_agent_knowledge",  # 跟 config.json 和 RAGConfig 默认值一致
        embedding_model_path="models/bge-m3",
        chunk_size=500,
        chunk_overlap=50,
    )
    rag = RAGSystem(api_key="", config=rag_cfg)

    knowledge_dir = Path(__file__).resolve().parent / "knowledge"
    md_files = sorted(knowledge_dir.glob("*.md"))

    if not md_files:
        print(f"❌ 在 {knowledge_dir} 没有找到 .md 文件")
        return

    print(f"📄 找到 {len(md_files)} 个文档，开始导入...")

    total_chunks = 0
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        chunks = rag.ingest_text(text, source=str(md_file.name))
        total_chunks += chunks
        print(f"  ✅ {md_file.name} → {chunks} 个片段")

    print(f"\n🎉 导入完成！共 {len(md_files)} 个文档，{total_chunks} 个向量片段")
    print(f"📌 集合: mult_agent_memory")


if __name__ == "__main__":
    main()
