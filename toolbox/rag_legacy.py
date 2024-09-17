import os
from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader
import dotenv
import openai


def index_algorithm_files(base_path):
    index_map = {}

    # 遍历 base_path 目录下的每个算法文件夹
    for algorithm_name in os.listdir(base_path):
        algorithm_path = os.path.join(base_path, algorithm_name)

        # 使用 SimpleDirectoryReader 加载文档数据
        document = SimpleDirectoryReader(algorithm_path).load_data()
        print(algorithm_path, len(document))
        index_map[algorithm_name] = document

    return index_map


def query_algorithms(query):
    results = []

    for algorithm_name, index in indices.items():
        print(f"Query {algorithm_name}")

        # 获取检索器，提取每个响应的文本内容
        retriever = index.as_retriever(verbose=True)
        responses = retriever.retrieve(query)
        response_texts = [response.text for response in responses]
        results.append((algorithm_name, response_texts))

    return results


if __name__ == "__main__":
    dotenv.load_dotenv(dotenv_path="../.env")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.base_url = "https://api.gptsapi.net/v1"

    base_pth = './code_base'
    idx_map = index_algorithm_files(base_pth)

    # 使用文档创建 VectorStoreIndex
    indices = {name: VectorStoreIndex.from_documents(doc) for name, doc in idx_map.items()}

    q = "旋转"
    print("Start querying")
    documents = query_algorithms(q)

    for alg_name, resp in documents:
        print(f"Results from {alg_name}:")
        print(resp)
