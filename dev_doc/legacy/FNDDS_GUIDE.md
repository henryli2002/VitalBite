好的，准备 FNDDS 数据库并将其集成为一个 RAG (Retrieval-Augmented Generation) 系统是一个相对复杂但非常有价值的过程。这将极大地提升食物识别的准确性和营养分析的深度。

我为您准备了一个详细的步骤指南。

### FNDDS 数据库准备与 RAG 集成指南

#### 目标
将 FNDDS 数据转换成一个向量数据库，并创建一个检索工具，该工具可以根据 LLM 生成的食物描述，精确地从中查询相关的营养信息。

#### 步骤 1: 下载 FNDDS 数据

1.  **访问 USDA 官网**: FNDDS 数据由美国农业部 (USDA) 提供。您需要访问 [USDA FoodData Central](https://www.ars.usda.gov/northeast-area/beltsville-md-20705/beltsville-human-nutrition-research-center/food-surveys-research-group/docs/fndds-download-databases/) 页面。
2.  **下载数据文件**: 寻找最新版本的 FNDDS 数据。通常，这些数据会以 SAS (`.XPT`) 或 Access (`.mdb`) 格式提供。您需要将它们下载下来，并可能需要使用工具（如 Python 的 `pandas` 库或 `pyreadstat` 库）将 SAS 文件转换为更通用的 CSV 格式。

#### 步骤 2: 数据预处理和格式化

下载后的原始数据是规范化的关系型数据，包含多个文件（例如，食物描述、营养素含量、分量单位等）。为了用于 RAG，我们需要将其处理成一种易于检索的格式。

1.  **合并数据**: 将多个相关的表单合并。我们的目标是为每种食物创建一个单一的“文档”，这个文档包含：
    *   **食物描述**: 详细的食物名称和描述。
    *   **主要营养素**: 卡路里、蛋白质、碳水、脂肪等。
    *   **全营养素信息**: 所有 65 种营养素的列表及其含量和单位。
    *   **常见分量**: 如“1 杯”、“100 克”等对应的克数。

2.  **格式化为 JSON 或文本**: 将每种食物的合并信息格式化为一个 JSON 对象或一段结构清晰的文本。例如：

    ```json
    {
      "food_code": "27117200",
      "description": "Milk, whole, 3.25% milkfat",
      "serving_options": [
        {"unit": "cup", "grams": 244},
        {"unit": "gram", "grams": 1}
      ],
      "nutrients": [
        {"name": "Protein", "amount": 8.05, "unit": "g", "per_100g": 3.3},
        {"name": "Total lipid (fat)", "amount": 7.93, "unit": "g", "per_100g": 3.25},
        ... 63 more nutrients
      ]
    }
    ```

#### 步骤 3: 创建 RAG 检索器

这是最核心的部分。我们将使用 `LangChain` 或类似的框架来简化这个过程。

1.  **选择 Embedding 模型**:
    您需要一个模型将您的食物描述文本转换成向量（一串数字）。这些向量能捕捉文本的语义含义。推荐使用高质量的开源模型，如 `sentence-transformers`，或者使用来自 OpenAI、Google 的 API。

2.  **构建向量数据库 (Vector Store)**:
    a.  **加载数据**: 用 LangChain 的 Document Loaders (如 `JSONLoader` 或 `TextLoader`) 加载您在步骤 2 中格式化好的数据。
    b.  **切分文档**: 如果您的文档过长，需要将其切分为更小的块 (`RecursiveCharacterTextSplitter` 是一个好选择)。
    c.  **嵌入并存储**:
    *   遍历所有文档块，使用您选择的 Embedding 模型生成向量。
    *   将这些向量和原始文本块一起存入一个向量数据库中。对于本地开发，**FAISS** 或 **ChromaDB** 是非常好的选择，它们快速且易于设置。

3.  **创建 Retriever**:
    向量数据库本身可以被配置为一个 Retriever。这个 Retriever 对象将有一个方法，如 `get_relevant_documents(query)`。当您调用它时，它会：
    *   将查询字符串（例如 "a glass of whole milk"）转换为向量。
    *   在数据库中执行相似性搜索（通常是余弦相似度）。
    *   返回最相似的 Top-K 个文档（即最相关的食物营养信息）。

#### 步骤 4: 集成到您的代码中

现在，您需要用刚刚创建的 RAG Retriever 替换掉我之前写的占位符工具。

1.  **修改 `langgraph_app/tools/nutrition/fndds.py`**:
    *   在文件顶部，初始化您的向量数据库和 Retriever。这可能需要加载已经构建好的 FAISS 索引文件。
    *   修改 `fndds_nutrition_search_tool` 函数的内部逻辑。
    *   让它调用您的 `retriever.get_relevant_documents(food_description)` 方法。
    *   将返回的文档（可能是多个）格式化为一个 JSON 字符串，然后返回。

    **示例代码**:
    ```python
    # langgraph_app/tools/nutrition/fndds.py (修改后)
    from langchain_core.tools import tool
    from pydantic import BaseModel, Field
    import json

    # --- 在这里初始化您的 RAG Retriever (这部分代码依赖于您的具体实现) ---
    # 伪代码:
    # my_retriever = initialize_fndds_retriever("path/to/your/vector_store")
    # ---

    class FnddsSearchInput(BaseModel):
        food_description: str = Field(description="A description of the food to search for.")

    @tool("fndds_nutrition_search", args_schema=FnddsSearchInput)
    def fndds_nutrition_search_tool(food_description: str) -> str:
        """
        Searches the FNDDS RAG database for nutritional information about a food.
        """
        print(f"Searching FNDDS RAG for: {food_description}")
        
        # 使用真实的 Retriever 进行搜索
        # relevant_docs = my_retriever.get_relevant_documents(food_description)
        
        # # 将文档内容格式化为 JSON
        # results = [doc.page_content for doc in relevant_docs]
        # return json.dumps(results, ensure_ascii=False)

        # 在您完成上述操作前的临时占位符
        mock_data = { "food_name": food_description, "calories": 250, "status": "real implementation pending" }
        return json.dumps(mock_data, ensure_ascii=False)

    ```

这个过程需要一些数据工程和机器学习的知识，但完成后，您的应用将拥有一个功能强大且高度准确的营养分析核心。