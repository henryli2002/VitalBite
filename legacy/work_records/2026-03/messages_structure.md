<!-- SUMMARY: 消息结构说明 | DATE: 2026-03-12 -->
# 消息 (Messages) 结构说明

为了有效地管理对话历史并支持多模态交互（文本和图片），我们设计了以下的消息结构。这种结构清晰、可扩展，并且非常适合长期上下文管理。

## 核心理念

我们将每一次对话都看作是一个由多个消息组成的列表（`messages`）。每个消息都来自于一个明确的角色（`role`），可以是用户（`human`）或 AI 助手（`ai`）。每个消息的内容（`content`）可以包含多个部分，例如一段文字和一张图片，这使得单次交互可以非常丰富。

## 数据结构

### 顶层结构：`messages` 列表

`messages` 是一个列表，其中包含了多个消息对象。这个列表按照时间顺序记录了整个对话的流程。

```json
[
    {...message_1...},
    {...message_2...},
    ...
]
```

### 单个消息对象

每个消息对象包含两个关键字段：

-   `role` (字符串): 消息的发送者。
    -   `"human"`: 代表最终用户。
    -   `"ai"`: 代表 AI 助手。
-   `content` (列表): 一个包含了该消息所有内容的列表。

```json
{
  "role": "human",
  "content": [
    {...content_part_1...},
    {...content_part_2...}
  ]
}
```

### 内容部分 (Content Parts)

`content` 列表中的每个元素都是一个对象，代表消息的一部分。我们通过 `type` 字段来区分不同类型的内容。

#### 文本内容 (`text`)

-   `type`: `"text"`
-   `text` (字符串): 消息的文本内容。

```json
{
  "type": "text",
  "text": "你好，这张图里是什么？"
}
```

#### 图片内容 (`image`)

-   `type`: `"image"`
-   `source` (对象): 图片的来源。
    -   `type`: 图片数据的格式，例如 `"base64"` 或 `"url"`。
    -   `media_type`: 图片的 MIME 类型，例如 `"image/jpeg"` 或 `"image/png"`。
    -   `data`: 图片的具体数据（例如 Base64 编码的字符串）。

```json
{
  "type": "image",
  "source": {
    "type": "base64",
    "media_type": "image/jpeg",
    "data": "<base64_encoded_image_string>"
  }
}
```

## 完整示例

下面是一个包含用户提问（文本+图片）和 AI 回答的 `messages` 列表的完整示例。

```json
[
  {
    "role": "human",
    "content": [
      {
        "type": "text",
        "text": "help me with this"
      },
      {
        "type": "image",
        "source": {
          "type": "base64",
          "media_type": "image/jpeg",
          "data": "iVBORw0KGgoAAAANSUhEUg..."
        }
      }
    ]
  },
  {
    "role": "ai",
    "content": [
      {
        "type": "text",
        "text": "..."
      }
    ]
  }
]
```

## Python Pydantic 模型

为了在代码中更方便、更安全地处理这些数据，我们可以使用 Pydantic 来定义相应的模型。

```python
from pydantic import BaseModel, Field
from typing import List, Literal, Union

class ImageSource(BaseModel):
    type: Literal["base64", "url"] = "base64"
    media_type: str = Field(..., description="The MIME type of the image, e.g., 'image/jpeg'.")
    data: str

class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    source: ImageSource

ContentPart = Union[TextContent, ImageContent]

class Message(BaseModel):
    role: Literal["human", "ai"]
    content: List[ContentPart]

class Messages(BaseModel):
    messages: List[Message]

# 示例用法
# messages_data = [...] # 从 API 请求或数据库中获取的 JSON 数据
# messages_obj = Messages(messages=messages_data)
# print(messages_obj.messages[0].content[0].text)

```

## 对长期上下文管理的好处

这种结构化的消息列表为长期上下文管理提供了坚实的基础：

1.  **历史保真性**: 列表完整、有序地记录了对话的每一个环节，不会丢失任何信息。
2.  **多模态支持**: 通过 `content` 列表和 `type` 字段，可以轻松地在同一个消息中支持文本、图片以及未来可能扩展的其他媒体类型（如音频、视频）。
3.  **易于处理**: 结构化的数据（而不是非结构化的字符串）使得程序可以轻松地解析、遍历和操作对话历史。例如，可以很容易地统计图片数量、提取所有文本，或者为不同模态的内容采用不同的处理策略。
4.  **可扩展性强**: 当需要支持新的内容类型（例如 "video"）或者新的角色（例如 "system"）时，只需扩展 `ContentPart` 的 `Union` 类型或 `role` 的 `Literal` 类型即可，对现有结构影响很小。
5.  **LLM 兼容性**: 这种格式与当前主流的大语言模型（LLM）的多模态输入格式（如 OpenAI的 GPT-4o, Google的 Gemini）高度兼容，可以直接用于 API 调用。
