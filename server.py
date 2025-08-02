from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import os

# ===============================
# Config
# ===============================
DOCS_DIR = "./docs"
LLM_MODEL = "qwen2:7b"           # Model LLM chính
EMBED_MODEL = "nomic-embed-text" # Model embedding
REQUEST_TIMEOUT = 120.0

# ===============================
# Flask App
# ===============================
app = Flask(__name__)
CORS(app)

# ===============================
# Init Ollama models
# ===============================
llm = Ollama(model=LLM_MODEL, request_timeout=REQUEST_TIMEOUT)
embed_model = OllamaEmbedding(model_name=EMBED_MODEL)

# ===============================
# Load or create vector index
# ===============================
def load_or_create_index():
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    return VectorStoreIndex.from_documents(documents, embed_model=embed_model)

index = load_or_create_index()

# ===============================
# OpenAPI JSON
# ===============================
@app.route("/openapi.json", methods=["GET"])
def openapi_json():
    """Trả về cấu hình OpenAPI để Open WebUI nhận diện"""
    return jsonify({
        "openapi": "3.0.0",
        "info": {
            "title": "BroOffline Backend API",
            "version": "1.0.0",
            "description": "API đa chế độ cho BroOffline (Chat LLM & Tài liệu Offline)"
        },
        "paths": {
            "/chat": {
                "post": {
                    "summary": "Gửi tin nhắn để chat",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"},
                                        "mode": {"type": "string", "enum": ["auto", "llm", "docs"]}
                                    },
                                    "required": ["message"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Phản hồi từ backend",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "mode": {"type": "string"},
                                            "response": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/reload-docs": {
                "post": {
                    "summary": "Tải lại tài liệu offline",
                    "responses": {
                        "200": {
                            "description": "Kết quả reload",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string"},
                                            "message": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    })

# ===============================
# Chat API
# ===============================
@app.route("/chat", methods=["POST"])
def chat():
    """
    API Chat đa chế độ
    mode:
        - auto   : tự phát hiện
        - llm    : chỉ chat tự do
        - docs   : hỏi đáp tài liệu offline
    """
    data = request.json
    if not data or "message" not in data:
        return jsonify({"error": "Thiếu 'message'"}), 400

    user_message = data["message"].strip()
    mode = data.get("mode", "auto")

    # Tự phát hiện chế độ nếu mode = auto
    if mode == "auto":
        if any(k in user_message.lower() for k in ["tài liệu", "document", "file", "docs"]):
            mode = "docs"
        else:
            mode = "llm"

    if mode == "llm":
        response = llm.complete(user_message)
        return jsonify({"mode": "llm", "response": response.text})

    elif mode == "docs":
        query_engine = index.as_query_engine(llm=llm)
        response = query_engine.query(user_message)
        return jsonify({"mode": "docs", "response": str(response)})

    else:
        return jsonify({"error": f"Mode '{mode}' không hợp lệ"}), 400

# ===============================
# Reload docs
# ===============================
@app.route("/reload-docs", methods=["POST"])
def reload_docs():
    """Reload lại tài liệu khi có file mới"""
    global index
    index = load_or_create_index()
    return jsonify({"status": "ok", "message": "Tài liệu đã được reload"})

# ===============================
# Main
# ===============================
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
