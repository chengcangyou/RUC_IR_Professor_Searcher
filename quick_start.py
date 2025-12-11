#!/usr/bin/env python3
"""
Quick start Flask app - loads search engine in background
"""

from flask import Flask, render_template, request, jsonify
import logging
import threading

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

# Global search engine variable
search_engine = None
engine_ready = False


def init_search_engine():
    """Initialize search engine in background"""
    global search_engine, engine_ready
    try:
        logger.info("🚀 Initializing search engine in background...")
        from scripts.search_engine import ProfessorSearchEngine

        search_engine = ProfessorSearchEngine(enable_ai_summary=True)
        engine_ready = True
        logger.info("✅ Search engine ready!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize search engine: {e}")
        import traceback

        traceback.print_exc()


@app.route("/")
def index():
    """Main page"""
    return render_template("index.html")


@app.route("/api/status")
def get_status():
    """Get engine status"""
    return jsonify(
        {
            "ready": engine_ready,
            "message": (
                "Search engine is ready"
                if engine_ready
                else "Search engine is initializing..."
            ),
        }
    )


@app.route("/api/filter-options")
def get_filter_options():
    """Get available filter options"""
    if not engine_ready:
        return (
            jsonify({"success": False, "error": "搜索引擎正在初始化，请稍候..."}),
            503,
        )

    try:
        options = search_engine.get_filter_options()
        return jsonify({"success": True, "data": options})
    except Exception as e:
        logger.error(f"Error getting filter options: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/search", methods=["POST"])
def search():
    """Search professors"""
    if not engine_ready:
        return (
            jsonify({"success": False, "error": "搜索引擎正在初始化，请稍候..."}),
            503,
        )

    try:
        data = request.get_json()

        query = data.get("query", "").strip()
        regions = data.get("regions", [])
        sub_regions = data.get("sub_regions", [])
        countries = data.get("countries", [])
        top_k = data.get("top_k", 50)

        if not query:
            return jsonify({"success": False, "error": "搜索关键词不能为空"}), 400

        logger.info(f"Search query: {query}, top_k: {top_k}")

        # Perform search
        result = search_engine.search(
            query=query,
            top_k=top_k,
            regions=regions,
            sub_regions=sub_regions,
            countries=countries,
            generate_summary=True,
        )

        return jsonify({"success": True, "data": result})

    except Exception as e:
        logger.error(f"Search error: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 Professor Search Engine - Web Interface")
    print("=" * 80)
    print("\n✅ Flask server starting...")
    print("📍 访问地址: http://localhost:5001")
    print("⏳ 搜索引擎正在后台初始化（需要 30-60 秒）...")
    print("\n💡 提示: 页面会立即打开，但需要等待搜索引擎初始化完成才能搜索")
    print("\n按 Ctrl+C 停止服务器\n")
    print("=" * 80 + "\n")

    # Start search engine initialization in background thread
    init_thread = threading.Thread(target=init_search_engine, daemon=True)
    init_thread.start()

    # Start Flask app
    app.run(debug=True, host="0.0.0.0", port=5001, use_reloader=False)
