#!/usr/bin/env python3
"""
Flask web application for Professor Search Engine
"""

from flask import Flask, render_template, request, jsonify
from scripts.search_engine import ProfessorSearchEngine
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# Initialize search engine (will take 30-60 seconds)
logger.info("Initializing search engine...")
search_engine = ProfessorSearchEngine(enable_ai_summary=True)
logger.info("Search engine ready!")


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/filter-options')
def get_filter_options():
    """Get available filter options"""
    try:
        options = search_engine.get_filter_options()
        return jsonify({
            'success': True,
            'data': options
        })
    except Exception as e:
        logger.error(f"Error getting filter options: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/search', methods=['POST'])
def search():
    """Search professors"""
    try:
        data = request.get_json()
        
        query = data.get('query', '').strip()
        regions = data.get('regions', [])
        sub_regions = data.get('sub_regions', [])
        countries = data.get('countries', [])
        top_k = data.get('top_k', 50)
        
        if not query:
            return jsonify({
                'success': False,
                'error': '搜索关键词不能为空'
            }), 400
        
        logger.info(f"Search query: {query}, top_k: {top_k}")
        
        # Perform search
        result = search_engine.search(
            query=query,
            top_k=top_k,
            regions=regions,
            sub_regions=sub_regions,
            countries=countries,
            generate_summary=True
        )
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 Professor Search Engine - Web Interface")
    print("="*80)
    print("\n访问地址: http://localhost:5001")
    print("\n按 Ctrl+C 停止服务器\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)

