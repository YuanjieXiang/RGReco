from flask import Flask, request, jsonify
import json
import os
from datetime import datetime

app = Flask(__name__)

# 简单的内存存储（生产环境建议使用文件或数据库）
data_store = {}


# 基础路由
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


# API 1: 用户相关
@app.route('/api/users', methods=['GET'])
def get_users():
    return jsonify({'users': data_store.get('users', [])})


@app.route('/api/users', methods=['POST'])
def create_user():
    user_data = request.get_json()
    if not user_data:
        return jsonify({'error': 'No data provided'}), 400

    users = data_store.get('users', [])
    user_data['id'] = len(users) + 1
    user_data['created_at'] = datetime.now().isoformat()
    users.append(user_data)
    data_store['users'] = users

    return jsonify(user_data), 201


@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    users = data_store.get('users', [])
    user = next((u for u in users if u['id'] == user_id), None)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    return jsonify(user)


# API 2: 文件上传处理
@app.route('/api/upload', methods=['POST'])
def upload_data():
    if 'file' in request.files:
        file = request.files['file']
        # 检查文件大小（1MB限制）
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)

        if size > 1024 * 1024:  # 1MB
            return jsonify({'error': 'File too large'}), 413

        content = file.read().decode('utf-8')
        return jsonify({
            'filename': file.filename,
            'size': size,
            'content_preview': content[:100] + '...' if len(content) > 100 else content
        })

    elif request.get_json():
        data = request.get_json()
        # 检查JSON大小
        json_str = json.dumps(data)
        if len(json_str.encode('utf-8')) > 1024 * 1024:
            return jsonify({'error': 'Data too large'}), 413

        return jsonify({
            'received': True,
            'data_size': len(json_str),
            'timestamp': datetime.now().isoformat()
        })

    return jsonify({'error': 'No file or data provided'}), 400


# API 3: 配置管理
@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify(data_store.get('config', {}))


@app.route('/api/config', methods=['POST'])
def update_config():
    config_data = request.get_json()
    data_store['config'] = {**data_store.get('config', {}), **config_data}
    return jsonify(data_store['config'])


# API 4: 搜索功能
@app.route('/api/search', methods=['GET'])
def search():
    query = request.args.get('q', '')
    category = request.args.get('category', 'all')
    limit = int(request.args.get('limit', 10))

    # 模拟搜索逻辑
    results = []
    if query:
        for i in range(min(limit, 5)):
            results.append({
                'id': i + 1,
                'title': f'Result {i + 1} for "{query}"',
                'category': category,
                'score': 0.9 - (i * 0.1)
            })

    return jsonify({
        'query': query,
        'category': category,
        'total': len(results),
        'results': results
    })


# API 5: 统计信息
@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify({
        'users_count': len(data_store.get('users', [])),
        'config_items': len(data_store.get('config', {})),
        'uptime': 'Unknown',  # 可以添加启动时间跟踪
        'memory_usage': len(str(data_store))  # 简单的内存使用指标
    })


# 错误处理
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request'}), 400


@app.errorhandler(413)
def payload_too_large(error):
    return jsonify({'error': 'Payload too large'}), 413


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# 请求日志中间件（可选）
@app.before_request
def log_request():
    print(f"[{datetime.now()}] {request.method} {request.path}")


if __name__ == '__main__':
    print("Starting Flask server...")
    print("Available endpoints:")
    print("  GET  /")
    print("  GET  /api/users")
    print("  POST /api/users")
    print("  GET  /api/users/<id>")
    print("  POST /api/upload")
    print("  GET  /api/config")
    print("  POST /api/config")
    print("  GET  /api/search")
    print("  GET  /api/stats")

    # 开发环境运行
    app.run(debug=True, host='0.0.0.0', port=5000)