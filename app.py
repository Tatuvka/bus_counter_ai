import os
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file
import cv2
import numpy as np
from ultralytics import YOLO
import openpyxl

app = Flask(__name__)

# Загрузка модели YOLOv8
print("Загрузка модели YOLOv8...")
model = YOLO('yolov8n.pt')
print("✅ Модель загружена!")

# Инициализация БД
def init_db():
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS requests 
                      (id INTEGER PRIMARY KEY, timestamp TEXT, bus_count INTEGER, filename TEXT)''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "Файл не загружен"}), 400
    
    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({"error": "Не удалось прочитать изображение"}), 400
    
    # Инференс модели
    results = model(img)
    
    # Подсчет именно автобусов
    bus_count = 0
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        class_name = results[0].names[cls_id]
        if class_name == 'bus':
            bus_count += 1
    
    # Визуализация и сохранение
    output_img = results[0].plot()
    output_path = os.path.join('static', 'result.jpg')
    cv2.imwrite(output_path, output_img)
    
    # Сохранение в БД
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('INSERT INTO requests (timestamp, bus_count, filename) VALUES (?, ?, ?)',
                   (timestamp, bus_count, file.filename))
    conn.commit()
    conn.close()
    
    return jsonify(count=bus_count, image_url='/static/result.jpg')

@app.route('/export_excel')
def export_excel():
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('SELECT timestamp, bus_count, filename FROM requests')
    rows = cursor.fetchall()
    conn.close()
    
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "История подсчетов"
    ws.append(["Дата и время", "Количество автобусов", "Имя файла"])
    for row in rows:
        ws.append(row)
    
    excel_path = "static/bus_count_report.xlsx"
    wb.save(excel_path)
    return send_file(excel_path, as_attachment=True)

@app.route('/get_history')
def get_history():
    conn = sqlite3.connect('history.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT timestamp, bus_count, filename FROM requests ORDER BY timestamp DESC LIMIT 10')
    rows = cursor.fetchall()
    conn.close()
    
    history = [dict(row) for row in rows]
    return jsonify(history=history)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)