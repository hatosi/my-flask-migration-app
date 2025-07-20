from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import joblib
import os
import json
import numpy as np
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
import io

app = Flask(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

model_bundle = joblib.load("migration_model_growth_stack.pkl")

xgb_model = model_bundle["gb_model"]
stack_model = model_bundle["stack_model"]
region_encoder = model_bundle["region_encoder"]
province_encoder = model_bundle["province_encoder"]
feature_columns = model_bundle["feature_columns"]

train_df = pd.read_csv("Train.csv")
detailed_df = pd.read_csv("migration_forecast_growth_model.csv")

province_names = list(province_encoder.classes_)

def get_province_name_mapping():
    """Create mapping between different province name formats"""
    return {
        # GeoJSON name -> Model/CSV data name
        "TP. Hồ Chí Minh": "TP Hồ Chí Minh",
        "Thừa Thiên Huế": "Huế",
        "Thừa Thiên - Huế": "Huế",  # GeoJSON uses dash
        "Hoà Bình": "Hòa Bình",     # Different accent marks
        "Khánh Hoà": "Khánh Hòa",   # Different accent marks  
        "Thanh Hoá": "Thanh Hóa"    # Different accent marks
    }

def normalize_province_name(province_name, reverse=False):
    """Normalize province names for consistent matching"""
    mapping = get_province_name_mapping()
    
    if reverse:
        # CSV -> GeoJSON mapping
        reverse_mapping = {v: k for k, v in mapping.items()}
        return reverse_mapping.get(province_name, province_name)
    else:
        # GeoJSON -> CSV mapping  
        return mapping.get(province_name, province_name)

def get_province_data(province, year):
    """Get province data from appropriate dataset"""
    if 2004 <= year <= 2024:
        csv_province_name = normalize_province_name(province)
        historical_row = train_df[
            (train_df["province"] == csv_province_name) & 
            (train_df["year"] == year)
        ]   
        
        if historical_row.empty:
            return None
        
        return historical_row.iloc[0]
    
    elif 2025 <= year <= 2030:
        csv_province_name = normalize_province_name(province)
        detailed_row = detailed_df[
            (detailed_df["province_name"] == csv_province_name) & 
            (detailed_df["year"] == year)
        ]   
        
        if detailed_row.empty:
            return None
        
        return detailed_row.iloc[0]
    
    return None

def get_historical_migration_data(province, year):
    """Get historical migration data from Train.csv for years 2004-2024"""
    # Normalize province name for CSV lookup
    csv_province_name = normalize_province_name(province)
    
    historical_row = train_df[
        (train_df["province"] == csv_province_name) & 
        (train_df["year"] == year)
    ]   
    
    if historical_row.empty:
        return None, None
    
    row_data = historical_row.iloc[0]

    # Extract migration rate directly from the historical data
    migration_rate = float(row_data["migration_rate(‰)"])
    
    base = {
        "year": convert_numpy_types(row_data["year"]),
        "province": csv_province_name,
        "region": str(row_data["region"]),
        "area": convert_numpy_types(row_data["area"]),
        "population": convert_numpy_types(row_data["population"]),
        "monthly_income_per_capita": convert_numpy_types(row_data["monthly_income_per_capita"]),
        "grdp_per_capita": convert_numpy_types(row_data["grdp_per_capita"]),
        "temp_mean": convert_numpy_types(row_data["temp_mean"]),
        "total_precip": convert_numpy_types(row_data["total_precip"]),
        "precip_hours": convert_numpy_types(row_data["precip_hours"]),
        "sunshine_hours": convert_numpy_types(row_data["sunshine_hours"]),
        "snowfall": convert_numpy_types(row_data["snowfall"]),
        "central_administrated": convert_numpy_types(row_data["central_administrated"]),
        "airport": convert_numpy_types(row_data["airport"]),
        "maritime_port": str(row_data["maritime_port"]),
        "population_density": convert_numpy_types(row_data["population_density"])
    }

    return migration_rate, base

def generate_prediction_with_data(province, year):
    # Check if this is historical data (2004-2024) or prediction data (2025-2030)
    if 2004 <= year <= 2024:
        return get_historical_migration_data(province, year)
    elif 2025 <= year <= 2030:
        # Use existing prediction logic for future years
        # Normalize province name for CSV lookup
        csv_province_name = normalize_province_name(province)
        
        row_data = get_province_data(csv_province_name, year)
        
        if row_data is None:
            return None, None
        
        base = {
            "year": year,
            "province": province_encoder.transform([province])[0],
            "region": region_encoder.transform([row_data["region_name"]])[0],
            "area": row_data["area"],
            "population": row_data["population"],
            "monthly_income_per_capita": row_data["monthly_income_per_capita"],
            "grdp_per_capita": row_data["grdp_per_capita"],
            "temp_mean": row_data["temp_mean"],
            "total_precip": row_data["total_precip"],
            "precip_hours": row_data["precip_hours"],
            "sunshine_hours": row_data["sunshine_hours"],
            "snowfall": row_data["snowfall"],
            "central_administrated": row_data["central_administrated"],
            "airport": row_data["airport"],
            "maritime_port": row_data["maritime_port"],
            "population_density": row_data["population_density"]
        }

        df_input = pd.DataFrame([base])
        df_encoded = pd.get_dummies(df_input, columns=["region", "province"], drop_first=False)

        for col in feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[feature_columns]

        pred_xgb = xgb_model.predict(df_encoded).reshape(-1, 1)
        pred = stack_model.predict(pred_xgb)[0]

        prediction = float(round(pred, 2))
        return prediction, base
    else:
        return None, None

@app.route('/')
def index():
    return render_template("index.html", provinces=province_names)

@app.route('/info')
def info():
    """Serve the project information page"""
    try:
        # Read content from infopage.txt
        with open('infopage.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the content into sections
        sections = {}
        current_section = None
        current_content = []
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.')):
                # Save previous section
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = line
                current_content = []
            elif line:
                current_content.append(line)
        
        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return render_template("info.html", sections=sections)
        
    except FileNotFoundError:
        return render_template("info.html", sections={"Error": "Không tìm thấy file thông tin dự án"})
    except Exception as e:
        return render_template("info.html", sections={"Error": f"Lỗi đọc file: {str(e)}"})

@app.route('/api/vietnam-geojson')
def vietnam_geojson():
    """Serve Vietnam GeoJSON data for Plotly choropleth"""
    try:
        # Try to load more detailed GeoJSON if available
        try:
            with open('static/vietnam-provinces-detailed.geojson', 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)
        except FileNotFoundError:
            # Fallback to basic GeoJSON
            with open('static/vietnam-provinces.geojson', 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)
        
        return jsonify(geojson_data)
    except Exception as e:
        return jsonify({"error": f"Failed to load GeoJSON data: {str(e)}"}), 500

@app.route('/api/svg-map/<int:year>')
def generate_svg_map(year):
    """Generate SVG choropleth map for a specific year"""
    try:
        # Get predictions for the specified year
        predictions = {}
        for province in province_names:
            prediction, _ = generate_prediction_with_data(province, year)
            if prediction is not None:
                # Use the original province name (GeoJSON format) as key
                geojson_province_name = normalize_province_name(province, reverse=True)
                predictions[geojson_province_name] = prediction
        
        # Load GeoJSON data
        geojson_paths = [
            "Vietnam-Choropleth-Map/Vietnam_Population_Density/cleaned_geo.geojson",
            "Vietnam-Choropleth-Map/Vietnam_Forest_Fire/cleaned_geo.geojson",
            "static/vietnam-provinces.geojson"
        ]
        
        geojson_data = None
        for path in geojson_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    geojson_data = json.load(f)
                break
            except FileNotFoundError:
                continue
        
        if geojson_data is None:
            return jsonify({"error": "No GeoJSON file found"}), 404
        
        # Create SVG
        svg_content = create_svg_map_content(geojson_data, predictions, year)
        
        # Return SVG as response
        return send_file(
            io.BytesIO(svg_content.encode('utf-8')),
            mimetype='image/svg+xml',
            as_attachment=False,
            download_name=f'vietnam_migration_map_{year}.svg'
        )
        
    except Exception as e:
        return jsonify({"error": f"Failed to generate SVG map: {str(e)}"}), 500

def create_svg_map_content(geojson_data, migration_data, year):
    """Create SVG content for the choropleth map"""
    
    # Calculate bounds
    all_coords = []
    for feature in geojson_data['features']:
        if feature['geometry']['type'] == 'Polygon':
            coords = feature['geometry']['coordinates'][0]
        elif feature['geometry']['type'] == 'MultiPolygon':
            coords = []
            for polygon in feature['geometry']['coordinates']:
                coords.extend(polygon[0])
        else:
            continue
        all_coords.extend(coords)
    
    if not all_coords:
        raise ValueError("No valid coordinates found")
    
    lons = [coord[0] for coord in all_coords]
    lats = [coord[1] for coord in all_coords]
    
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    
    # SVG dimensions
    width = 1200  # Increased from 1000 to 1200 for more legend space
    height = 700
    padding = 80
    map_offset_x = 50  # Shift map to the right by 50px
    
    # Calculate scale
    lon_scale = (width - 2 * padding - map_offset_x) / (max_lon - min_lon)  # Account for offset
    lat_scale = (height - 2 * padding) / (max_lat - min_lat)
    scale = min(lon_scale, lat_scale)
    
    def project_point(lon, lat):
        x = padding + map_offset_x + (lon - min_lon) * scale  # Add offset to x
        y = height - padding - (lat - min_lat) * scale
        return x, y
    
    def get_color_for_value(value, min_val, max_val):
        """Get color for migration rate value using red-white-green scheme"""
        # Handle case where all values are the same
        if max_val == min_val:
            return '#FFFFFF'  # White for neutral
        
        # Define balance threshold (around 0)
        balance_threshold = 0.1  # Values between -0.1 and 0.1 are considered balanced
        
        if abs(value) <= balance_threshold:
            # Balanced - white or very light colors
            return '#FFFFFF'  # Pure white for balanced
        elif value < 0:
            # Out-migration - shades of red
            # Normalize the negative value to 0-1 range for color intensity
            abs_value = abs(value)
            if abs_value <= 2:
                return '#FFCCCC'  # Very light red
            elif abs_value <= 5:
                return '#FF9999'  # Light red
            elif abs_value <= 10:
                return '#FF6666'  # Medium red
            elif abs_value <= 15:
                return '#FF3333'  # Red
            elif abs_value <= 20:
                return '#FF0000'  # Bright red
            else:
                return '#CC0000'  # Dark red for very high out-migration
        else:
            # In-migration - shades of green
            if value <= 2:
                return '#CCFFCC'  # Very light green
            elif value <= 5:
                return '#99FF99'  # Light green
            elif value <= 10:
                return '#66FF66'  # Medium green
            elif value <= 15:
                return '#33FF33'  # Green
            elif value <= 20:
                return '#00FF00'  # Bright green
            else:
                return '#00CC00'  # Dark green for very high in-migration
    
    # Get min/max migration values
    if migration_data:
        min_migration = min(migration_data.values())
        max_migration = max(migration_data.values())
    else:
        min_migration = max_migration = 0
    
    # Create SVG root
    svg = Element('svg')
    svg.set('width', str(width))
    svg.set('height', str(height))
    svg.set('viewBox', f'0 0 {width} {height}')
    svg.set('xmlns', 'http://www.w3.org/2000/svg')
    
    # Add styles
    style = SubElement(svg, 'style')
    style.text = """
        .province { 
            stroke: #333; 
            stroke-width: 0.8; 
            cursor: pointer; 
            transition: all 0.3s ease;
        }
        .province:hover { 
            stroke: #007acc; 
            stroke-width: 2; 
            filter: brightness(1.1); 
            opacity: 0.9;
        }
        .title { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            font-size: 24px; 
            font-weight: bold; 
            fill: #007acc; 
            text-anchor: middle; 
        }
        .subtitle { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            font-size: 16px; 
            fill: #666; 
            text-anchor: middle; 
        }
        .legend { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            font-size: 13px; 
            fill: #333; 
        }
        .legend-title { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            font-size: 14px; 
            font-weight: bold; 
            fill: #007acc; 
        }
        /* Tooltip styles */
        .province-tooltip {
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .province-tooltip.visible {
            opacity: 1;
        }
    """
    
    # Add title
    main_title = SubElement(svg, 'text')
    main_title.set('x', str(width // 2))
    main_title.set('y', '20')  # Moved up from 35 to 25
    main_title.set('class', 'title')
    
    # Determine if this is historical or predicted data
    is_historical = 2004 <= year <= 2024
    if is_historical:
        main_title.text = f'Bản đồ Tỷ lệ Di cư Việt Nam {year}'
    else:
        main_title.text = f'Bản đồ Tỷ lệ Di cư Việt Nam {year} (Dự đoán)'
    
    # Add subtitle
    subtitle = SubElement(svg, 'text')
    subtitle.set('x', str(width // 2))
    subtitle.set('y', '40')  # Moved up from 55 to 45
    subtitle.set('class', 'subtitle')
    subtitle_text = 'Tỷ lệ di cư thuần (đơn vị: ‰ - phần nghìn) - Dữ liệu thực tế' if is_historical else 'Tỷ lệ di cư thuần (đơn vị: ‰ - phần nghìn) - Dự đoán từ mô hình AI'
    subtitle.text = subtitle_text
    
    # Draw provinces
    for feature in geojson_data['features']:
        # Try different property names for province names
        province_name = (feature['properties'].get('ten_tinh') or 
                        feature['properties'].get('NAME_1') or 
                        'Unknown')
        migration_rate = migration_data.get(province_name, 0)
        
        # Get color
        color = get_color_for_value(migration_rate, min_migration, max_migration)
        
        # Create path element
        path = SubElement(svg, 'path')
        path.set('class', 'province')
        path.set('fill', color)
        path.set('data-province', province_name)
        path.set('data-rate', f"{migration_rate:.2f}")
        
        # Build path data
        path_data = []
        
        if feature['geometry']['type'] == 'Polygon':
            coordinates = feature['geometry']['coordinates']
            for ring in coordinates:
                path_data.append('M')
                for i, (lon, lat) in enumerate(ring):
                    x, y = project_point(lon, lat)
                    if i == 0:
                        path_data.append(f'{x:.2f},{y:.2f}')
                    else:
                        path_data.append(f'L{x:.2f},{y:.2f}')
                path_data.append('Z')
        
        elif feature['geometry']['type'] == 'MultiPolygon':
            coordinates = feature['geometry']['coordinates']
            for polygon in coordinates:
                for ring in polygon:
                    path_data.append('M')
                    for i, (lon, lat) in enumerate(ring):
                        x, y = project_point(lon, lat)
                        if i == 0:
                            path_data.append(f'{x:.2f},{y:.2f}')
                        else:
                            path_data.append(f'L{x:.2f},{y:.2f}')
                    path_data.append('Z')
        
        path.set('d', ' '.join(path_data))
        
        # Note: Removed SVG title element to avoid duplicate tooltips
        # Custom JavaScript tooltip will handle hover interactions
    
    # Add legend
    legend_x = width - 350  # Increased space from 200 to 350 for longer text
    legend_y = 100
    
    legend_title = SubElement(svg, 'text')
    legend_title.set('x', str(legend_x))
    legend_title.set('y', str(legend_y))
    legend_title.set('class', 'legend-title')
    legend_title.text = 'Tỷ lệ di cư (‰)'
    
    # Legend items with new color scheme
    legend_items = [
        ('#CC0000', 'Xuất cư rất mạnh <tspan font-weight="bold">(<-20‰)</tspan>'),
        ('#FF0000', 'Xuất cư mạnh <tspan font-weight="bold">(-20 đến -15‰)</tspan>'),
        ('#FF3333', 'Xuất cư vừa <tspan font-weight="bold">(-15 đến -10‰)</tspan>'),
        ('#FF6666', 'Xuất cư nhẹ <tspan font-weight="bold">(-10 đến -5‰)</tspan>'),
        ('#FF9999', 'Xuất cư rất nhẹ <tspan font-weight="bold">(-5 đến -0.1‰)</tspan>'),
        ('#FFFFFF', 'Cân bằng <tspan font-weight="bold">(-0.1 đến 0.1‰)</tspan>'),
        ('#CCFFCC', 'Nhập cư rất nhẹ <tspan font-weight="bold">(0.1 đến 5‰)</tspan>'),
        ('#99FF99', 'Nhập cư nhẹ <tspan font-weight="bold">(5 đến 10‰)</tspan>'),
        ('#66FF66', 'Nhập cư vừa <tspan font-weight="bold">(10 đến 15‰)</tspan>'),
        ('#33FF33', 'Nhập cư mạnh <tspan font-weight="bold">(15 đến 20‰)</tspan>'),
        ('#00CC00', 'Nhập cư rất mạnh <tspan font-weight="bold">(>20‰)</tspan>')
    ]
    
    for i, (color, label) in enumerate(legend_items):
        y = legend_y + 25 + i * 20
        
        # Color rectangle
        rect = SubElement(svg, 'rect')
        rect.set('x', str(legend_x))
        rect.set('y', str(y - 12))
        rect.set('width', '15')
        rect.set('height', '15')
        rect.set('fill', color)
        rect.set('stroke', '#333')
        rect.set('stroke-width', '0.5')
        
        # Label
        label_elem = SubElement(svg, 'text')
        label_elem.set('x', str(legend_x + 20))
        label_elem.set('y', str(y))
        label_elem.set('class', 'legend')
        
        # Parse the label to handle tspan formatting
        if '<tspan' in label:
            # Split text and tspan parts
            parts = label.split('<tspan font-weight="bold">')
            if len(parts) == 2:
                # Normal text part
                label_elem.text = parts[0]
                # Bold part
                bold_part = parts[1].replace('</tspan>', '')
                tspan = SubElement(label_elem, 'tspan')
                tspan.set('font-weight', 'bold')
                tspan.text = bold_part
        else:
            label_elem.text = label
    
    # Add statistics
    stats_y = legend_y + 260  # Increased to accommodate longer legend
    stats_title = SubElement(svg, 'text')
    stats_title.set('x', str(legend_x))
    stats_title.set('y', str(stats_y))
    stats_title.set('class', 'legend-title')
    stats_title.text = 'Thống kê:'
    
    avg_migration = sum(migration_data.values()) / len(migration_data) if migration_data else 0
    positive_count = sum(1 for v in migration_data.values() if v > 0)
    negative_count = sum(1 for v in migration_data.values() if v < 0)
    
    stats_items = [
        f'Trung bình: {avg_migration:.2f}‰',
        f'Số tỉnh thành có xu hướng nhập cư: {positive_count}',
        f'Số tỉnh thành có xu hướng xuất cư: {negative_count}',
        f'Tổng số tỉnh thành: {len(migration_data)}'
    ]
    
    for i, stat in enumerate(stats_items):
        y = stats_y + 20 + i * 18
        stat_elem = SubElement(svg, 'text')
        stat_elem.set('x', str(legend_x))
        stat_elem.set('y', str(y))
        stat_elem.set('class', 'legend')
        stat_elem.text = stat
    
    # Convert to string
    rough_string = tostring(svg, 'unicode')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    province = data['province']
    year = int(data['year'])

    prediction, input_data = generate_prediction_with_data(province, year)
    
    if prediction is None:
        return jsonify({"error": "Không tìm thấy dữ liệu cho tỉnh và năm đã chọn."}), 400

    # Determine if this is historical or predicted data
    is_historical = 2004 <= year <= 2024
    data_type = "dữ liệu thực tế" if is_historical else "ước tính"
    
    trend_text = f"""
<div class="trend-text">
    📊 Xu hướng thuận lợi cho di cư 
    <span style="color: {'#28A745' if prediction > 0 else '#007ACC' if prediction == 0 else '#DC3545'}; font-weight: bold;">
        {"vào" if prediction > 0 else "cân bằng" if prediction == 0 else "ra khỏi"}
    </span> 
    khu vực này ({data_type}).
</div>
    """

    explanation = f"""
<div class="data-row">
    <span class="data-label">🏙️ Dân số {'thực tế' if is_historical else 'dự kiến'}:</span>
    <span class="data-value">{int(input_data['population']):,} {'triệu người' if not is_historical else 'người'}</span>
</div>
<div class="data-row">
    <span class="data-label">💰 GRDP bình quân đầu người:</span>
    <span class="data-value">{round(input_data['grdp_per_capita'], 1)} triệu VNĐ/Năm</span>
</div>
<div class="data-row">
    <span class="data-label">💵 Thu nhập trung bình:</span>
    <span class="data-value">{round(input_data['monthly_income_per_capita'], 2)} triệu VNĐ/Tháng</span>
</div>
<div class="data-row">
    <span class="data-label">👥 Mật độ dân số:</span>
    <span class="data-value">{round(input_data['population_density'], 1)} người/km²</span>
</div>
<div class="data-row">
    <span class="data-label">🌡️ Nhiệt độ TB:</span>
    <span class="data-value">{round(input_data['temp_mean'], 2)}°C</span>
</div>
<div class="data-row">
    <span class="data-label">🌧️ Tổng lượng mưa trong năm:</span>
    <span class="data-value">{round(input_data['total_precip'], 1)} mm</span>
</div>
<div class="data-row">
    <span class="data-label">☀️ Tổng số giờ nắng trong năm:</span>
    <span class="data-value">{int(input_data['sunshine_hours']):,} giờ</span>
</div>
    """

    return jsonify({
        "prediction": prediction,
        "trend_text": trend_text.strip(),
        "explanation": explanation.strip(),
        "is_historical": is_historical
    })

@app.route('/predict_all', methods=['POST'])
def predict_all():
    """Generate predictions for all provinces for map visualization"""
    data = request.json
    year = int(data['year'])
    
    predictions = {}
    for province in province_names:
        prediction, _ = generate_prediction_with_data(province, year)
        if prediction is not None:
            # Use the original province name (GeoJSON format) as key
            geojson_province_name = normalize_province_name(province, reverse=True)
            predictions[geojson_province_name] = prediction
    
    return jsonify(predictions)

@app.route('/api/migration-table/<int:year>')
def get_migration_table(year):
    """Get migration data for all provinces in table format"""
    
    try:
        # Determine if this is historical or predicted data
        is_historical = 2004 <= year <= 2024
        
        # Get predictions for all provinces
        migration_data = []
        for province in province_names:
            prediction, data = generate_prediction_with_data(province, year)
            if prediction is not None:
                # Determine trend and color using new red-white-green scheme
                if abs(prediction) <= 0.1:  # Updated balance threshold
                    trend = "Cân bằng"
                    color = "#FFFFFF"
                elif prediction > 20:
                    trend = "Nhập cư rất mạnh"
                    color = "#00CC00"
                elif prediction > 15:
                    trend = "Nhập cư mạnh"
                    color = "#33FF33"
                elif prediction > 10:
                    trend = "Nhập cư vừa"
                    color = "#66FF66"
                elif prediction > 5:
                    trend = "Nhập cư nhẹ"
                    color = "#99FF99"
                elif prediction > 0.1:
                    trend = "Nhập cư rất nhẹ"
                    color = "#CCFFCC"
                elif prediction >= -0.1:
                    trend = "Cân bằng"
                    color = "#FFFFFF"
                elif prediction > -5:
                    trend = "Xuất cư rất nhẹ"
                    color = "#FF9999"
                elif prediction > -10:
                    trend = "Xuất cư nhẹ"
                    color = "#FF6666"
                elif prediction > -15:
                    trend = "Xuất cư vừa"
                    color = "#FF3333"
                elif prediction > -20:
                    trend = "Xuất cư mạnh"
                    color = "#FF0000"
                else:
                    trend = "Xuất cư rất mạnh"
                    color = "#CC0000"
                
                # Handle different data formats for historical vs predicted data
                if is_historical:
                    population = int(data['population'])
                    grdp_per_capita = float(round(data['grdp_per_capita'], 1))
                    monthly_income = float(round(data['monthly_income_per_capita'], 2))
                    population_density = float(round(data['population_density'], 1))
                else:
                    population = int(data['population'])
                    grdp_per_capita = float(round(data['grdp_per_capita'], 1))
                    monthly_income = float(round(data['monthly_income_per_capita'], 2))
                    population_density = float(round(data['population_density'], 1))
                
                migration_data.append({
                    'province': str(province),
                    'migration_rate': float(round(prediction, 2)),
                    'trend': str(trend),
                    'color': str(color),
                    'population': population,
                    'grdp_per_capita': grdp_per_capita,
                    'monthly_income': monthly_income,
                    'population_density': population_density
                })
        
        # Sort by migration rate (descending - highest first)
        migration_data.sort(key=lambda x: x['migration_rate'], reverse=True)
        
        result = {
            'year': year,
            'is_historical': is_historical,
            'data_type': 'Dữ liệu thực tế' if is_historical else 'Ước tính',
            'data': migration_data,
            'total_provinces': len(migration_data),
            'summary': {
                'avg_migration': float(round(sum(d['migration_rate'] for d in migration_data) / len(migration_data), 2)),
                'max_migration': float(max(d['migration_rate'] for d in migration_data)),
                'min_migration': float(min(d['migration_rate'] for d in migration_data)),
                'positive_count': int(sum(1 for d in migration_data if d['migration_rate'] > 0)),
                'negative_count': int(sum(1 for d in migration_data if d['migration_rate'] < 0))
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Failed to generate migration table: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))