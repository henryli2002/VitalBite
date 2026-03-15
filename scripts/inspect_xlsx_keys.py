import pandas as pd
import os
import sys

def inspect_xlsx_files():
    """
    适配 FNDDS 官方 Excel 格式：跳过标题行，提取真实数据表头
    Reads FNDDS XLSX files, skips title rows, and prints actual column headers
    """
    # 替换为你的实际路径（确保路径正确）
    base_path = 'langgraph_app/agents/food_recognition/databases'
    
    if not os.path.exists(base_path):
        print(f"❌ Error: Directory not found at '{base_path}'")
        return

    # 筛选 XLSX 文件，排除临时文件
    xlsx_files = [
        f for f in os.listdir(base_path) 
        if f.endswith('.xlsx') and not f.startswith('~$')
    ]

    if not xlsx_files:
        print(f"⚠️ No XLSX files found in '{base_path}'")
        return

    print(f"🔍 Inspecting FNDDS XLSX files ({len(xlsx_files)} total)...\n")

    # FNDDS Excel 通用配置：跳过前几行标题，找到真实表头
    fndds_config = {
        # 文件名关键词: 跳过的行数（根据你的文件调整）
        "Ingredients": {"skip_rows": 1},       # 配料表跳过前1行
        "Foods and Beverages": {"skip_rows": 1},# 食物表跳过前1行
        "Portions and Weights": {"skip_rows": 1},# 分量表跳过前1行
        "Nutrient Values": {"skip_rows": 1},   # 营养素表跳过前1行
        "default": {"skip_rows": 1}            # 默认跳过前1行
    }

    for file_name in xlsx_files:
        file_path = os.path.join(base_path, file_name)
        print(f"=== File: {file_name} ===")
        
        # 匹配文件类型，确定跳过的行数
        skip_rows = fndds_config["default"]["skip_rows"]
        for key in fndds_config:
            if key in file_name and key != "default":
                skip_rows = fndds_config[key]["skip_rows"]
                break

        try:
            # 1. 先读取所有行，找到真实表头
            # header=None: 不自动设表头，手动指定
            df_raw = pd.read_excel(file_path, header=None)
            
            # 2. 提取真实表头行（跳过标题行后的第一行）
            header_row = skip_rows  # 真实表头所在行（从0开始计数）
            real_headers = df_raw.iloc[header_row].tolist()
            
            # 3. 清理表头：去掉空值、合并单元格的Unnamed、多余空格
            cleaned_headers = []
            for idx, col in enumerate(real_headers):
                # 处理空值/Unnamed
                if pd.isna(col) or str(col).startswith("Unnamed"):
                    # 用列索引命名空列（如 Col_0, Col_1）
                    cleaned_headers.append(f"Col_{idx}")
                else:
                    # 清理字符串（去空格、换行）
                    clean_col = str(col).strip().replace("\n", " ").replace("  ", " ")
                    cleaned_headers.append(clean_col)
            
            # 4. 打印有效表头（过滤纯空列）
            print(f"✅ Skipped {skip_rows} title rows | Real headers:")
            valid_headers = [h for h in cleaned_headers if h != "Col_" and not h.startswith("Col_") or len(h) > 5]
            for idx, header in enumerate(valid_headers, 1):
                print(f"  {idx}. {header}")
            
            # # 5. 可选：打印前2行数据验证
            # print(f"\n📌 First 2 rows of data (preview):")
            # df_data = pd.read_excel(file_path, skiprows=skip_rows+1, header=None, nrows=2)
            # df_data.columns = cleaned_headers  # 用清理后的表头命名
            # print(df_data[valid_headers].to_string(index=False))
            
        except Exception as e:
            print(f"❌ Error processing file: {str(e)}")
            print(f"  Error type: {type(e).__name__}")
        
        print("\n" + "="*60 + "\n")

if __name__ == '__main__':
    # 解决中文/特殊字符打印问题
    sys.stdout.reconfigure(encoding='utf-8')
    inspect_xlsx_files()