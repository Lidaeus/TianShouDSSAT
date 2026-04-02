import os
import requests
import pandas as pd
from datetime import datetime

def fetch_nasa_power(lat, lon, start_date, end_date):
    """从 NASA POWER API 获取日尺度气象数据"""
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M_MAX,T2M_MIN,ALLSKY_SFC_SW_DWN,PRECTOTCORR",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON"
    }
    print(f"正在请求 NASA POWER 数据 ({lat}, {lon}) 从 {start_date} 到 {end_date}...")
    response = requests.get(url, params=params, timeout=30)
    if response.status_code != 200:
        raise Exception(f"API 请求失败: {response.text}")
    return response.json()

def json_to_dssat_wth(json_data, station_id="NCP1"):
    """将 JSON 转换为 DSSAT .WTH 格式字符串，严格遵守固定列宽"""
    features = json_data['properties']['parameter']
    dates = list(features['T2M_MAX'].keys())
    
    # 严格遵循 ICASA 对齐标准
    header = "*WEATHER DATA : North China Plain Core\n\n"
    header += "@ INSI      LAT     LONG  ELEV   TAV   TAMP   REFHT   WNDHT\n"
    
    lat = json_data['geometry']['coordinates'][1]
    lon = json_data['geometry']['coordinates'][0]
    
    # 格式说明: INSI(4位) LAT(8位) LONG(8位) ELEV(5位) TAV(5位) TAMP(6位) REFHT(7位) WNDHT(7位)
    header += f"  {station_id[:4]:4}  {lat:7.3f} {lon:8.3f}   -99.   15.0   15.0    2.0    2.0\n"
    header += "@DATE  SRAD  TMAX  TMIN  RAIN\n"
    
    body = ""
    for d_str in dates:
        dt = datetime.strptime(d_str, "%Y%m%d")
        yydoy = dt.strftime("%y%j")
        
        srad = features['ALLSKY_SFC_SW_DWN'].get(d_str, -99.0)
        tmax = features['T2M_MAX'].get(d_str, -99.0)
        tmin = features['T2M_MIN'].get(d_str, -99.0)
        rain = features['PRECTOTCORR'].get(d_str, -99.0)
        
        srad = max(0, srad)
        # DATE(5位) + SRAD(6位) + TMAX(6位) + TMIN(6位) + RAIN(6位)
        body += f"{yydoy}{srad:6.1f}{tmax:6.1f}{tmin:6.1f}{rain:6.1f}\n"
    
    return header + body

def main():
    # 华北平原核心区
    LAT, LON = 35.0, 115.0
    # 抓取过去 30 年 (为了节省 API 配额，先分段抓取 1991-2021)
    # 注意：NASA POWER API 每次请求有限制，我们分 10 年一组
    
    output_dir = "expert_factory/data/weather"
    os.makedirs(output_dir, exist_ok=True)
    
    periods = [
        ("19910101", "20001231"),
        ("20010101", "20101231"),
        ("20110101", "20211231")
    ]
    
    all_content = ""
    for start, end in periods:
        try:
            data = fetch_nasa_power(LAT, LON, start, end)
            wth_content = json_to_dssat_wth(data)
            # 合并逻辑：去除后续 header 的元数据行
            if all_content == "":
                all_content = wth_content
            else:
                body_only = "\n".join(wth_content.split("@DATE  SRAD  TMAX  TMIN  RAIN\n")[1:])
                all_content += body_only
        except Exception as e:
            print(f"抓取 {start}-{end} 失败: {e}")
            
    with open(os.path.join(output_dir, "UFGA_NCP.WTH"), "w") as f:
        f.write(all_content)
    print(f"数据抓取完成！保存至: {output_dir}/UFGA_NCP.WTH")

if __name__ == "__main__":
    main()
