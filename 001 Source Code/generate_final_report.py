from openai import OpenAI
import json
import base64
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

DETECTED_DATA = "../data/gold/json/detection_log.json"
API_KEY_FILE = './api_key.json'
# OUTPUT_FILE = "../data/gold/json/final_reports.json"
OUTPUT_FILE = "../data/json/only_image_final_reports.json"
MAX_WORKERS = 3 


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def load_api():
    with open(API_KEY_FILE, 'r') as f:
        config = json.load(f)
        OPEN_API_KEY = config.get("OPEN_API_KEY")
        return OpenAI(api_key=OPEN_API_KEY)


def process_single_item(client, image, log):
    try:
        img_base64 = encode_image(image)
        prompt = f"""
                당신은 차량 화재 감지 시스템의 AI 분석가입니다. 아래 제공된 로그와 이미지를 바탕으로 원칙에 따라 보고서를 작성하시오.
        
    보고서를 작성할 조건:
	1.	이미지에 자동차 객체가 포함된 화재 또는 연기가 있을 때만 작성.
        - 보고서 미작성일 경우 "보고서 미작성: "로 시작하는 한 줄로 간단하게 작성하시오.
            예: "보고서 미작성: 차량 객체 없음", "보고서 미작성: 탐지된 객체 없음"
        
--- 화재 조사 보고서 작성 원칙 ---
    1.	탐지된 객체(class)와 이미지 내 위치 관계를 기반으로 상황에 대해 구체적으로 서술하십시오.
        •	좌표값을 그대로 언급하지 말고, 이미지 내 사물(예: 차량 앞부분, 엔진룸, 뒷좌석, 도로, 배경 건물 등)을 기준으로 표현합니다.
        •   여러 객체가 있을 경우 객체가 특정될 수 있도록 구체적인 표현을 사용하십시오.
        •	예시: “차량 전면부 하단에서 불꽃이 발생했으며, 연기가 차량 상단 방향으로 확산되고 있음.”
       
    2.	객관성을 유지하여 서술하십시오.
        •	“심하게”, “위험해 보임” 등 주관적 표현은 사용하지 않습니다.
        •	관찰된 사실만 간결히 기술합니다.
            
                        
--- 세부내용 작성 원칙 ---
    1.	차량 중심 분석
        •	차량의 구조적 구역(전면부, 엔진룸, 하부, 운전석, 뒷좌석, 후면, 지붕)을 기준으로 탐지 객체 위치를 기술합니다.
        •	차량이 화면에 여러 대 있을 경우, 가장 명확하게 탐지된 차량을 중심으로 기술합니다.
    2.	화재 및 연기 분석
        •	불꽃(fire)의 위치, 크기, 확산 방향을 구체적으로 기술합니다.
            예시: “불꽃은 차량 하부 중앙부에서 시작되어 운전석 방향으로 확산됨.”
        •	연기(smoke)가 있을 경우, 발생 지점과 이동 방향을 함께 기술합니다.
            예시: “연기는 차량 후면에서 발생해 상단으로 확산됨.”
    3.	차량 상태 서술
        •	차량 외관의 손상, 변색, 연소 흔적 등 관찰 가능한 사실을 기술합니다.
            예시: “차량 전면부 도장면에 그을음이 확인됨.”


--- 화재 조사 보고서 형식 ---
    1. 발생일시: 0000년 00월 00일 00시 00분
    2. 세부내용: 
    2-1 상황 분석
    • 화재 및 연기 분석, 차량 상태 서술   
    • 차량의 개수를 포함하여 서술            
    
    2-2 추가 확산 가능성
    • 주변 위험물 존재 여부
    예시: “화염 발생 차량 우측에 인접 차량 1대가 탐지되며, 두 차량 간 거리는 근접 상태로 판단됨. 연기가 두 차량 사이 상단으로 확산되는 양상이 관찰되어, 인접 차량으로 화재 확산 가능성이 있음.”, “배경 내 구조물(건물 벽체, 기둥 등)은 불꽃으로부터 떨어진 위치에 존재하며, 직접적인 화재 확산 가능성은 낮음.”
    
    2-3 위험도 등급
    •	높음 (차량 및 주변 구조물 대비 화재 및 연기 면적이 더 큰 경우, 주변으로 확산 가능성이 존재하는 경우)
    •	중간 (차량의 30% 이상 화재)
    •	낮음 (차량의 일부에서 화재 및 연기 발생)
    
    2-4 이후 대응 조치
    • 2-1, 2-2, 2-3을 종합적으로 판단하여 대응 조치 서술
  
    """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a car fire detection system AI analyst."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                    ],
                },
            ],
        )

        report = response.choices[0].message.content
        report = report.replace("\n", " ")
        print(f"[완료] {image}")
        return {"image": image, "report": report}

    except Exception as e:
        print(e)


def generate_final_report_batch():
    try:
        client = load_api()

        with open(DETECTED_DATA, 'r', encoding='utf-8') as f:
            log_data = json.load(f)

        if not os.path.exists(OUTPUT_FILE):
            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
            f_out.write("[\n")

        first_item = True
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_single_item, client, data["image"], data["log"]) for data in log_data}

            for future in as_completed(futures):
                result = future.result()
                with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
                    if not first_item:
                        f_out.write(",\n")
                    json.dump(result, f_out, ensure_ascii=False, indent=4)
                    first_item = False

        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
            f_out.write("\n]")

        print(f"모든 보고서 생성 완료. 결과: {OUTPUT_FILE}")

    except FileNotFoundError as fnfe:
        logging.error(f"파일을 찾을 수 없습니다: {fnfe}")
    except ValueError as ve:
        logging.error(f"설정 오류: {ve}")
    except Exception as e:
        logging.error(f"예상치 못한 오류 발생: {e}")


# 잘 안나온 리포트 재생성 함수
def generate_single_report():
    try:
        client = load_api()
        
        with open(DETECTED_DATA, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
            
        regen_image = "../data/gold/images/MirrorWEBFire2235_jpg.rf.a902385aee54d27d23be0f020870e969.jpg"
        regen_log = next((item for item in log_data if item["image"] == regen_image), None)
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_single_item, client, regen_image, regen_log)}

            for future in as_completed(futures):
                result = future.result()
                print(result)
                    
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
            f_out.write("\n]")       
            
    except FileNotFoundError as fnfe:
        logging.error(f"파일을 찾을 수 없습니다: {fnfe}")
    except ValueError as ve:
        logging.error(f"설정 오류: {ve}")
    except Exception as e:
        logging.error(f"예상치 못한 오류 발생: {e}")
        
        
        
if __name__ == '__main__':
    generate_final_report_batch()
    # generate_single_report()
