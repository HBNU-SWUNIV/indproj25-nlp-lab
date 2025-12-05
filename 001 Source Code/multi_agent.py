from langgraph.graph import StateGraph, END
import json, logging
import re
from generate_final_report import encode_image, load_api
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import TypedDict, Any, Optional, Literal
from langchain_core.messages import HumanMessage
from agents.analysis_agent import analyser_agent
from agents.writer_agent import writer_agent
from agents.judge_agent import judge_agent
from agents.router import router


API_KEY_FILE = "../config/api_key.json"
DETECTED_DATA = "../data/gold/json/detection_log.json"
MA_FINAL_REPORT = "../data/gold/json/ma_final_reports.json"
MA_REPORT = "../data/gold/json/ma_reports.json"

client = load_api()
llm = ChatOpenAI(temperature=0, openai_api_key=client.api_key, model="gpt-4o-mini")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(MA_REPORT, encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"사용 모델: {llm.model_name}")
logger.info("=" * 50)

    
class GraphState(TypedDict):
    """
    멀티 에이전트 그래프의 상태를 정의합니다.

    Attributes:
        image (str): Base64로 인코딩된 이미지 문자열
        log (Any): YOLO 탐지 로그 데이터
        analysis (Optional[str]): Analyser의 분석 결과
        report (Optional[str]): Writer가 작성한 보고서 초안
        evaluation (Optional[str]): Judge의 평가 결과
        feedback (Optional[str]): Router의 피드백
        router_decision (Optional[str]): Router의 결정 (재분석/재작성/재평가/승인)
        iteration (int): 현재 반복 횟수
        final_report (Optional[str]): Router가 승인한 최종 보고서
    """
    image: str
    log: Any 
    analysis: Optional[str]
    report: Optional[str]
    evaluation: Optional[str]
    feedback: Optional[str]
    router_decision: Optional[str]
    iteration: int
    final_report: Optional[str]


def select_decision(state: GraphState) -> Literal["analyser", "writer", "judger", "end"]:
    """
    라우팅 조건 함수: Router의 결정에 따라 다음 노드 결정
    """
    router_decision = state.get("router_decision", "승인")
    iteration = state.get("iteration", 0)
    MAX_ITERATIONS = 3
    
    if iteration >= MAX_ITERATIONS:
        logger.info(f"[ROUTING] 최대 반복 도달 - 종료")
        return "end"
    
    if router_decision == "재분석":
        logger.info(f"[ROUTING] Router 결정: 재분석 → Analyser로 이동")
        return "analyser"
    elif router_decision == "재작성":
        logger.info(f"[ROUTING] Router 결정: 재작성 → Writer로 이동")
        return "writer"
    elif router_decision == "재평가":
        logger.info(f"[ROUTING] Router 결정: 재평가 → Judger로 이동")
        return "judger"
    else:  
        logger.info(f"[ROUTING] Router 결정: 승인 → 종료")
        return "end"


def generate_ma_report():
    with open(DETECTED_DATA, 'r', encoding='utf-8') as f:
        log_data = json.load(f)
    results = []

    for idx, data in enumerate(log_data):
        logger.info(f"처리 중: {idx+1}/{len(log_data)}")
        state = {
            "image": encode_image(data["image"]),
            "log": data["log"],
            "iteration": 0
        }

        while True:
            result = graph.invoke(state)
            decision = result.get("router_decision", "승인")

            if decision == "승인":
                results.append({
                    "image": data["image"],
                    "report": result["final_report"]
                })
                break
            else:
                state["feedback"] = result.get("feedback", "")
                state["router_decision"] = decision
                state["iteration"] = state.get("iteration", 0) + 1
                logger.info(f"[WORKFLOW] 반복 {state['iteration']}회 - 다음 단계: {decision}")

    with open(MA_FINAL_REPORT, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"최종 보고서 {len(results)}건 저장 완료: {MA_FINAL_REPORT}")



workflow = StateGraph(GraphState)

workflow.add_node("analyser", analyser_agent)
workflow.add_node("writer", writer_agent)
workflow.add_node("judge", judge_agent)
workflow.add_node("router", router)

workflow.set_entry_point("analyser")

workflow.add_edge("analyser", "writer")
workflow.add_edge("writer", "judge")
workflow.add_edge("judge", "router")
graph = workflow.compile()

workflow.add_conditional_edges(
    "router",
    select_decision,
    {
        "analyser": "analyser", 
        "writer": "writer",     
        "judge": "judge",      
        "end": END              
    }
)


if __name__ == "__main__":
    generate_ma_report()