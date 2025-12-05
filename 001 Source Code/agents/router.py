import re
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage     

from multi_agent import GraphState, logger


def router(state: GraphState) -> dict:
    """
    Router 에이전트: 멀티 에이전트 워크플로우 관리 및 최종 결정
    
    역할:
    - 전체 워크플로우 제어
    - Judger 평가 결과 기반 재분석/재작성/재평가/승인 결정
    - 최종 보고서 생성
    
    참고: ArchiDocGen (2025), MARG (2024)
    """
    logger.info("\n[ROUTER] 최종 판단 시작...")
    
    image = state["image"]
    log = state["log"]
    analysis = state["analysis"]
    report = state["report"]
    evaluation = state["evaluation"]
    iteration = state.get("iteration", 0)
    
    # 최대 3번까지 반복하도록 설정
    MAX_ITERATIONS = 3
    
    if iteration >= MAX_ITERATIONS:
        logger.info(f"[ROUTER] 최대 반복 횟수 도달 ({MAX_ITERATIONS}회) - 강제 승인")
        logger.info("=" * 50)
        logger.info(f"최종 보고서:\n{report}\n")
        logger.info("=" * 50)
        return {
            "final_report": report,
            "router_decision": "승인",
            "feedback": None
        }
    

    score_match = re.search(r'총점[:\s]*(\d+)\s*/\s*20', evaluation)
    total_score = int(score_match.group(1)) if score_match else 0
    
    logger.info(f"[ROUTER] Judger 평가 점수: {total_score}/20점")
    
    prompt = PromptTemplate.from_template("""
당신은 최종 승인 담당자(Router)입니다. 
이미지, 로그, 분석 내용, 보고서, 평가 결과를 종합하여 다음 중 하나를 결정하십시오.

[조건]
1. 로그와 이미지에 차량 객체가 포함된 화재 또는 연기가 있는 경우에만 판단합니다.
2. 평가 점수가 0점이고, 이미지와 로그에서 차량 화재가 없다고 판단되면:
   - 승인으로 판단. 
   - [출력 형식]
    # 차량 화재 보고서:
        "보고서 미작성: 화재 차량 없음"
        
3. 평가 점수가 16점 이상이면:
   - 타당성 확인 후 최종 보고서 출력

[결정 옵션]
- 승인: 보고서가 충분히 우수 → 최종 보고서만 출력
- 재분석: 분석 단계 문제 → Analyser에게 재분석 요청
- 재작성: 보고서 형식/내용 문제 → Writer에게 재작성 요청
- 재평가: 평가 기준 부적절 → Judger에게 재평가 요청

[판단 기준]
1. 총점 16점 이상 → "승인" 우선 고려
2. 총점 16점 미만:
   - "상황 판단" 또는 "사실 기반 정확성" 점수 낮음 → "재분석"
   - "보고서 형식 준수" 점수 낮음 → "재작성"
   - 점수 높지만 이미지/로그와 불일치 → "재평가"

[출력 형식]
- 승인 시: 
# 차량 화재 보고서:
...

- 재분석/재작성/재평가 시: 한 줄씩
  결정: [재분석/재작성/재평가]
  근거: [결정 이유 구체 설명]
  피드백: [개선 방향 3줄 이내]

[입력 데이터]
- YOLO 탐지 로그: {log}
- 분석 내용: {analysis}
- 보고서: {report}
- 평가 결과: {evaluation}
- 평가 총점: {total_score}/20점
- 현재 반복 횟수: {iteration}/3회

위 조건과 형식에 따라 정확히 결정하십시오.
""")
    
    resp = llm.invoke([
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt.format(total_score=total_score, iteration=iteration, log=log, analysis=analysis, report=report, evaluation=evaluation)},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
                ]
            )
        ])
    
    logger.info(f"[ROUTER] 전체 응답: {resp.content}\n")
    
    decision_match = re.search(r'결정[:\s]*(재분석|재작성|재평가)', resp.content)
    feedback_match = re.search(r'피드백[:\s]*(.+?)(?=\n\n|\Z)', resp.content, re.DOTALL)
    
    if decision_match:
        decision = decision_match.group(1)
        feedback_text = feedback_match.group(1).strip() if feedback_match else ""
        
        logger.info(f"[ROUTER] 추출된 결정: {decision}")
        logger.info(f"[ROUTER] 추출된 피드백: {feedback_text}\n")
        
        if decision in ["재분석", "재작성", "재평가"]:
            feedback_to_send = feedback_text if feedback_text != "없음" else evaluation
            return {
                "router_decision": decision,
                "feedback": feedback_to_send
            }
        else:  
            return {
                "final_report": report,
                "router_decision": "승인",
                "feedback": None
            }
            
    else: 
        return {
                "final_report": report,
                "router_decision": "승인",
                "feedback": None
            }