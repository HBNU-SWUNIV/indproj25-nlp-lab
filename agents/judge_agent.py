import re
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from typing import TypedDict

from multi_agent import GraphState, logger


def judge_agent(state: GraphState) -> dict:
   """
   Judge 에이전트: 작성된 보고서를 평가하고 피드백 제공
   
   역할:
   - 보고서의 사실성, 완전성, 명확성 평가
   - 근거 기반 점수화 (1-5점)
   - 개선을 위한 구체적 피드백 생성
   - 총점 16점 이상 시 승인, 미만 시 재평가 요청
   
   참고: Agent-as-a-Judge (2024)
   """
   logger.info("\n[JUDGER] 보고서 평가 시작...")
   
   report = state["report"]
   log = state["log"]
   image = state["image"]
   feedback = state.get("feedback", "")
   router_decision = state.get("router_decision", "")
   
   # Router로부터 재평가 피드백이 있으면 반영
   if feedback and router_decision == "재평가":
      prompt = PromptTemplate.from_template("""
당신은 보고서 검증 전문가입니다.
Router로부터 재평가 요청을 받았습니다. 아래 피드백을 반영하여 로그와 이미지 정보를 바탕으로 리포트를 보다 엄격하게 재평가하십시오.

1. 로그와 이미지에 자동차 객체가 포함된 화재 또는 연기가 있을 때만 점수 평기
2. 조건 미충족 시: "보고서 미작성: 화재 차량 없음" 형식으로 한 줄 작성


[Router 피드백]
{feedback}

[평가 기준]
각 항목을 1점(매우 낮음) ~ 5점(매우 높음)으로 평가하고 구체적 근거를 제시하십시오.

1. 상황 판단 (5점 만점)
   - 자동차 객체 기준 작성 여부
   - 자동차 개수 정확성

2. 사실 기반 정확성 (5점 만점)
   - 불꽃/연기 위치와 확산 방향의 이미지 일치도
   - 차량 색깔 등 특징 일치도
   - 탐지 로그와 보고서 내용 일치도

3. 객관성 (5점 만점)
   - 주관적 표현 배제 여부
   - 관찰된 사실 기반 서술 여부

4. 보고서 형식 준수 (5점 만점)
   - 필수 항목 포함 여부: 발생일시, 상황 분석, 추가 확산 가능성, 위험도 등급, 대응 조치

[평가 형식]
1. 상황 판단: X점
   근거: ...
   개선 방향: ...

2. 사실 기반 정확성: X점
   근거: ...
   개선 방향: ...

3. 객관성: X점
   근거: ...
   개선 방향: ...

4. 보고서 형식 준수: X점
   근거: ...
   개선 방향: ...

[종합 평가]
총점: XX/20점
우선 개선사항: ...

[입력 데이터]
- 보고서: {report}
- YOLO 탐지 로그: {log}

위 형식에 따라 평가하십시오.
""")
      resp = llm.invoke([
         HumanMessage(
               content=[
                  {"type": "text", "text": prompt.format(feedback=feedback, log=log, report=report)},
                  {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
               ]
         )
      ])
   else:
      prompt = PromptTemplate.from_template("""
당신은 보고서 검증 전문가(Judger)입니다.
YOLO 탐지 로그와 이미지를 바탕으로 리포트를 평가하십시오.

1. 로그와 이미지에 자동차 객체가 포함된 화재 또는 연기가 있을 때만 점수 평기
2. 조건 미충족 시: "보고서 미작성: 화재 차량 없음" 형식으로 한 줄 작성


[평가 기준]
각 항목을 1점(매우 낮음) ~ 5점(매우 높음)으로 평가하고 구체적 근거를 제시하십시오.

1. 상황 판단 (5점 만점)
   - 자동차 객체 기준 작성 여부
   - 자동차 개수 정확성

2. 사실 기반 정확성 (5점 만점)
   - 불꽃/연기 위치와 확산 방향의 이미지 일치도
   - 차량 색깔 등 특징 일치도
   - 탐지 로그와 보고서 내용 일치도

3. 객관성 (5점 만점)
   - 주관적 표현 배제 여부
   - 관찰된 사실 기반 서술 여부

4. 보고서 형식 준수 (5점 만점)
   - 필수 항목 포함 여부: 발생일시, 상황 분석, 추가 확산 가능성, 위험도 등급, 대응 조치

[평가 형식]
1. 상황 판단: X점
   근거: ...

2. 사실 기반 정확성: X점
   근거: ...

3. 객관성: X점
   근거: ...

4. 보고서 형식 준수: X점
   근거: ...

[종합 평가]
총점: XX/20점
우선 개선사항: ...

**중요: 총점을 반드시 "총점: XX/20점" 형식으로 명시하십시오.**

[입력 데이터]
- 보고서: {report}
- YOLO 탐지 로그: {log}

위 형식에 따라 평가하십시오.
""")
      resp = llm.invoke([
         HumanMessage(
               content=[
                  {"type": "text", "text": prompt.format(log=log, report=report)},
                  {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
               ]
         )
      ])
   
   logger.info(f"[JUDGER] 평가 완료")
   logger.info(f"평가 결과:\n{resp.content}\n")
   
   score_match = re.search(r'총점[:\s]*(\d+)\s*/\s*20', resp.content)
   
   total_score = 0
   if score_match:
      total_score = int(score_match.group(1))
      logger.info(f"[JUDGER] 추출된 총점: {total_score}/20점")
   else:
      logger.warning(f"[JUDGER] 총점 추출 실패")
   
   return {
      "evaluation": resp.content
   }