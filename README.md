# 한밭대학교 SW중심대학 산학연계프로젝트 - 과제명

## **팀 구성**
### 지도교수
 - 박천음 교수님 (국립한밭대학교)

### 기업체 
 - 

### 참여학생
 - 202xxxxx 노동원 
 - 2020xxxx 고동혁
 - 20222039 이현정
 - 202xxxxx 전승재

## Project Background
- ### 필요성
  - 차량은 화재 사고가 빈번하며 인명, 재산 피해로 이어질 수 있으므로 단순 탐지를 넘어 사고 원인 분석 및 보고서 자동 생성 기술이 필요함.
    
- ### 기존 해결책의 문제점
  - 기존 영상 기반 탐지 모델은 객체 인식에는 강점이 있으나, 상황 설명과 인과 관계나 상황에 대한 서술을 하지 않음.
  - 기존 보고서 생성 MAS는 텍스트 기반 태스크에 한정된다.
  
## System Design
  - ### FireReport-MAS
    - **Analysis Agent**: 멀티모달 입력을 받아 각 객체의 상태, 화재 및 연기의 추정 확상 방향, 주변 물체 등 세부적인 상태를 확인하고 주변의 물체와 거리를 판단하여 확산 가능성과 위험도 등 종합적인 상황을 판단한다.
    - **Writer Agent**: Analysis Agent로부터 분석 결과를 입력받아, 사전에 정의딘 보고서 템플릿에 맞추어 자연어 보고서를 작성한다.
    - **Judge Agent**: Writer Agent가 생성한 보고서의 품질을 멀티모달 입력(로그+이미지)를 기반으로 평가하고 검증한다.
    - **Router Agent**: 앞선 모든 데이터(로그+이미지+보고서+평가 점수)를 입력받아 내용을 종합적으로 판단하고 최종 결정을 내린다. 피드백 기반 재순환을 하거나 최종 보고서를 제출한다.
    <img width="700" alt="Image" src="https://github.com/user-attachments/assets/aabf1112-301b-4f26-88aa-25a62b0fa2ff" />
    
## Case Study
  - ### Description
  
  
## Conclusion
  - ### 4단계 에이전트의 역할 분담과 반복적 피드백 워크플로우를 통해, 복잡한 멀티모달 태스크에서 발생하는 단일 VLM의 hallucination을 완화하고 결과 품질을 향상시킨다.
  
## Project Outcome
- ### 2025년 KSC(Korea Software Congress)
