---
title: "마크다운 표 스타일링 가이드"
date: 2025-02-23
categories: [Guide]
tags: [markdown, css, table]
---

# 마크다운 표 스타일링 가이드

블로그에서 표를 더 예쁘고 효과적으로 사용하는 방법을 알아봅시다.

## 1. 기본 표 만들기

기본적인 표 스타일입니다:

![기본 표 스타일](/assets/images/003_table_guide/01.png)

위 표를 만드는 코드입니다:

*****
| 제품명 | 가격 | 재고 |
|--------|------|------|
| 노트북 | 1,200,000원 | 5개 |
| 스마트폰 | 800,000원 | 10개 |
| 태블릿 | 500,000원 | 8개 |
*****

## 2. 정렬된 표 만들기

열마다 다른 정렬을 적용한 표입니다:

![정렬된 표 스타일](/assets/images/003_table_guide/02.png)

정렬을 지정하는 코드입니다:

*****
| 제품명 | 판매량 | 매출액 |
|:-------|:------:|--------:|
| 노트북 | 50대 | ₩60,000,000 |
| 스마트폰 | 100대 | ₩80,000,000 |
| 태블릿 | 80대 | ₩40,000,000 |
*****

## 3. 강조된 표 만들기

중요한 정보를 강조 표시한 표입니다:

![강조된 표 스타일](/assets/images/003_table_guide/03.png)

강조 표를 만드는 HTML 코드입니다:

*****
<table>
  <tr>
    <th>분기</th>
    <th>목표</th>
    <th>달성</th>
    <th>달성률</th>
  </tr>
  <tr class="highlight">
    <td>1분기</td>
    <td>1,000만원</td>
    <td>1,200만원</td>
    <td>120%</td>
  </tr>
  <tr>
    <td>2분기</td>
    <td>1,500만원</td>
    <td class="highlight">1,800만원</td>
    <td>120%</td>
  </tr>
</table>
*****

## 4. 크기가 다른 표 만들기

### 4.1 작은 표

간단한 정보를 위한 작은 표입니다:

![작은 표 스타일](/assets/images/003_table_guide/04.png)

*****
<table class="table-small">
  <tr>
    <th>날짜</th>
    <th>일정</th>
  </tr>
  <tr>
    <td>3월 1일</td>
    <td>프로젝트 시작</td>
  </tr>
  <tr>
    <td>3월 15일</td>
    <td>중간 점검</td>
  </tr>
</table>
*****

### 4.2 큰 표

상세한 정보를 위한 큰 표입니다:

![큰 표 스타일](/assets/images/003_table_guide/05.png)

*****
<table class="table-large">
  <tr>
    <th>프로젝트명</th>
    <th>담당자</th>
    <th>시작일</th>
    <th>종료일</th>
    <th>진행상태</th>
  </tr>
  <tr>
    <td>웹사이트 리뉴얼</td>
    <td>김철수</td>
    <td>2025-01-01</td>
    <td>2025-03-31</td>
    <td>진행중</td>
  </tr>
</table>
*****

## 5. 반응형 표 만들기

### 데스크톱 화면에서의 모습:
![반응형 표 데스크톱](/assets/images/003_table_guide/06.png)

### 모바일 화면에서의 모습:
![반응형 표 모바일](/assets/images/003_table_guide/07.png)

반응형 표를 만드는 코드입니다:

*****
<table class="responsive-table">
  <tr>
    <th>상품명</th>
    <th>가격</th>
    <th>할인율</th>
    <th>할인가</th>
  </tr>
  <tr>
    <td data-label="상품명">울트라 노트북</td>
    <td data-label="가격">2,000,000원</td>
    <td data-label="할인율">20%</td>
    <td data-label="할인가">1,600,000원</td>
  </tr>
</table>
*****

## CSS 스타일 모음

모든 표 스타일링에 필요한 CSS 코드입니다:

*****
/* 기본 표 스타일 */
table {
    width: 100%;
    margin: 1rem 0;
    border-collapse: collapse;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
}

/* 강조 스타일 */
tr.highlight {
    background-color: var(--highlight-bg);
}

td.highlight {
    font-weight: bold;
    color: var(--highlight-color);
}

/* 크기 변형 스타일 */
.table-small {
    width: 50%;
    margin: 1rem auto;
    font-size: 0.9em;
}

.table-large {
    width: 100%;
    margin: 1.5rem auto;
    font-size: 1.2em;
}

/* 반응형 스타일 */
@media (max-width: 768px) {
    .responsive-table thead {
        display: none;
    }
    
    .responsive-table td {
        display: block;
        text-align: right;
        padding-left: 50%;
        position: relative;
    }
    
    .responsive-table td::before {
        content: attr(data-label);
        position: absolute;
        left: 0;
        width: 50%;
        padding-left: 15px;
        font-weight: bold;
        text-align: left;
    }
}
*****

## 마치며

이제 다양한 상황에 맞는 표를 만들 수 있습니다. 각 예시의 실제 모습을 참고하여 원하는 스타일의 표를 만들어보세요! 