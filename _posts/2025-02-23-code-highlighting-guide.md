---
title: "코드 하이라이팅 가이드"
date: 2025-02-23
categories: [Guide]
tags: [code, syntax-highlighting, rouge]
---

# 코드 하이라이팅 가이드

블로그에서 코드를 보기 좋게 표시하는 방법을 알아봅시다.

## 1. 인라인 코드 표시하기

문장 중간에 코드를 넣을 때는 백틱(\`)으로 감싸면 됩니다. 예를 들어 ```print("Hello World")``` 처럼 표시됩니다.

## 2. 코드 블록 만들기

여러 줄의 코드를 표시할 때는 세 개의 백틱을 사용합니다. 언어를 명시하면 해당 언어에 맞는 구문 강조가 적용됩니다.

``` python
def hello_world():
    print("Hello, World!")
    return True
```

## 3. 지원하는 언어 목록

Rouge 구문 강조기는 다음과 같은 프로그래밍 언어를 지원합니다:
- Python (python, py)
- JavaScript (javascript, js)
- Ruby (ruby, rb)
- Java (java)
- C++ (cpp)
- CSS (css)
- HTML (html)
- YAML (yaml, yml)
- SQL (sql)
- Shell Script (bash, sh)

## 4. 줄 번호 표시하기

코드 블록에 줄 번호를 추가하려면 다음과 같이 작성합니다:

``` python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

## 5. 특정 줄 강조하기

코드의 특정 줄을 강조하고 싶을 때는 다음과 같이 작성합니다:

``` python
def important_function():
    # 이 줄이 강조됩니다
    critical_operation()
    return True
```

## 6. 파일명 표시하기

코드 블록 위에 파일명을 표시하고 싶을 때는 다음과 같이 작성합니다:

``` python
# app.py
from flask import Flask
app = Flask(__name__)
```

## 7. 긴 코드 줄 처리

화면 너비를 벗어나는 긴 코드 줄은 자동으로 줄 바꿈되거나 가로 스크롤이 생깁니다:

``` python
def very_long_function_name_that_does_something_important(first_parameter, second_parameter, third_parameter, fourth_parameter):
    return "This is a very long line of code that demonstrates how long lines are handled"
```

## 8. 코드 블록 스타일링

코드 블록의 배경색, 글자색, 폰트 등을 커스텀하는 방법도 있습니다. _sass 디렉토리의 설정을 통해 변경할 수 있습니다.

## 마치며

이렇게 코드 하이라이팅을 활용하면 코드를 보기 좋게 표시할 수 있습니다. 프로그래밍 관련 포스트를 작성할 때 유용하게 사용해보세요. 