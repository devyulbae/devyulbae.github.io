---
title: "MathJax로 수식 작성하기"
date: 2025-02-23
categories: [Guide]
tags: [mathjax, mathematics, latex]
---

# MathJax로 수식 작성하기

블로그에서 수학 수식을 작성하는 방법을 알아봅시다.

## 1. 인라인 수식

문장 중간에 수식을 넣을 때도 `$$...$$` 를 사용합니다.

예시:
- 질량-에너지 등가식 $$E = mc^2$$ 처럼 간단한 수식
- 피타고라스의 정리 $$a^2 + b^2 = c^2$$
- 제곱근은 $$\sqrt{2}$$ 처럼 표현

<br>

## 2. 디스플레이 수식

별도의 줄에 수식을 크게 표시할 때도 동일하게 `$$...$$` 를 사용합니다.

예시:

$$
\frac{-b \pm \sqrt{b^2-4ac}}{2a}
$$

$$
e^{i\pi} + 1 = 0
$$

```
$$
\frac{-b \pm \sqrt{b^2-4ac}}{2a}
$$

$$
e^{i\pi} + 1 = 0
$$
```

<br>

## 3. 분수와 지수

분수와 지수는 다음과 같이 작성합니다:

$$
y = \frac{1}{2}x^2 + \frac{x}{3} + \frac{1}{4}
$$

중첩된 분수도 가능합니다:

$$
\frac{1}{\frac{1}{2} + \frac{1}{3}}
$$

```
$$
y = \frac{1}{2}x^2 + \frac{x}{3} + \frac{1}{4}
$$

$$
\frac{1}{\frac{1}{2} + \frac{1}{3}}
$$
```

<br>

## 4. 시그마와 적분

시그마와 적분은 다음과 같이 표현합니다:

시그마:

$$
\sum_{i=1}^{n} i = \frac{n(n+1)}{2}
$$

적분:

$$
\int_{0}^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
$$

```
$$
\sum_{i=1}^{n} i = \frac{n(n+1)}{2}
$$

$$
\int_{0}^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
$$
```

<br>

## 5. 행렬

행렬은 다음과 같이 작성합니다:

$$
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix} =
\begin{bmatrix}
ax + by \\
cx + dy
\end{bmatrix}
$$

```
$$
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix} =
\begin{bmatrix}
ax + by \\
cx + dy
\end{bmatrix}
$$
```

<br>

## 6. 수학 기호 모음

자주 사용되는 수학 기호들:

- 그리스 문자: $$\alpha, \beta, \gamma, \theta, \pi$$
- 관계 연산자: $$\leq, \geq, \neq, \approx, \equiv$$
- 집합 기호: $$\in, \notin, \subset, \supset, \cup, \cap$$
- 화살표: $$\rightarrow, \leftarrow, \Rightarrow, \Leftarrow$$

```
- 그리스 문자: $$\alpha, \beta, \gamma, \theta, \pi$$
- 관계 연산자: $$\leq, \geq, \neq, \approx, \equiv$$
- 집합 기호: $$\in, \notin, \subset, \supset, \cup, \cap$$
- 화살표: $$\rightarrow, \leftarrow, \Rightarrow, \Leftarrow$$
```

<br>

## 7. 여러 줄 수식

align 환경을 사용하여 여러 줄의 수식을 정렬할 수 있습니다:

$$
\begin{align}
(x+1)^2 &= (x+1)(x+1) \\
&= x^2 + 2x + 1
\end{align}
$$

```
$$
\begin{align}
(x+1)^2 &= (x+1)(x+1) \\
&= x^2 + 2x + 1
\end{align}
$$
```

<br>

## 8. 케이스 분류

cases 환경으로 조건에 따른 다른 수식을 표현합니다:

$$
|x| = \begin{cases}
x & \text{if } x \geq 0 \\
-x & \text{if } x < 0
\end{cases}
$$

```
$$
|x| = \begin{cases}
x & \text{if } x \geq 0 \\
-x & \text{if } x < 0
\end{cases}
$$
```

<br>

## 9. 색상 사용

수식에 색상을 입힐 수 있습니다:

$$
\color{red}{E} = \color{blue}{m}\color{green}{c^2}
$$

```
$$
\color{red}{E} = \color{blue}{m}\color{green}{c^2}
$$
```

<br>

## 10. 수식 번호 사용

수식에 번호를 붙이고 참조할 수 있습니다:

$$
\begin{equation}
\label{eq:einstein}
E = mc^2
\end{equation}
$$

```
$$
\begin{equation}
\label{eq:einstein}
E = mc^2
\end{equation}
```

<br>

## 마치며

이제 MathJax를 사용하여 다양한 수학 수식을 표현할 수 있습니다. 수학이나 과학 관련 포스트를 작성할 때 유용하게 활용해보세요!
