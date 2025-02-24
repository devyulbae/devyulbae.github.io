---
layout: post
title: "초안 작성 예시"
categories: [Guide]
tags: [draft, example]
---

# 초안 작성 예시

이 파일은 초안 작성을 위한 예시 파일입니다.

## 초안 작성 방법

1. `_drafts` 폴더에 `제목.md` 형식으로 파일을 생성합니다.
   - 날짜를 파일명에 포함하지 않아도 됩니다.
   - 예: `my-new-post.md`

2. 프론트매터(YAML)를 포함해야 합니다:
   ```yaml
   ---
   layout: post
   title: "포스트 제목"
   categories: [카테고리]
   tags: [태그1, 태그2]
   ---
   ```

3. 초안 미리보기:
   ```bash
   bundle exec jekyll serve --drafts
   ```
   - 초안은 가장 최근 날짜로 표시됩니다.

4. 게시 준비가 되면:
   - `_posts` 폴더로 이동
   - 파일명 앞에 날짜 추가 (YYYY-MM-DD-)

## 마크다운 예시

### 1. 텍스트 강조
- **굵게**
- *기울임*
- ~~취소선~~

### 2. 코드 블록
```python
def hello():
    print("Hello, World!")
```

### 3. 인용
> 인용문을 작성할 수 있습니다.
> 여러 줄도 가능합니다.

### 4. 이미지
![이미지 설명](/assets/images/example.png)

## 작성 팁

1. 초안은 얼마든지 수정 가능합니다.
2. 커밋해도 실제 블로그에는 표시되지 않습니다.
3. 최종 게시 전에 충분히 검토하세요. 