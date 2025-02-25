# Jekyll 테크 블로그 템플릿

기술 블로그를 빠르게 시작하기 위한 Jekyll 템플릿입니다. 개발자를 위한 다양한 기능과 깔끔한 디자인을 제공합니다.

## 📚 목차
- [시작하기](#시작하기)
- [기본 기능](#기본-기능)
- [개발 환경 설정](#개발-환경-설정)
- [커스터마이징](#커스터마이징)
- [문제 해결 및 기여](#문제-해결-및-기여)
- [개발 예정 기능](#개발-예정-기능)

## 🚀 시작하기

1. 이 저장소를 Fork 또는 Clone
2. 저장소 이름을 `{username}.github.io`로 변경
3. `_config.yml` 설정 수정
4. GitHub Pages 활성화

### 디렉토리 구조
```
{username}.github.io/
├── _posts/           # 블로그 포스트
├── _pages/           # 개별 페이지
├── categories/       # 카테고리 페이지
├── search/          # 검색 페이지
├── _data/           # 설정 데이터
├── _includes/       # 재사용 컴포넌트
├── _layouts/        # 페이지 레이아웃
├── assets/          
│   ├── css/         # 스타일시트 (기능별로 분리됨)
│   ├── js/          # 자바스크립트
│   └── images/      # 이미지 파일
│       └── posts/   # 포스트별 이미지 (자동 생성)
└── index.html       # 메인 페이지
```

## 💎 기본 기능

- 📱 반응형 디자인
- 🎨 4가지 테마 지원 (Dark/Light/Blue/Neon)
- 🔍 포스트 검색 기능
- 📑 목차 자동 생성
- 🏷️ 카테고리 시스템
- 📊 GitHub 활동 그래프
- 🌙 테마 전환 버튼
- 📁 포스트별 이미지 폴더 자동 생성

### CSS 구조
- 기능별로 분리된 CSS 파일들
  - base.css: 기본 스타일
  - layout.css: 레이아웃
  - header.css: 헤더 컴포넌트
  - footer.css: 푸터 컴포넌트
  - sidebar.css: 사이드바
  - post.css: 포스트 페이지
  - search.css: 검색 기능
  - toc.css: 목차
  - github-calendar.css: GitHub 캘린더

## ⚙️ 개발 환경 설정

1. Ruby 설치
2. Python 설치 (이미지 관리 자동화를 위해 필요)
3. 의존성 설치:
```
gem install bundler jekyll
bundle install
pip install watchdog  # Python 의존성 설치
```
4. 로컬 서버 실행:
```
./scripts/start
```
이 스크립트는 다음 작업을 수행합니다:
- Jekyll 로컬 서버 실행
- Python 모니터링 스크립트 실행 (포스트 이미지 폴더 자동 생성 및 관리)
- 이미지 메타데이터 CSV 관리

## 🎨 커스터마이징

### _config.yml 설정
```yaml
# 사이트 기본 설정
title: 블로그 제목
description: 블로그 설명
default_theme: dark  # dark, light, blue, neon 중 선택

# 프로필 정보
author:
  name: 이름
  social:
    github: username
    linkedin: username
```

### 포스트 작성
```markdown
---
layout: post
title: "제목"
categories: [카테고리]
---
```

### 이미지 관리
- 포스트 작성 시 자동으로 `assets/images/posts/YYYY-MM-DD-title` 형식의 이미지 폴더가 생성됩니다
- 해당 폴더에 이미지를 넣고 마크다운에서 상대 경로로 참조할 수 있습니다

## 🔧 문제 해결 및 기여

- 이슈는 GitHub Issues에 등록
- 개선 제안은 Pull Request로 제출
- 라이선스: MIT

## 🔜 개발 예정 기능

1. 💬 댓글 시스템
2. 👀 포스트 조회수 추적
3. 📋 코드 블록 복사 버튼
4. 🗺️ SEO 최적화
5. 📱 모바일 메뉴 개선

---
⭐ 이 템플릿이 유용하다면 Star를 눌러주세요!