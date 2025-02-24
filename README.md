# Jekyll 테크 블로그 템플릿

기술 블로그를 빠르게 시작하기 위한 Jekyll 템플릿입니다. 개발자를 위한 다양한 기능과 자동화된 이미지 관리 시스템을 제공합니다.

## 📚 목차
- [시작하기](#시작하기)
- [기본 기능](#기본-기능)
- [Post Manager 시스템](#post-manager-시스템)
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
├── .github/
│   └── workflows/    # GitHub Actions 자동화
├── _posts/           # 블로그 포스트
├── _pages/           # 개별 페이지
├── categories/       # 카테고리 페이지
├── tags/            # 태그 페이지
├── _data/           # 설정 데이터
├── _includes/       # 재사용 컴포넌트
├── _layouts/        # 페이지 레이아웃
├── assets/          # 정적 파일
│   ├── css/
│   ├── images/
│   └── js/
├── scripts/         # 자동화 스크립트
├── _config.yml      # 설정 파일
├── Gemfile          # Ruby 의존성 관리
└── index.html       # 메인 페이지
```

## 💎 기본 기능

- 📱 깔끔한 마크다운 작성 환경
- 🎨 4가지 테마 지원 (Dark/Light/Blue/Neon)
- 🏷️ 카테고리 및 태그 시스템
- 📁 자동화된 이미지 관리 시스템

## 🤖 Post Manager 시스템

### Post Manager란?

자동화된 이미지 관리 시스템으로 다음 기능을 제공합니다:

- 📁 포스트별 자동 이미지 폴더 생성
- 📊 이미지 현황 자동 추적 (posts.csv)
- 🔄 일관된 이미지 파일 구조 유지
- 🔍 기존 이미지 자동 정리

### 실행 방법

**Linux/Mac:**
*****
chmod +x scripts/start.sh  # 최초 1회 실행
./scripts/start.sh
*****

**Windows:**
*****
scripts\start.bat
*****

### 자동화 기능

1. **로컬 개발 시**
   - 스크립트 실행으로 자동 처리
   - 실시간 변경사항 감지

2. **GitHub Push 시**
   - GitHub Actions 자동 실행
   - 폴더/파일 자동 생성
   - posts.csv 자동 업데이트

3. **이미지 자동 정리**
   - 기존 이미지 자동 감지 및 이동
   - 파일명 자동 변경
   - 포스트 내 경로 자동 수정

### 이미지 관리 예시

*****
# 변경 전
assets/images/
├── example1.png
└── example2.png

# 변경 후
assets/images/
└── 001_post_title/
    ├── 01.png
    └── 02.png
*****

## ⚙️ 개발 환경 설정

1. Ruby 설치
2. 의존성 설치:
*****
gem install bundler jekyll
bundle install
*****

3. 로컬 서버 실행:
   - ⚠️ 반드시 제공된 스크립트 사용
   - `http://localhost:4000` 접속

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
  bio: 자기소개
  social:
    github: username
    linkedin: username
```

### 📖 가이드 포스트 시리즈

시작하기 전에 다음 순서로 가이드를 읽어주세요:

1. [시작하기] Jekyll 블로그 설정하기
2. [기본기] 마크다운 작성 가이드
3. [기본기] 테마 변경과 커스터마이징
4. [중급] 이미지 관리와 Post Manager 시스템
5. [중급] 카테고리와 태그 활용하기
6. [고급] GitHub Actions 이해하기

초안 작성은 `_drafts` 폴더에 할 수 있습니다:
- 파일명: `포스트-제목.md` (날짜 없이)
- 초안 미리보기: `bundle exec jekyll serve --drafts`

### 포스트 작성
```markdown
---
layout: post
title: "제목"
categories: [카테고리]
tags: [태그1, 태그2]
---
```

## 🔧 문제 해결 및 기여

- 이슈는 GitHub Issues에 등록
- 개선 제안은 Pull Request로 제출
- 라이선스: MIT

### ⚠️ 주의사항

- Jekyll 서버는 반드시 제공된 스크립트로 실행
- 이미지 파일명은 순차적 번호 사용
- 자동 생성된 폴더 구조 임의 변경 금지
- GitHub Actions 실행 중 수동 push 자제

---
⭐ 이 템플릿이 유용하다면 Star를 눌러주세요!

## 🔜 개발 예정 기능

순차적으로 개발될 예정인 기능들입니다:

1. 💬 GitHub 계정 기반 댓글 시스템
2. 👀 포스트 조회수 추적
3. 🔍 검색 기능
4. 🗺️ SEO 최적화
5. 📋 코드 블록에 칸 복사 버튼 넣어주기
6. 

각 기능은 개발 완료 시 별도의 포스트를 통해 사용 방법을 안내할 예정입니다.