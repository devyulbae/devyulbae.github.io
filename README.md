# Jekyll 테크 블로그 템플릿

기술 블로그를 빠르게 시작하기 위한 Jekyll 템플릿입니다. 이 템플릿은 개발자들이 자신의 기술 블로그를 쉽게 구축할 수 있도록 설계되었습니다.

## 시작하기

1. 이 저장소를 Fork 또는 Clone 합니다.
2. 저장소 이름을 `{username}.github.io`로 변경합니다.
3. `_config.yml`에서 개인 정보를 수정합니다.
4. GitHub Pages를 활성화합니다.

## 특징
- 깔끔한 반응형 디자인
- 개발자 친화적인 마크다운 작성 환경
- 코드 하이라이팅 지원
- 카테고리 및 태그 시스템
- 소셜 미디어 통합
- 댓글 시스템 지원

## 디렉토리 구조

{username}.github.io/
├── _posts/          # 블로그 포스트
├── _pages/          # 개별 페이지
├── _data/          # 설정 데이터
├── _includes/      # 재사용 가능한 컴포넌트
│   ├── header.html  # 네비게이션 바
│   ├── footer.html  # 푸터 (소셜 링크 포함)
│   └── sidebar.html # 사이드바 (프로필, 카테고리, 태그)
├── _layouts/       # 페이지 레이아웃
├── assets/         # 정적 파일
│   ├── css/
│   ├── images/
│   └── js/
├── _config.yml     # 설정 파일
└── index.html      # 메인 페이지

## 커스터마이징

### 1. 기본 설정 (_config.yml)
# 사이트 기본 설정
title: 당신의 블로그 제목
description: 블로그 설명

# 프로필 정보
author:
  name: 이름
  bio: 자기소개
  avatar: /assets/images/profile.jpg
  social:
    github: https://github.com/username
    linkedin: https://linkedin.com/in/username
    instagram: https://instagram.com/username
    email: mailto:your.email@example.com

### 2. 포스트 작성
`_posts` 디렉토리에 `YYYY-MM-DD-제목.md` 형식으로 파일을 생성합니다:

---
layout: post
title: "포스트 제목"
categories: [카테고리]
tags: [태그1, 태그2]
---

포스트 내용

## 로컬 개발 환경 설정

1. Ruby 설치 (Jekyll은 Ruby 기반입니다)
2. 필요한 gem 설치:
gem install bundler jekyll
bundle install

3. 로컬 서버 실행:
bundle exec jekyll serve

4. 브라우저에서 `http://localhost:4000` 접속

## 지원하는 기능
- 📝 마크다운 포스트 작성
- 🏷️ 카테고리 및 태그 관리
- 🔍 검색 기능
- 📱 반응형 디자인
- 💬 댓글 시스템
- 📊 구글 애널리틱스 지원
- 🗺️ SEO 최적화

## 문제 해결
- 이슈는 GitHub Issues에 등록해주세요
- 개선 제안은 Pull Request로 보내주세요

## 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다. 자유롭게 사용, 수정, 배포가 가능합니다.

---
⭐ 이 템플릿이 마음에 드신다면 Star를 눌러주세요!