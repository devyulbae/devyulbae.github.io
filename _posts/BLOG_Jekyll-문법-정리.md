---
title: "Jekyll 문법 정리"
last_modified_at: 2025-02-07
date: 2025-02-07
categories:
  - Blog
tags:
  - Jekyll
  - markdown
---

### 1. _config.yml

1. title: 사이트의 타이틀을 작성합니다. 기본적으로 head에서 사용되지만 우리는 페이지의 title을 사용함으로 이 설정 변수는 사용하고 있지 않습니다.
2. email: 사이트에 이메일을 설정합니다. 우리는 이 설정값으로 문의하기 페이지에서 메일 발송 기능에 사용하고 있습니다. 메일 발송 기능에 관해서는 메일 발송을 참고하세요.
3. description: title과 마찬가지로 페이지 head에 표시될 내용을 작성합니다. 하지만 우리는 페이지의 description을 사용하여 표기함으로 사용하지 않고 있습니다.
4. url: 해당 사이트의 URL입니다. 해당 사이트에 실제 URL(https://dev-yakuza.github.io)을 할당해서 사용합니다. bundle exec jekyll serve를 사용하여 로컬에서 테스트를 하는 경우, jekyll은 해당 URL을 무시하고 http://localhost:4000을 할당합니다. 사이트를 실제 서버에 배포시 bundle exec jekyll build로 빌드할 때, 이 URL이 사용되여 빌드됩니다.
5. author: 사이트의 작성자입니다. head의 author 메타 태그에 사용됩니다.
6. plugins: 사이트에서 사용할 플러그인들을 설정합니다.


2. 변수

- site : 사이트 전반에 대한 정보
- page : 
- content : _layouts

