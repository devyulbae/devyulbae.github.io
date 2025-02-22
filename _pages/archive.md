---
layout: page
title: Archive
permalink: /archive/
---

{% assign postsByYear = site.posts | group_by_exp:"post", "post.date | date: '%Y'" %}
{% for year in postsByYear %}
## {{ year.name }}
{% for post in year.items %}
- {{ post.date | date: "%Y-%m-%d" }} &raquo; [{{ post.title }}]({{ post.url }})
{% endfor %}
{% endfor %} 