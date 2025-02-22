---
layout: page
permalink: /about/
---

<div class="about-container">
    <section class="about-header">
        <div class="profile-background"></div>
        <img src="{{ site.author.avatar }}" alt="{{ site.author.name }}" class="about-avatar">
        <h1 class="about-name">{{ site.author.name }}</h1>
        <p class="about-title">AI/ML Engineer</p>
        <div class="about-social">
            {% if site.author.social.github %}
            <a href="{{ site.author.social.github }}" target="_blank" class="social-icon">
                <i class="fab fa-github"></i>
            </a>
            {% endif %}
            {% if site.author.social.linkedin %}
            <a href="{{ site.author.social.linkedin }}" target="_blank" class="social-icon">
                <i class="fab fa-linkedin"></i>
            </a>
            {% endif %}
            {% if site.author.social.instagram %}
            <a href="{{ site.author.social.instagram }}" target="_blank" class="social-icon">
                <i class="fab fa-instagram"></i>
            </a>
            {% endif %}
        </div>
    </section>

    <section class="about-intro">
        <div class="section-content">
            <h2>About Me</h2>
            <p class="intro-text">AI/ML 엔지니어로서 데이터를 통해 가치있는 인사이트를 도출하고, 
            실제 비즈니스 문제를 해결하는 것에 큰 관심을 가지고 있습니다.</p>
        </div>
    </section>

    <section class="about-skills">
        <div class="section-content">
            <h2>Technical Skills</h2>
            <div class="skills-grid">
                <div class="skill-card">
                    <div class="skill-icon"><i class="fas fa-brain"></i></div>
                    <h3>Machine Learning</h3>
                    <ul>
                        <li>TensorFlow</li>
                        <li>PyTorch</li>
                        <li>Scikit-learn</li>
                    </ul>
                </div>
                <div class="skill-card">
                    <div class="skill-icon"><i class="fas fa-code"></i></div>
                    <h3>Programming</h3>
                    <ul>
                        <li>Python</li>
                        <li>R</li>
                        <li>SQL</li>
                    </ul>
                </div>
                <div class="skill-card">
                    <div class="skill-icon"><i class="fas fa-database"></i></div>
                    <h3>Data Engineering</h3>
                    <ul>
                        <li>Pandas</li>
                        <li>PySpark</li>
                        <li>BigQuery</li>
                    </ul>
                </div>
            </div>
        </div>
    </section>

    <section class="about-experience">
        <div class="section-content">
            <h2>Experience</h2>
            <div class="timeline">
                <div class="timeline-item">
                    <div class="timeline-dot"></div>
                    <div class="timeline-content">
                        <h3>AI/ML Engineer</h3>
                        <p class="timeline-date">2023 - Present</p>
                        <p class="timeline-company">Company Name</p>
                        <ul>
                            <li>주요 프로젝트 및 성과 1</li>
                            <li>주요 프로젝트 및 성과 2</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    </section>
</div> 