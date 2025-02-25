document.addEventListener('DOMContentLoaded', function() {

    const article = document.querySelector('.post-content');
    const sidebar = document.querySelector('.sidebar-content');
    
    if (!article || !sidebar) return;

    // 헤더 태그들 수집
    const headers = article.querySelectorAll('h1, h2, h3, h4');
    if (headers.length === 0) return;

    // 목차 생성
    const tocContainer = document.createElement('div');
    tocContainer.className = 'toc-container';
    
    const tocTitle = document.createElement('h3');
    tocTitle.textContent = '목차';
    tocContainer.appendChild(tocTitle);

    const tocList = document.createElement('ul');
    tocList.className = 'toc-list';

    headers.forEach((header, index) => {
        if (!header.id) {
            header.id = `toc-${index}`;
        }

        const listItem = document.createElement('li');
        listItem.className = `toc-item toc-${header.tagName.toLowerCase()}`;
        
        const link = document.createElement('a');
        link.href = `#${header.id}`;
        // 클릭 이벤트 추가
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetPosition = header.offsetTop - 100; // 100px 여유 공간
            window.scrollTo({
                top: targetPosition,
                behavior: 'smooth'  // 부드러운 스크롤 효과
            });
        });
        link.textContent = header.textContent;
        
        listItem.appendChild(link);
        tocList.appendChild(listItem);
    });

    tocContainer.appendChild(tocList);
    sidebar.innerHTML = '';
    sidebar.appendChild(tocContainer);

    // 스크롤 시 현재 섹션 하이라이트
    window.addEventListener('scroll', () => {
        const scrollPosition = window.scrollY + 50; // 여유 공간을 고려한 스크롤 위치

        headers.forEach((header, index) => {
            const nextHeader = headers[index + 1];
            const headerTop = header.offsetTop;
            const headerBottom = nextHeader ? nextHeader.offsetTop : document.body.scrollHeight;

            if (scrollPosition >= headerTop && scrollPosition < headerBottom) {
                tocList.querySelectorAll('.active').forEach(item => item.classList.remove('active'));
                tocList.querySelector(`a[href="#${header.id}"]`).parentElement.classList.add('active');
            }
        });
    });
}); 