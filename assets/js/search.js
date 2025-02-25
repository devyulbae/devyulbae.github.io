document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('search-input');
    const searchResults = document.getElementById('search-results');
    const clearButton = document.getElementById('clear-search');
    let posts = [];

    // Clear 버튼 표시/숨김 처리
    function toggleClearButton() {
        clearButton.style.display = searchInput.value ? 'block' : 'none';
    }

    // Clear 버튼 클릭 이벤트
    clearButton.addEventListener('click', function() {
        searchInput.value = '';
        searchResults.innerHTML = '';
        toggleClearButton();
        searchInput.focus(); // 입력창에 포커스
    });

    // 입력창 값 변경 감지
    searchInput.addEventListener('input', function() {
        toggleClearButton();
        const searchTerm = this.value.toLowerCase().trim();
        
        if (searchTerm.length < 2) {
            searchResults.innerHTML = '';
            return;
        }

        const results = posts.filter(post => {
            if (!post.title || !post.content) return false;
            
            const titleMatch = post.title.toLowerCase().includes(searchTerm);
            const contentMatch = post.content.toLowerCase().includes(searchTerm);
            return titleMatch || contentMatch;
        });

        console.log('Search results:', results); // 결과 확인
        displaySearchResults(results);
    });

    // 포스트 데이터 가져오기
    fetch('/posts.json')
        .then(response => response.json())
        .then(data => {
            posts = data;
            console.log('Posts loaded:', posts); // 데이터 확인
        })
        .catch(error => console.error('Error loading posts:', error));
});

function displaySearchResults(results) {
    const searchResults = document.getElementById('search-results');
    if (results.length) {
        let output = '<ul class="search-results-list">';
        results.forEach(result => {
            // $ 기호 제거
            const content = result.content.replace(/\$/g, '');
            output += `
                <li class="search-result-item">
                    <a href="${result.url}" class="search-result-title">${result.title}</a>
                    <p class="search-result-excerpt">${content.substring(0, 150)}...</p>
                    <span class="search-result-date">${result.date}</span>
                </li>
            `;
        });
        output += '</ul>';
        searchResults.innerHTML = output;
    } else {
        searchResults.innerHTML = '<p>검색 결과가 없습니다.</p>';
    }
} 