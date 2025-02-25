document.addEventListener('DOMContentLoaded', () => {
    // config에서 설정한 default_theme 가져오기 (없으면 'dark'로 설정)
    const defaultTheme = document.documentElement.getAttribute('data-default-theme') || 'dark';
    
    // localStorage에 저장된 테마가 없으면 default_theme 사용
    const savedTheme = localStorage.getItem('default_theme');
    const themeToApply = savedTheme || defaultTheme;
    
    // data-theme 속성 설정
    document.documentElement.setAttribute('data-theme', themeToApply);
    console.log('Initial theme:', themeToApply);
});

function toggleTheme() {
    const themes = ['dark', 'light', 'blue', 'neon'];
    const currentTheme = document.documentElement.getAttribute('data-theme');
    
    let currentIndex = themes.indexOf(currentTheme);
    if (currentIndex === -1) currentIndex = 0;
    
    const nextIndex = (currentIndex + 1) % themes.length;
    const newTheme = themes[nextIndex];
    
    console.log('Switching from', currentTheme, 'to', newTheme);
    
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
} 