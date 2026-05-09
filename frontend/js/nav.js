const NAV_ITEMS = [
  { id: 'home',        label: 'Home',        href: '/',                 icon: 'home' },
  { id: 'predict',     label: 'Predict',     href: '/predict.html',     icon: 'predict' },
  { id: 'analysis',    label: 'Analysis',    href: '/analysis.html',    icon: 'chart' },
  { id: 'methodology', label: 'Methodology', href: '/methodology.html', icon: 'book' },
];

const ICONS = {
  home: '<svg class="menu-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>',
  predict: '<svg class="menu-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3"/></svg>',
  chart: '<svg class="menu-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/></svg>',
  book: '<svg class="menu-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>',
  menu: '<svg class="menu-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>',
  logo: '<svg viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M16 4C9.373 4 4 9.373 4 16c0 3.866 1.832 7.305 4.674 9.5C9.5 22 12 20 16 20s6.5 2 7.326 5.5C26.168 23.305 28 19.866 28 16c0-6.627-5.373-12-12-12z" fill="url(#g1)"/><circle cx="12" cy="13" r="1.5" fill="#0a0e14"/><circle cx="20" cy="13" r="1.5" fill="#0a0e14"/><defs><linearGradient id="g1" x1="4" y1="4" x2="28" y2="28"><stop stop-color="#4ade80"/><stop offset="0.5" stop-color="#22d3ee"/><stop offset="1" stop-color="#818cf8"/></linearGradient></defs></svg>',
};

function getCurrentPageId() {
  const path = window.location.pathname;
  if (path === '/' || path.endsWith('/index.html')) return 'home';
  if (path.endsWith('/predict.html'))     return 'predict';
  if (path.endsWith('/analysis.html'))    return 'analysis';
  if (path.endsWith('/methodology.html')) return 'methodology';
  return 'home';
}

function renderNavbar() {
  const currentId = getCurrentPageId();

  const desktopLinks = NAV_ITEMS.map(item => {
    const isActive = item.id === currentId ? 'active' : '';
    return `
      <a href="${item.href}" class="nav-link ${isActive}">
        ${ICONS[item.icon]}
        <span>${item.label}</span>
      </a>
    `;
  }).join('');

  const dropdownLinks = NAV_ITEMS.map(item => {
    const isActive = item.id === currentId ? 'active' : '';
    return `
      <a href="${item.href}" class="${isActive}">
        ${ICONS[item.icon]}
        <span>${item.label}</span>
      </a>
    `;
  }).join('');

  const navHtml = `
    <nav class="nav-shell">
      <div class="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
        <a href="/" class="flex items-center gap-3 text-white no-underline">
          <div style="width:32px;height:32px;">${ICONS.logo}</div>
          <div>
            <div class="font-display text-lg leading-tight">FruitVision</div>
            <div class="text-xs text-gray-500 -mt-0.5 tracking-wide">Quality &amp; Adulteration AI</div>
          </div>
        </a>

        <div class="hidden md:flex items-center gap-1">
          ${desktopLinks}
        </div>

        <div class="md:hidden relative">
          <button id="mobile-menu-btn" class="btn btn-ghost" aria-label="Open menu">
            ${ICONS.menu}
          </button>
          <div id="mobile-menu" class="menu-dropdown">
            ${dropdownLinks}
          </div>
        </div>
      </div>
    </nav>
  `;

  const placeholder = document.getElementById('navbar-root');
  if (placeholder) {
    placeholder.innerHTML = navHtml;
  } else {
    document.body.insertAdjacentHTML('afterbegin', navHtml);
  }

  const menuBtn = document.getElementById('mobile-menu-btn');
  const menu = document.getElementById('mobile-menu');
  if (menuBtn && menu) {
    menuBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      menu.classList.toggle('open');
    });
    document.addEventListener('click', (e) => {
      if (!menu.contains(e.target) && !menuBtn.contains(e.target)) {
        menu.classList.remove('open');
      }
    });
  }
}

function renderFooter() {
  const footerHtml = `
    <footer class="mt-auto border-t border-gray-800/60 bg-black/20">
      <div class="max-w-7xl mx-auto px-6 py-6 flex flex-col md:flex-row items-center justify-between gap-3 text-sm text-gray-500">
        <div class="flex items-center gap-2">
          <span class="pulse-dot"></span>
          <span>FruitVision Demo · Running locally</span>
        </div>
        <div class="font-mono text-xs">
          Built with FastAPI · PyTorch · timm · Tailwind CSS
        </div>
      </div>
    </footer>
  `;
  const placeholder = document.getElementById('footer-root');
  if (placeholder) {
    placeholder.innerHTML = footerHtml;
  } else {
    document.body.insertAdjacentHTML('beforeend', footerHtml);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  renderNavbar();
  renderFooter();
});