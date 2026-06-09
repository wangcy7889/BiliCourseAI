    const lightbox = document.getElementById('image-lightbox');
    const lightboxImage = document.getElementById('lightbox-image');
    const closeLightbox = () => {
      lightbox.classList.remove('open');
      lightbox.setAttribute('aria-hidden', 'true');
      lightboxImage.removeAttribute('src');
    };
    document.querySelectorAll('.image-button').forEach((button) => {
      button.addEventListener('click', () => {
        lightboxImage.src = button.dataset.lightboxSrc;
        lightboxImage.alt = button.dataset.lightboxAlt || '';
        lightbox.classList.add('open');
        lightbox.setAttribute('aria-hidden', 'false');
      });
    });
    lightbox.addEventListener('click', (event) => {
      if (event.target === lightbox) closeLightbox();
    });
    document.querySelector('.lightbox-close').addEventListener('click', closeLightbox);
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') closeLightbox();
    });

    const toc = document.querySelector('.toc');
    const tocStateKey = `bilicourse:toc:${window.location.origin}${window.location.pathname}:${document.title}`;

    function findTocLinkForHash(hash) {
      if (!hash) return null;
      return Array.from(document.querySelectorAll('.toc a[href^="#"]'))
        .find((link) => link.getAttribute('href') === hash) || null;
    }

    function readTocState() {
      try {
        return JSON.parse(window.sessionStorage.getItem(tocStateKey) || '{}');
      } catch (error) {
        return {};
      }
    }

    function saveTocState() {
      if (!toc) return;
      const expanded = Array.from(document.querySelectorAll('.toc-toggle[aria-expanded="true"]'))
        .map((button) => button.getAttribute('aria-controls'))
        .filter(Boolean);
      window.sessionStorage.setItem(tocStateKey, JSON.stringify({
        expanded,
        scrollTop: toc.scrollTop
      }));
    }

    const toggleTocNode = (button, forceExpanded = null, persist = true) => {
      const children = document.getElementById(button.getAttribute('aria-controls'));
      if (!children) return;
      const isExpanded = button.getAttribute('aria-expanded') === 'true';
      const nextExpanded = forceExpanded === null ? !isExpanded : Boolean(forceExpanded);
      button.setAttribute('aria-expanded', nextExpanded ? 'true' : 'false');
      button.setAttribute('aria-label', `${nextExpanded ? '收起' : '展开'} ${button.closest('.toc-row')?.querySelector('.toc-node-link')?.textContent?.trim() || '目录节点'}`);
      children.hidden = !nextExpanded;
      if (persist) saveTocState();
    };

    function restoreTocState() {
      const state = readTocState();
      const expanded = new Set(Array.isArray(state.expanded) ? state.expanded : []);
      document.querySelectorAll('.toc-toggle').forEach((button) => {
        if (expanded.has(button.getAttribute('aria-controls'))) {
          toggleTocNode(button, true, false);
        }
      });
    }

    function expandTocAncestorsForHash() {
      const link = findTocLinkForHash(window.location.hash);
      if (!link) return null;
      let node = link.closest('.toc-node') || link.closest('.toc-part');
      while (node) {
        const children = node.parentElement;
        if (children && children.classList.contains('toc-children')) {
          const button = document.querySelector(`.toc-toggle[aria-controls="${children.id}"]`);
          if (button) toggleTocNode(button, true, false);
        }
        node = children ? children.closest('.toc-node') || children.closest('.toc-part') : null;
      }
      return link;
    }

    function restoreTocScroll(targetLink) {
      if (!toc) return;
      const state = readTocState();
      window.requestAnimationFrame(() => {
        if (typeof state.scrollTop === 'number') {
          toc.scrollTop = state.scrollTop;
        } else if (targetLink) {
          targetLink.scrollIntoView({ block: 'center', inline: 'nearest' });
        }
      });
    }

    restoreTocState();
    const hashTocTarget = expandTocAncestorsForHash();
    restoreTocScroll(hashTocTarget);
    window.addEventListener('pagehide', saveTocState);
    window.addEventListener('hashchange', () => {
      const target = expandTocAncestorsForHash();
      restoreTocScroll(target);
      saveTocState();
    });

    document.querySelectorAll('.toc-toggle').forEach((button) => {
      button.addEventListener('click', () => toggleTocNode(button));
    });
    document.querySelectorAll('[data-toc-branch]').forEach((link) => {
      link.addEventListener('click', (event) => {
        event.preventDefault();
        const button = link.closest('.toc-row')?.querySelector('.toc-toggle');
        if (button) toggleTocNode(button);
      });
    });

    const isServedReport = window.location.protocol === 'http:' || window.location.protocol === 'https:';
    let activeNodeAction = false;
    let serverBusySeen = false;
    let serverStatusTimer = null;
    const globalActionStatus = document.querySelector('[data-global-action-status]');

    function setNodeStatus(status, message) {
      if (!status) {
        return;
      }
      status.textContent = message || '';
      status.classList.toggle('has-message', Boolean(message));
    }

    function setAllNodeActionsDisabled(disabled) {
      document.querySelectorAll('.node-action').forEach((item) => {
        item.disabled = disabled;
      });
    }

    async function refreshServerActionStatus() {
      if (!isServedReport) {
        return;
      }
      try {
        const response = await fetch('/api/status', { cache: 'no-store' });
        const payload = await response.json();
        if (!payload.ok) {
          return;
        }
        if (payload.busy) {
          serverBusySeen = true;
          setAllNodeActionsDisabled(true);
          if (globalActionStatus) {
            const actionText = payload.action === 'redo' ? '重做' : '展开';
            const elapsed = Number(payload.elapsed || 0).toFixed(0);
            globalActionStatus.textContent = `后端正在${actionText} ${payload.block_id}，已运行 ${elapsed}s，完成后会自动刷新。`;
          }
          if (!serverStatusTimer) {
            serverStatusTimer = window.setInterval(refreshServerActionStatus, 2500);
          }
          return;
        }
        if (serverBusySeen) {
          if (globalActionStatus) {
            globalActionStatus.textContent = '后端处理完成，正在刷新...';
          }
          saveTocState();
          window.location.reload();
          return;
        }
        if (!activeNodeAction) {
          setAllNodeActionsDisabled(false);
          if (globalActionStatus) {
            globalActionStatus.textContent = '';
          }
        }
        if (serverStatusTimer) {
          window.clearInterval(serverStatusTimer);
          serverStatusTimer = null;
        }
      } catch (error) {
        if (serverStatusTimer) {
          window.clearInterval(serverStatusTimer);
          serverStatusTimer = null;
        }
      }
    }

    refreshServerActionStatus();

    document.querySelectorAll('.node-action').forEach((button) => {
      button.addEventListener('click', async () => {
        if (activeNodeAction) {
          return;
        }
        const container = button.closest('.node-actions') || button.closest('.toc-row');
        const status = container ? container.querySelector('.node-action-status') : null;
        const action = button.dataset.action;
        if (!isServedReport) {
          setNodeStatus(status, '请先在报告目录运行 bilicourse serve .');
          return;
        }
        activeNodeAction = true;
        serverBusySeen = true;
        setAllNodeActionsDisabled(true);
        if (globalActionStatus) {
          globalActionStatus.textContent = '正在处理当前节点，完成刷新后再继续操作。';
        }
        setNodeStatus(status, action === 'redo' ? '正在重做...' : '正在生成...');
        try {
          const response = await fetch(`/api/${action}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              report: button.dataset.report,
              block_id: button.dataset.blockId
            })
          });
          const payload = await response.json().catch(() => ({}));
          if (!response.ok || !payload.ok) {
            throw new Error(payload.error || payload.message || response.statusText || '请求失败');
          }
          setNodeStatus(status, '完成，正在刷新...');
          saveTocState();
          window.location.reload();
        } catch (error) {
          setNodeStatus(status, `失败：${error.message}`);
          activeNodeAction = false;
          setAllNodeActionsDisabled(false);
          if (globalActionStatus) {
            globalActionStatus.textContent = '';
          }
        }
      });
    });
