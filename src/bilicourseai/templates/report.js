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

    const toggleTocNode = (button) => {
      const children = document.getElementById(button.getAttribute('aria-controls'));
      if (!children) return;
      const expanded = button.getAttribute('aria-expanded') === 'true';
      button.setAttribute('aria-expanded', expanded ? 'false' : 'true');
      button.setAttribute('aria-label', `${expanded ? '展开' : '收起'} ${button.closest('.toc-row')?.querySelector('.toc-node-link')?.textContent?.trim() || '目录节点'}`);
      children.hidden = expanded;
    };
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
