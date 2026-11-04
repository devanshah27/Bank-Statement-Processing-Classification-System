/**
 * Bank Statement Processing & Classification System
 * Enterprise Fintech Dashboard Client Engine
 */

(function () {
  "use strict";

  // State Management
  const state = {
    theme: localStorage.getItem("fintech_theme") || "dark",
    sidebarCollapsed: localStorage.getItem("fintech_sidebar_collapsed") === "true",
    tableDensity: localStorage.getItem("fintech_table_density") || "comfortable",
    activeView: "overview",
    charts: {},
    keyBuffer: "",
    keyTimeout: null,
  };

  // -------------------------------------------------------------------------
  // Lifecycle Initialization
  // -------------------------------------------------------------------------
  document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    initSidebar();
    initCommandPalette();
    initKeyboardShortcuts();
    initDropzone();
    initTableDensity();
    initRowClickListeners();
    initCharts();
    initLucide();
    initViewSwitching();
  });

  // Re-run bindings whenever HTMX swaps partial DOM
  document.addEventListener("htmx:afterSwap", (e) => {
    initLucide();
    initRowClickListeners();
    initCharts();
    bindInlinePatchToasts(e.target);
  });

  // -------------------------------------------------------------------------
  // 1. Theme Management (Dark / Light)
  // -------------------------------------------------------------------------
  function initTheme() {
    applyTheme(state.theme);

    const toggleBtns = document.querySelectorAll(".theme-toggle-btn");
    toggleBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const nextTheme = state.theme === "dark" ? "light" : "dark";
        setTheme(nextTheme);
      });
    });
  }

  function setTheme(newTheme) {
    state.theme = newTheme;
    localStorage.setItem("fintech_theme", newTheme);
    applyTheme(newTheme);
    showToast(`Switched to ${newTheme} theme`, "info");
    updateChartsTheme();
  }

  function applyTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    const sunIcons = document.querySelectorAll(".theme-icon-sun");
    const moonIcons = document.querySelectorAll(".theme-icon-moon");

    if (t === "dark") {
      sunIcons.forEach((el) => (el.style.display = "block"));
      moonIcons.forEach((el) => (el.style.display = "none"));
    } else {
      sunIcons.forEach((el) => (el.style.display = "none"));
      moonIcons.forEach((el) => (el.style.display = "block"));
    }
  }

  // -------------------------------------------------------------------------
  // 2. Sidebar Navigation (Expanded / Collapsed / Mobile)
  // -------------------------------------------------------------------------
  function initSidebar() {
    const sidebar = document.getElementById("app-sidebar");
    const collapseBtn = document.getElementById("sidebar-collapse-btn");
    const mobileToggleBtn = document.getElementById("mobile-menu-btn");

    if (!sidebar) return;

    if (state.sidebarCollapsed && window.innerWidth > 768) {
      sidebar.classList.add("collapsed");
    }

    if (collapseBtn) {
      collapseBtn.addEventListener("click", () => {
        sidebar.classList.toggle("collapsed");
        const isCollapsed = sidebar.classList.contains("collapsed");
        state.sidebarCollapsed = isCollapsed;
        localStorage.setItem("fintech_sidebar_collapsed", isCollapsed);
        initLucide();
      });
    }

    if (mobileToggleBtn) {
      mobileToggleBtn.addEventListener("click", () => {
        sidebar.classList.toggle("mobile-open");
      });
    }

    // Close mobile drawer on outside click
    document.addEventListener("click", (e) => {
      if (
        sidebar.classList.contains("mobile-open") &&
        !sidebar.contains(e.target) &&
        mobileToggleBtn &&
        !mobileToggleBtn.contains(e.target)
      ) {
        sidebar.classList.remove("mobile-open");
      }
    });
  }

  // -------------------------------------------------------------------------
  // 3. View Switching (Overview, Transactions, Analytics, Validation, Recurring, Export)
  // -------------------------------------------------------------------------
  function initViewSwitching() {
    const navItems = document.querySelectorAll("[data-nav-view]");
    navItems.forEach((item) => {
      item.addEventListener("click", (e) => {
        const view = item.getAttribute("data-nav-view");
        if (!view) return;

        // If it's a direct page link like "upload", allow default navigation
        if (item.getAttribute("href") && item.getAttribute("href") === "/") {
          return;
        }

        e.preventDefault();
        switchToView(view);
      });
    });
  }

  window.switchToView = function (viewName) {
    state.activeView = viewName;

    // Update active nav in sidebar
    document.querySelectorAll("[data-nav-view]").forEach((el) => {
      if (el.getAttribute("data-nav-view") === viewName) {
        el.classList.add("active");
      } else {
        el.classList.remove("active");
      }
    });

    // Update breadcrumb
    const breadcrumbCurrent = document.getElementById("breadcrumb-current-page");
    if (breadcrumbCurrent) {
      const titles = {
        overview: "Overview Dashboard",
        transactions: "Transaction Ledger",
        analytics: "Analytics & Trends",
        validation: "Ledger Validation & Quality",
        recurring: "Recurring Payments",
        export: "Export & Reporting",
      };
      breadcrumbCurrent.textContent = titles[viewName] || viewName.toUpperCase();
    }

    // Show/hide view sections
    const sections = document.querySelectorAll(".fintech-view-section");
    sections.forEach((sec) => {
      if (sec.id === `view-${viewName}`) {
        sec.style.display = "block";
      } else {
        sec.style.display = "none";
      }
    });

    // If switching to analytics or validation, trigger chart re-render
    if (viewName === "analytics" || viewName === "overview") {
      setTimeout(() => initCharts(), 50);
    }

    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  // -------------------------------------------------------------------------
  // 4. Command Palette (Ctrl+K / Cmd+K)
  // -------------------------------------------------------------------------
  function initCommandPalette() {
    const paletteBackdrop = document.getElementById("cmd-palette-backdrop");
    const cmdInput = document.getElementById("cmd-input");
    const cmdItems = document.querySelectorAll(".cmd-item");

    if (!paletteBackdrop || !cmdInput) return;

    window.openCommandPalette = function () {
      paletteBackdrop.classList.add("active");
      cmdInput.value = "";
      cmdInput.focus();
      filterCmdItems("");
    };

    window.closeCommandPalette = function () {
      paletteBackdrop.classList.remove("active");
    };

    paletteBackdrop.addEventListener("click", (e) => {
      if (e.target === paletteBackdrop) {
        closeCommandPalette();
      }
    });

    cmdInput.addEventListener("input", (e) => {
      filterCmdItems(e.target.value);
    });

    function filterCmdItems(query) {
      const q = query.toLowerCase().trim();
      cmdItems.forEach((item) => {
        const text = item.textContent.toLowerCase();
        if (!q || text.includes(q)) {
          item.style.display = "flex";
        } else {
          item.style.display = "none";
        }
      });
    }

    cmdItems.forEach((item) => {
      item.addEventListener("click", () => {
        const action = item.getAttribute("data-cmd-action");
        executeCmdAction(action);
        closeCommandPalette();
      });
    });
  }

  function executeCmdAction(action) {
    if (!action) return;
    if (action.startsWith("view:")) {
      const view = action.split(":")[1];
      switchToView(view);
    } else if (action === "theme:toggle") {
      setTheme(state.theme === "dark" ? "light" : "dark");
    } else if (action === "upload:goto") {
      window.location.href = "/";
    } else if (action === "export:xlsx") {
      const exportLink = document.querySelector('a[href$="/xlsx"]');
      if (exportLink) exportLink.click();
      else showToast("No active statement to export", "warning");
    } else if (action === "export:csv") {
      const exportLink = document.querySelector('a[href$="/csv"]');
      if (exportLink) exportLink.click();
      else showToast("No active statement to export", "warning");
    } else if (action === "settings:open") {
      toggleSettingsDrawer();
    }
  }

  // -------------------------------------------------------------------------
  // 5. Global Keyboard Shortcuts
  // -------------------------------------------------------------------------
  function initKeyboardShortcuts() {
    document.addEventListener("keydown", (e) => {
      // Ctrl+K / Cmd+K -> Command Palette
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        if (window.openCommandPalette) window.openCommandPalette();
        return;
      }

      // Escape -> close any modal / drawer
      if (e.key === "Escape") {
        if (window.closeCommandPalette) window.closeCommandPalette();
        closeTransactionDrawer();
        closeSettingsDrawer();
        return;
      }

      // Ignore standard input typing
      const activeTag = document.activeElement ? document.activeElement.tagName.toLowerCase() : "";
      if (activeTag === "input" || activeTag === "textarea" || activeTag === "select") {
        return;
      }

      // '/' -> Focus global search
      if (e.key === "/") {
        e.preventDefault();
        const searchInput = document.getElementById("search-input") || document.getElementById("topbar-search-input");
        if (searchInput) searchInput.focus();
        return;
      }

      // Key sequence 'g' then ...
      handleKeySequences(e.key.toLowerCase());
    });
  }

  function handleKeySequences(key) {
    clearTimeout(state.keyTimeout);
    state.keyBuffer += key;

    if (state.keyBuffer === "go") {
      switchToView("overview");
      state.keyBuffer = "";
    } else if (state.keyBuffer === "gt") {
      switchToView("transactions");
      state.keyBuffer = "";
    } else if (state.keyBuffer === "ga") {
      switchToView("analytics");
      state.keyBuffer = "";
    } else if (state.keyBuffer === "gv") {
      switchToView("validation");
      state.keyBuffer = "";
    } else if (state.keyBuffer === "ge") {
      switchToView("export");
      state.keyBuffer = "";
    }

    state.keyTimeout = setTimeout(() => {
      state.keyBuffer = "";
    }, 600);
  }

  // -------------------------------------------------------------------------
  // 6. Toast Notifications System
  // -------------------------------------------------------------------------
  window.showToast = function (message, type = "info", duration = 3500) {
    let container = document.getElementById("toast-container");
    if (!container) {
      container = document.createElement("div");
      container.id = "toast-container";
      container.className = "toast-container";
      document.body.appendChild(container);
    }

    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;

    const icons = {
      success: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="color:var(--emerald)"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`,
      error: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="color:var(--rose)"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`,
      warning: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="color:var(--amber)"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`,
      info: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="color:var(--primary)"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>`,
    };

    toast.innerHTML = `
      ${icons[type] || icons.info}
      <span style="flex:1;">${message}</span>
    `;

    container.appendChild(toast);

    setTimeout(() => {
      toast.style.animation = "toastSlideDown 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards";
      setTimeout(() => toast.remove(), 300);
    }, duration);
  };

  function bindInlinePatchToasts(root) {
    const selects = (root || document).querySelectorAll('select[name="new_category"]');
    selects.forEach((sel) => {
      sel.addEventListener("change", (e) => {
        showToast(`Category updated to "${e.target.value}" & Ledger re-reconciled`, "success");
      });
    });
  }

  // -------------------------------------------------------------------------
  // 7. Transaction Details Slide-out Drawer
  // -------------------------------------------------------------------------
  function initRowClickListeners() {
    const rows = document.querySelectorAll("tr[data-tx-json]");
    rows.forEach((row) => {
      row.addEventListener("click", (e) => {
        // Ignore clicks if clicking inside the category select or interactive button
        if (e.target.closest("select") || e.target.closest("button") || e.target.closest("a")) {
          return;
        }

        try {
          const raw = row.getAttribute("data-tx-json");
          if (!raw) return;
          const tx = JSON.parse(raw);
          openTransactionDrawer(tx);
        } catch (err) {
          console.error("Error opening tx drawer", err);
        }
      });
    });
  }

  window.openTransactionDrawer = function (tx) {
    const overlay = document.getElementById("tx-drawer-overlay");
    const drawer = document.getElementById("tx-details-drawer");
    if (!overlay || !drawer) return;

    // Fill drawer content
    const isCredit = Boolean(tx.credit && parseFloat(tx.credit) > 0);
    const amountVal = isCredit
      ? `+₹${parseFloat(tx.credit).toLocaleString("en-IN", { minimumFractionDigits: 2 })}`
      : `-₹${parseFloat(tx.debit || 0).toLocaleString("en-IN", { minimumFractionDigits: 2 })}`;

    const amountEl = document.getElementById("drawer-tx-amount");
    if (amountEl) {
      amountEl.textContent = amountVal;
      amountEl.className = `drawer-amount-val ${isCredit ? "tx-amount-cr" : "tx-amount-dr"}`;
    }

    const typeBadge = document.getElementById("drawer-tx-type");
    if (typeBadge) {
      typeBadge.textContent = isCredit ? "CREDIT / INFLOW" : "DEBIT / OUTFLOW";
      typeBadge.className = `badge ${isCredit ? "badge-success" : "badge-danger"}`;
    }

    setText("drawer-tx-date", tx.date || "N/A");
    setText("drawer-tx-narration", tx.description || "N/A");
    setText("drawer-tx-merchant", tx.merchant_normalized || tx.description?.split("/")[0] || "Unknown");
    setText("drawer-tx-ref", tx.ref_no || "N/A");
    setText(
      "drawer-tx-balance",
      tx.balance !== null && tx.balance !== undefined
        ? `₹${parseFloat(tx.balance).toLocaleString("en-IN", { minimumFractionDigits: 2 })}`
        : "N/A"
    );
    setText("drawer-tx-category", tx.category || "Other/Uncategorized");
    setText("drawer-tx-source", (tx.source || "rule").toUpperCase());
    setText(
      "drawer-tx-confidence",
      tx.confidence ? `${Math.round(tx.confidence * 100)}%` : "100%"
    );

    // Flags section
    const flagsRow = document.getElementById("drawer-tx-flags-row");
    const flagsVal = document.getElementById("drawer-tx-flags");
    if (flagsRow && flagsVal) {
      if (tx.flag_reasons && tx.flag_reasons.length > 0) {
        flagsRow.style.display = "flex";
        flagsVal.textContent = tx.flag_reasons.join(", ");
      } else {
        flagsRow.style.display = "none";
      }
    }

    // Configure copy button
    const copyBtn = document.getElementById("drawer-copy-narration");
    if (copyBtn) {
      copyBtn.onclick = () => {
        navigator.clipboard.writeText(tx.description || "");
        showToast("Narration copied to clipboard", "success");
      };
    }

    overlay.classList.add("active");
    drawer.classList.add("active");
  };

  window.closeTransactionDrawer = function () {
    const overlay = document.getElementById("tx-drawer-overlay");
    const drawer = document.getElementById("tx-details-drawer");
    if (overlay) overlay.classList.remove("active");
    if (drawer) drawer.classList.remove("active");
  };

  function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  // -------------------------------------------------------------------------
  // 8. Settings Drawer
  // -------------------------------------------------------------------------
  window.toggleSettingsDrawer = function () {
    const drawer = document.getElementById("settings-drawer");
    const overlay = document.getElementById("tx-drawer-overlay");
    if (!drawer) return;

    if (drawer.classList.contains("active")) {
      drawer.classList.remove("active");
      if (overlay) overlay.classList.remove("active");
    } else {
      drawer.classList.add("active");
      if (overlay) overlay.classList.add("active");
    }
  };

  window.closeSettingsDrawer = function () {
    const drawer = document.getElementById("settings-drawer");
    if (drawer) drawer.classList.remove("active");
  };

  // -------------------------------------------------------------------------
  // 9. Table Density Toggle
  // -------------------------------------------------------------------------
  function initTableDensity() {
    setTableDensity(state.tableDensity);

    document.querySelectorAll(".density-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const d = btn.getAttribute("data-density");
        if (d) setTableDensity(d);
      });
    });
  }

  window.setTableDensity = function (density) {
    state.tableDensity = density;
    localStorage.setItem("fintech_table_density", density);

    const tables = document.querySelectorAll(".fintech-table");
    tables.forEach((t) => {
      t.classList.remove("table-comfortable", "table-compact");
      t.classList.add(`table-${density}`);
    });

    document.querySelectorAll(".density-btn").forEach((btn) => {
      if (btn.getAttribute("data-density") === density) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    });
  };

  // -------------------------------------------------------------------------
  // 10. Eye Mask Toggle for Account Number
  // -------------------------------------------------------------------------
  window.toggleAccountMask = function (btn) {
    const valEl = document.getElementById("account-number-display");
    if (!valEl) return;

    const isMasked = valEl.getAttribute("data-masked") === "true";
    const rawVal = valEl.getAttribute("data-raw") || "";
    const maskedVal = valEl.getAttribute("data-mask-str") || "**** **** ****";

    if (isMasked) {
      valEl.textContent = rawVal;
      valEl.setAttribute("data-masked", "false");
      btn.innerHTML = `<i data-lucide="eye-off" style="width:14px;height:14px;"></i> Hide`;
    } else {
      valEl.textContent = maskedVal;
      valEl.setAttribute("data-masked", "true");
      btn.innerHTML = `<i data-lucide="eye" style="width:14px;height:14px;"></i> Show`;
    }
    initLucide();
  };

  // -------------------------------------------------------------------------
  // 11. Upload Studio & Drag-and-Drop
  // -------------------------------------------------------------------------
  function initDropzone() {
    const dropzone = document.getElementById("premium-dropzone");
    const fileInput = document.getElementById("file-input");
    if (!dropzone || !fileInput) return;

    ["dragenter", "dragover"].forEach((evt) => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((evt) => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("dragover");
      });
    });

    dropzone.addEventListener("drop", (e) => {
      if (e.dataTransfer && e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        updateFileListUI(fileInput.files);
        triggerUploadForm();
      }
    });

    dropzone.addEventListener("click", () => {
      fileInput.click();
    });

    fileInput.addEventListener("change", () => {
      if (fileInput.files.length > 0) {
        updateFileListUI(fileInput.files);
        triggerUploadForm();
      }
    });
  }

  function updateFileListUI(files) {
    const queue = document.getElementById("upload-file-queue");
    if (!queue) return;
    queue.innerHTML = "";

    Array.from(files).forEach((f) => {
      const item = document.createElement("div");
      item.className = "meta-field";
      item.style.display = "flex";
      item.style.alignItems = "center";
      item.style.justifyContent = "space-between";
      item.innerHTML = `
        <div style="display:flex; align-items:center; gap:0.5rem;">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="color:var(--primary);"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          <span style="font-weight:600; font-size:0.875rem;">${f.name}</span>
          <span style="font-size:0.75rem; color:var(--text-muted);">(${(f.size / 1024).toFixed(1)} KB)</span>
        </div>
        <span class="badge badge-neutral">Ready to process</span>
      `;
      queue.appendChild(item);
    });
  }

  function triggerUploadForm() {
    const form = document.getElementById("upload-form");
    if (form && typeof htmx !== "undefined") {
      // Advance stepper state
      advanceStepper(2);
      htmx.trigger(form, "submit");
    }
  }

  function advanceStepper(stepIndex) {
    const steps = document.querySelectorAll(".step-item");
    steps.forEach((st, idx) => {
      if (idx < stepIndex) {
        st.classList.add("completed");
        st.classList.remove("active");
      } else if (idx === stepIndex) {
        st.classList.add("active");
        st.classList.remove("completed");
      } else {
        st.classList.remove("active", "completed");
      }
    });
  }

  // -------------------------------------------------------------------------
  // 12. Chart.js Data Visualizations
  // -------------------------------------------------------------------------
  function initCharts() {
    if (typeof Chart === "undefined") return;

    const isDark = state.theme === "dark";
    const textColor = isDark ? "#94a3b8" : "#475569";
    const gridColor = isDark ? "rgba(255, 255, 255, 0.06)" : "rgba(0, 0, 0, 0.06)";
    const tooltipBg = isDark ? "#111827" : "#ffffff";
    const tooltipText = isDark ? "#f8fafc" : "#0f172a";

    // A. Category Spend Doughnut Chart
    const catCanvas = document.getElementById("categoryChart");
    if (catCanvas && !catCanvas.dataset.rendered) {
      const raw = catCanvas.getAttribute("data-chart");
      if (raw) {
        try {
          const parsed = JSON.parse(raw);
          const chart = new Chart(catCanvas, {
            type: "doughnut",
            data: {
              labels: parsed.labels,
              datasets: [
                {
                  data: parsed.values,
                  backgroundColor: [
                    "#6366f1", "#10b981", "#f59e0b", "#f43f5e", "#06b6d4",
                    "#8b5cf6", "#ec4899", "#3b82f6", "#14b8a6", "#f97316",
                    "#84cc16", "#a855f7", "#64748b", "#e11d48", "#0284c7"
                  ],
                  borderWidth: 2,
                  borderColor: isDark ? "#0c1220" : "#ffffff",
                  hoverOffset: 6,
                },
              ],
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              cutout: "68%",
              plugins: {
                legend: {
                  position: "right",
                  labels: {
                    color: textColor,
                    boxWidth: 10,
                    boxHeight: 10,
                    usePointStyle: true,
                    pointStyle: "circle",
                    font: { family: "'Inter', sans-serif", size: 12 },
                    padding: 12,
                  },
                },
                tooltip: {
                  backgroundColor: tooltipBg,
                  titleColor: tooltipText,
                  bodyColor: tooltipText,
                  borderColor: isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)",
                  borderWidth: 1,
                  padding: 10,
                  boxPadding: 4,
                  callbacks: {
                    label: (context) => ` ₹${context.raw.toLocaleString("en-IN")}`,
                  },
                },
              },
            },
          });
          catCanvas.dataset.rendered = "true";
          state.charts.category = chart;
        } catch (e) {
          console.error("Failed to render category chart", e);
        }
      }
    }

    // B. Monthly Inflow vs Outflow Bar Chart
    const monthCanvas = document.getElementById("monthlyChart");
    if (monthCanvas && !monthCanvas.dataset.rendered) {
      const raw = monthCanvas.getAttribute("data-chart");
      if (raw) {
        try {
          const parsed = JSON.parse(raw);
          const chart = new Chart(monthCanvas, {
            type: "bar",
            data: {
              labels: parsed.months,
              datasets: [
                {
                  label: "Inflow (Credits)",
                  data: parsed.credits,
                  backgroundColor: "#10b981",
                  borderRadius: 6,
                  maxBarThickness: 32,
                },
                {
                  label: "Outflow (Debits)",
                  data: parsed.debits,
                  backgroundColor: "#f43f5e",
                  borderRadius: 6,
                  maxBarThickness: 32,
                },
              ],
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: {
                  position: "top",
                  align: "end",
                  labels: {
                    color: textColor,
                    usePointStyle: true,
                    pointStyle: "circle",
                    font: { family: "'Inter', sans-serif", size: 12 },
                  },
                },
                tooltip: {
                  backgroundColor: tooltipBg,
                  titleColor: tooltipText,
                  bodyColor: tooltipText,
                  borderColor: isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)",
                  borderWidth: 1,
                  padding: 10,
                  callbacks: {
                    label: (context) => `${context.dataset.label}: ₹${context.raw.toLocaleString("en-IN")}`,
                  },
                },
              },
              scales: {
                x: {
                  grid: { display: false },
                  ticks: { color: textColor, font: { family: "'Inter', sans-serif" } },
                },
                y: {
                  grid: { color: gridColor },
                  ticks: {
                    color: textColor,
                    font: { family: "'Inter', sans-serif" },
                    callback: (value) => `₹${value.toLocaleString("en-IN")}`,
                  },
                },
              },
            },
          });
          monthCanvas.dataset.rendered = "true";
          state.charts.monthly = chart;
        } catch (e) {
          console.error("Failed to render monthly chart", e);
        }
      }
    }
  }

  function updateChartsTheme() {
    Object.values(state.charts).forEach((c) => {
      if (c && typeof c.destroy === "function") {
        c.destroy();
      }
    });
    state.charts = {};

    const catCanvas = document.getElementById("categoryChart");
    if (catCanvas) catCanvas.removeAttribute("data-rendered");

    const monthCanvas = document.getElementById("monthlyChart");
    if (monthCanvas) monthCanvas.removeAttribute("data-rendered");

    initCharts();
  }

  // -------------------------------------------------------------------------
  // 13. Lucide Icons Renderer
  // -------------------------------------------------------------------------
  function initLucide() {
    if (window.lucide && typeof window.lucide.createIcons === "function") {
      window.lucide.createIcons();
    }
  }

})();
