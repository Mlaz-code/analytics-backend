// Pikkit Validation Review UI
// Generates validation reports and manages interactive corrections

const API_BASE = '/api/validation';
let currentReport = null;
let selectedCorrections = new Map();

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    // Don't auto-generate on page load - user clicks button to load reports
});

function setupEventListeners() {
    document.getElementById('load-report-btn').addEventListener('click', loadReportList);
    document.getElementById('refresh-btn').addEventListener('click', refreshReport);
    document.getElementById('apply-btn').addEventListener('click', reviewAndApply);
    document.getElementById('clear-selection-btn').addEventListener('click', clearSelection);
    document.getElementById('confirm-ok').addEventListener('click', confirmAction);
    document.getElementById('confirm-cancel').addEventListener('click', closeModal);
}

async function loadReportList() {
    try {
        const btn = document.getElementById('load-report-btn');
        const originalText = btn.textContent;
        btn.disabled = true;
        btn.textContent = '⏳ Generating full report (all bets)...';

        // Generate fresh report with ALL bets (no limit for full production data)
        const generateResponse = await fetch(`${API_BASE}/generate`);
        if (generateResponse.ok) {
            const data = await generateResponse.json();
            if (data.success && data.report) {
                currentReport = data.report;
                renderReport();
                btn.disabled = false;
                btn.textContent = originalText;
                return;
            }
        }

        throw new Error('Failed to generate report');
    } catch (error) {
        showError(`Error: ${error.message}`);
        const btn = document.getElementById('load-report-btn');
        btn.disabled = false;
        btn.textContent = '📥 Load Latest Report';
    }
}

function refreshReport() {
    if (currentReport) {
        renderReport();
    }
}

function renderReport() {
    if (!currentReport) return;

    const summary = currentReport.summary || {};

    // Update timestamp
    document.getElementById('report-info').textContent =
        `Generated: ${new Date(currentReport.timestamp).toLocaleString()}`;

    // Show summary grid
    const summaryGrid = document.getElementById('summary-grid');
    summaryGrid.style.display = 'grid';
    const totalIssues = (summary.status_mismatches || 0) +
                       (summary.profit_mismatches || 0) +
                       (summary.missing_in_pikkit || 0);
    const accuracy = summary.matched && summary.common ?
                     ((summary.matched / summary.common) * 100).toFixed(1) : 100;

    summaryGrid.innerHTML = `
        <div class="summary-card">
            <h3>Pikkit Total</h3>
            <div class="value">${(summary.pikkit_total || 0).toLocaleString()}</div>
        </div>
        <div class="summary-card">
            <h3>Supabase Total</h3>
            <div class="value">${(summary.supabase_total || 0).toLocaleString()}</div>
        </div>
        <div class="summary-card success">
            <h3>Matched</h3>
            <div class="value">${(summary.matched || 0).toLocaleString()}</div>
        </div>
        <div class="summary-card ${totalIssues > 0 ? 'warning' : ''}">
            <h3>Issues Found</h3>
            <div class="value">${totalIssues}</div>
        </div>
    `;

    // Render issues
    const issuesGrid = document.getElementById('issues-grid');
    issuesGrid.innerHTML = '';
    issuesGrid.style.display = 'grid';

    // Status mismatches
    renderStatusMismatches(currentReport.status_mismatches || []);

    // Profit mismatches
    renderProfitMismatches(currentReport.profit_mismatches || []);

    // Rogue entries
    renderRogueEntries(currentReport.missing_in_pikkit || []);

    // Enable refresh button
    document.getElementById('refresh-btn').disabled = false;
}

function renderStatusMismatches(mismatches) {
    if (mismatches.length === 0) {
        createIssuesPanel(
            '⚠️ Status Mismatches',
            [],
            'status'
        );
        return;
    }

    const pending = mismatches.filter(m => m.needs_update);
    createIssuesPanel(
        '⚠️ Status Mismatches',
        pending,
        'status',
        (m) => ({
            id: m.id,
            action: 'update_status',
            field: 'status',
            old_value: m.supabase_status,
            new_value: m.pikkit_status,
            profit: m.pikkit_profit
        }),
        (m) => `<span class="old">${m.supabase_status}</span> → <span class="new">${m.pikkit_status}</span>`
    );
}

function renderProfitMismatches(mismatches) {
    if (mismatches.length === 0) {
        createIssuesPanel(
            '💰 Profit Mismatches',
            [],
            'profit'
        );
        return;
    }

    createIssuesPanel(
        '💰 Profit Mismatches',
        mismatches.slice(0, 20),
        'profit',
        (m) => ({
            id: m.id,
            action: 'update_profit',
            field: 'profit',
            old_value: m.supabase_profit,
            new_value: m.pikkit_profit
        }),
        (m) => {
            const diff = (m.pikkit_profit || 0) - (m.supabase_profit || 0);
            return `<span class="old">$${(m.supabase_profit || 0).toFixed(2)}</span> → <span class="new">$${(m.pikkit_profit || 0).toFixed(2)}</span> <span style="opacity:0.7;">(Δ ${diff > 0 ? '+' : ''}${diff.toFixed(2)})</span>`;
        }
    );
}

function renderRogueEntries(rogueIds) {
    if (rogueIds.length === 0) {
        createIssuesPanel(
            '🚫 Rogue Entries',
            [],
            'rogue'
        );
        return;
    }

    const items = rogueIds.slice(0, 20).map(id => ({
        id,
        description: 'In Supabase but not in Pikkit'
    }));

    createIssuesPanel(
        '🚫 Rogue Entries',
        items,
        'rogue',
        (item) => ({
            id: item.id,
            action: 'delete',
            reason: 'rogue_entry'
        }),
        (item) => item.description
    );
}

function createIssuesPanel(title, items, type, getCorrection, getDisplay) {
    const issuesGrid = document.getElementById('issues-grid');
    const count = items.length;

    const panel = document.createElement('div');
    panel.className = 'issues-panel';
    panel.innerHTML = `
        <div class="issues-header">
            <h2>
                ${title}
                <span class="badge ${count > 0 ? 'warning' : 'empty'}">${count}</span>
            </h2>
        </div>
        <div class="issues-content" id="content-${type}">
            ${count === 0 ? '<div class="empty-state">✓ No issues</div>' : ''}
        </div>
    `;

    issuesGrid.appendChild(panel);

    if (items.length > 0) {
        const content = panel.querySelector(`#content-${type}`);
        content.innerHTML = items.map(item => {
            const correction = getCorrection(item);
            const display = getDisplay(item);
            return `
                <div class="issue-item" data-id="${item.id}">
                    <div class="issue-checkbox">
                        <input type="checkbox" data-type="${type}" data-correction='${JSON.stringify(correction).replace(/'/g, "&apos;")}' onchange="updateSelection(this)">
                    </div>
                    <div class="issue-details">
                        <div class="issue-id">${item.id}</div>
                        <div class="issue-text">${display}</div>
                    </div>
                </div>
            `;
        }).join('');

        if (items.length > 20) {
            const more = document.createElement('div');
            more.className = 'empty-state';
            more.textContent = `... and ${items.length - 20} more`;
            content.appendChild(more);
        }
    }
}

function updateSelection(checkbox) {
    try {
        const correction = JSON.parse(checkbox.dataset.correction);
        const item = checkbox.closest('.issue-item');

        if (checkbox.checked) {
            selectedCorrections.set(correction.id, correction);
            item.classList.add('selected');
        } else {
            selectedCorrections.delete(correction.id);
            item.classList.remove('selected');
        }

        updateCorrectionSummary();
    } catch (error) {
        console.error('Error updating selection:', error);
    }
}

function updateCorrectionSummary() {
    const summary = document.getElementById('correction-summary');
    const list = document.getElementById('correction-list');

    if (selectedCorrections.size === 0) {
        summary.classList.remove('active');
        document.getElementById('apply-btn').disabled = true;
        document.getElementById('clear-selection-btn').disabled = true;
        return;
    }

    summary.classList.add('active');
    document.getElementById('apply-btn').disabled = false;
    document.getElementById('clear-selection-btn').disabled = false;

    const byAction = {};
    selectedCorrections.forEach(c => {
        byAction[c.action] = (byAction[c.action] || 0) + 1;
    });

    list.innerHTML = Object.entries(byAction).map(([action, count]) => {
        let label = '❓ Unknown';
        if (action === 'update_status') label = '⚠️ Status Updates';
        else if (action === 'update_profit') label = '💰 Profit Updates';
        else if (action === 'delete') label = '🗑️ Deletions';

        return `
            <div class="correction-item ${action === 'delete' ? 'delete' : action === 'update_profit' ? 'profit' : ''}">
                <span>${label}</span>
                <span><strong>${count}</strong></span>
            </div>
        `;
    }).join('');
}

function clearSelection() {
    document.querySelectorAll('input[type="checkbox"]:checked').forEach(cb => {
        cb.checked = false;
        cb.dispatchEvent(new Event('change'));
    });
    selectedCorrections.clear();
    updateCorrectionSummary();
}

async function reviewAndApply() {
    if (selectedCorrections.size === 0) {
        showError('No corrections selected');
        return;
    }

    const corrections = Array.from(selectedCorrections.values());
    const summary = {};
    corrections.forEach(c => {
        summary[c.action] = (summary[c.action] || 0) + 1;
    });

    const message = `Apply ${selectedCorrections.size} corrections?\n\n${Object.entries(summary).map(([action, count]) =>
        `• ${action}: ${count}`).join('\n')}`;

    showConfirm(message, async () => {
        try {
            const response = await fetch(`${API_BASE}/apply`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ corrections })
            });

            if (!response.ok) throw new Error('Failed to apply corrections');

            const result = await response.json();

            showError(`Applied: ${result.applied} corrections${result.failed > 0 ? `, Failed: ${result.failed}` : ''}`);
            clearSelection();

            // Reload report after a short delay
            setTimeout(() => loadReportList(), 1000);
        } catch (error) {
            showError(`Error applying corrections: ${error.message}`);
        }
    });
}

function showConfirm(message, onConfirm) {
    document.getElementById('confirm-text').textContent = message;
    const okBtn = document.getElementById('confirm-ok');
    okBtn.onclick = () => {
        closeModal();
        onConfirm();
    };
    document.getElementById('confirm-modal').classList.add('active');
}

function closeModal() {
    document.getElementById('confirm-modal').classList.remove('active');
}

function confirmAction() {
    // Handled by the onclick in showConfirm
}

function showError(message) {
    console.error(message);
    alert(`${message}`);
}
