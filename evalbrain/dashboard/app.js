document.addEventListener('DOMContentLoaded', () => {
    // Navigation
    const navLinks = document.querySelectorAll('.nav-links a');
    const views = document.querySelectorAll('.view');
    const pageTitle = document.getElementById('page-title');

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const viewId = link.getAttribute('data-view');
            
            // Update active link
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');

            // Update title
            pageTitle.textContent = link.textContent.trim();

            // Show view
            views.forEach(v => {
                v.classList.remove('active');
                if (v.id === `view-${viewId}`) {
                    v.classList.add('active');
                    loadViewData(viewId);
                }
            });
        });
    });

    // Initial load
    loadViewData('overview');

    // Regression Run
    document.getElementById('btn-run-regression').addEventListener('click', async () => {
        const suite = document.getElementById('reg-suite').value;
        const target = document.getElementById('reg-target').value;
        const resultsArea = document.getElementById('regression-results');
        
        if (!suite || !target) {
            resultsArea.innerHTML = '<p class="text-muted" style="color: var(--danger);">Please provide suite and target.</p>';
            return;
        }

        resultsArea.innerHTML = '<p class="text-muted">Running regression suite...</p>';
        try {
            const res = await fetch('/regression/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ suite, target })
            });
            const data = await res.json();
            if (res.ok) {
                resultsArea.innerHTML = `<p style="color: var(--success);">${data.message}</p>`;
            } else {
                resultsArea.innerHTML = `<p style="color: var(--danger);">Error: ${data.detail}</p>`;
            }
        } catch (err) {
            resultsArea.innerHTML = `<p style="color: var(--danger);">Network error.</p>`;
        }
    });
});

async function loadViewData(viewId) {
    if (viewId === 'overview') {
        try {
            const sumRes = await fetch('/metrics/summary');
            const summary = await sumRes.json();
            document.getElementById('metric-total-traces').textContent = summary.total_traces || 0;
            document.getElementById('metric-total-spans').textContent = summary.total_spans || 0;

            const costRes = await fetch('/metrics/cost');
            const costData = await costRes.json();
            const cost = costData.total_cost_usd || 0;
            document.getElementById('metric-total-cost').textContent = `$${cost.toFixed(4)}`;
        } catch (e) {
            console.error('Failed to load overview data', e);
        }
    } else if (viewId === 'traces') {
        try {
            const res = await fetch('/traces');
            const traces = await res.json();
            const tbody = document.getElementById('traces-tbody');
            tbody.innerHTML = '';
            
            if (!traces || traces.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" class="text-muted text-center">No traces found.</td></tr>';
                return;
            }

            traces.forEach(t => {
                const tr = document.createElement('tr');
                const time = new Date(t.created_at).toLocaleString();
                tr.innerHTML = `
                    <td><code>${t.trace_id.substring(0, 8)}...</code></td>
                    <td>${t.project}</td>
                    <td>${t.spans ? t.spans.length : 0}</td>
                    <td>${time}</td>
                    <td><button class="btn" style="padding: 4px 8px; font-size: 0.8rem;">View</button></td>
                `;
                tbody.appendChild(tr);
            });
        } catch (e) {
            console.error('Failed to load traces', e);
        }
    } else if (viewId === 'cost') {
        renderCostChart();
    }
}

let costChartInstance = null;
function renderCostChart() {
    const ctx = document.getElementById('costChart');
    if (!ctx) return;
    
    if (costChartInstance) {
        costChartInstance.destroy();
    }

    costChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            datasets: [{
                label: 'Cost (USD)',
                data: [0.01, 0.05, 0.12, 0.08, 0.15, 0.22], // Dummy data for visual
                backgroundColor: '#6b46c1',
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#9494a0' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#9494a0' }
                }
            },
            plugins: {
                legend: { labels: { color: '#f0f0f5' } }
            }
        }
    });
}
