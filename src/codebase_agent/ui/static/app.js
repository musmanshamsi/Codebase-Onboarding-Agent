// Codebase Onboarding Agent - Web UI Interactive Logic

document.addEventListener("DOMContentLoaded", () => {
    fetchStatus();
    fetchModels();
});

function switchTab(tabName) {
    document.querySelectorAll(".nav-item").forEach(btn => btn.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(tab => tab.classList.remove("active"));

    document.getElementById(`nav-${tabName}`).classList.add("active");
    document.getElementById(`tab-${tabName}`).classList.add("active");

    if (tabName === "graph") {
        loadGraph();
    }
}

async function fetchStatus() {
    try {
        const res = await fetch("/api/status");
        if (res.ok) {
            const data = await res.json();
            document.getElementById("header-repo-path").innerText = data.repo_path || "...";
            document.getElementById("stat-files").innerText = data.indexed_files_count || 0;
            document.getElementById("stat-symbols").innerText = data.total_symbols || 0;
            document.getElementById("stat-nodes").innerText = data.graph_nodes || 0;
            document.getElementById("stat-edges").innerText = data.graph_edges || 0;
            document.getElementById("path-index").innerText = data.index_directory || "...";
            
            if (data.last_run) {
                document.getElementById("stat-last-run").innerText = `${data.last_run.status} at ${data.last_run.completed_at || 'N/A'}`;
            }
        }
    } catch (err) {
        console.error("Error fetching status:", err);
    }
}

async function fetchModels() {
    try {
        const res = await fetch("/api/models");
        if (res.ok) {
            const data = await res.json();
            const dot = document.getElementById("ollama-dot");
            const text = document.getElementById("ollama-status-text");

            if (data.ollama_online) {
                dot.className = "dot online";
                text.innerText = "Ollama: Online";
            } else {
                dot.className = "dot offline";
                text.innerText = "Ollama: Offline";
            }

            const select = document.getElementById("model-select");
            select.innerHTML = "";
            (data.models || ["qwen2.5-coder:7b", "qwen2.5-coder:1.5b"]).forEach(m => {
                const opt = document.createElement("option");
                opt.value = m;
                opt.innerText = m;
                select.appendChild(opt);
            });
        }
    } catch (err) {
        console.error("Error fetching models:", err);
    }
}

async function handleQuerySubmit(event) {
    event.preventDefault();

    const question = document.getElementById("query-input").value.trim();
    if (!question) return;

    const model_name = document.getElementById("model-select").value;
    const top_k = parseInt(document.getElementById("topk-input").value, 10);
    const similarity_threshold = parseFloat(document.getElementById("thresh-input").value);

    const submitBtn = document.getElementById("btn-submit-query");
    const outputContainer = document.getElementById("output-container");
    const answerText = document.getElementById("answer-text");
    const citationsGrid = document.getElementById("citations-grid");

    submitBtn.disabled = true;
    submitBtn.innerText = "Synthesizing...";
    outputContainer.classList.add("hidden");

    try {
        const res = await fetch("/api/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, model_name, top_k, similarity_threshold })
        });

        if (res.ok) {
            const data = await res.json();
            document.getElementById("output-model-badge").innerText = data.model_name || model_name;
            answerText.innerText = data.answer || "No answer generated.";

            citationsGrid.innerHTML = "";
            if (data.citations && data.citations.length > 0) {
                data.citations.forEach((cit, idx) => {
                    const card = document.createElement("div");
                    card.className = "citation-card";
                    card.innerHTML = `
                        <div class="path">[${idx + 1}] ${cit.file_path}:${cit.start_line}-${cit.end_line}</div>
                        <div class="symbol">Symbol: ${cit.symbol_name || 'N/A'} (Node: ${cit.graph_node_id || ''})</div>
                    `;
                    citationsGrid.appendChild(card);
                });
            } else {
                citationsGrid.innerHTML = `<div class="citation-card">No specific evidence citations returned.</div>`;
            }

            outputContainer.classList.remove("hidden");
        } else {
            alert("Error querying codebase agent.");
        }
    } catch (err) {
        console.error("Query error:", err);
        alert("Failed to connect to query server.");
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerText = "Ask Agent";
    }
}

async function triggerReindex() {
    if (!confirm("Run incremental re-index now?")) return;

    try {
        const res = await fetch("/api/index", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ full: false })
        });

        if (res.ok) {
            const data = await res.json();
            alert(`Re-indexing complete!\nProcessed files: ${data.summary.processed_files}\nNew chunks: ${data.summary.new_chunks}`);
            fetchStatus();
        } else {
            alert("Re-indexing failed.");
        }
    } catch (err) {
        console.error("Reindex error:", err);
    }
}

async function loadGraph() {
    try {
        const res = await fetch("/api/graph");
        if (res.ok) {
            const data = await res.json();
            renderSvgGraph(data);
        }
    } catch (err) {
        console.error("Graph error:", err);
    }
}

function renderSvgGraph(data) {
    const svg = document.getElementById("graph-svg");
    svg.innerHTML = "";

    if (!data.nodes || data.nodes.length === 0) {
        svg.innerHTML = `<text x="50%" y="50%" fill="#94a3b8" text-anchor="middle">No graph nodes found. Run 'index' first.</text>`;
        return;
    }

    const width = svg.clientWidth || 800;
    const height = 600;

    // Simple grid layout for graph nodes
    const cols = Math.ceil(Math.sqrt(data.nodes.length));
    const stepX = width / (cols + 1);
    const stepY = height / (cols + 1);

    const posMap = {};

    data.nodes.forEach((node, idx) => {
        const col = idx % cols;
        const row = Math.floor(idx / cols);
        const cx = stepX * (col + 1);
        const cy = stepY * (row + 1);
        posMap[node.id] = { cx, cy };

        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", cx);
        circle.setAttribute("cy", cy);
        circle.setAttribute("r", 16);
        circle.setAttribute("fill", node.type === "function" ? "#818cf8" : node.type === "class" ? "#f472b6" : "#2dd4bf");
        circle.style.cursor = "pointer";

        const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
        title.textContent = `ID: ${node.id}\nType: ${node.type}\nFile: ${node.file || 'N/A'}`;
        circle.appendChild(title);

        circle.addEventListener("click", () => {
            alert(`Symbol Node Detail:\n• ID: ${node.id}\n• Type: ${node.type}\n• File: ${node.file || 'N/A'}`);
        });

        svg.appendChild(circle);

        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", cx);
        text.setAttribute("y", cy + 28);
        text.setAttribute("fill", "#e2e8f0");
        text.setAttribute("font-size", "11");
        text.setAttribute("text-anchor", "middle");
        text.textContent = node.label;
        svg.appendChild(text);
    });

    data.links.forEach(link => {
        const src = posMap[link.source];
        const tgt = posMap[link.target];
        if (src && tgt) {
            const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
            line.setAttribute("x1", src.cx);
            line.setAttribute("y1", src.cy);
            line.setAttribute("x2", tgt.cx);
            line.setAttribute("y2", tgt.cy);
            line.setAttribute("stroke", "#475569");
            line.setAttribute("stroke-width", "1.5");
            line.setAttribute("stroke-dasharray", "4");
            svg.insertBefore(line, svg.firstChild);
        }
    });
}
