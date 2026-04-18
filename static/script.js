// Tab Switching Logic
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.viz-content').forEach(c => c.classList.remove('active'));
        
        e.target.classList.add('active');
        document.getElementById(e.target.dataset.target).classList.add('active');
        
        // Trigger resize event so Plotly can adjust
        window.dispatchEvent(new Event('resize'));
    });
});

// Demo Button
document.getElementById('demo-btn').addEventListener('click', () => {
    document.getElementById('sequence').value = "GGGCUAUUAGCUCAGUGGUAGAGCGCGCCCCUGAUAAGGGUGAGGUCGCUGAUUCGAAUUCAGCAUAGCCCA";
});

// Predict Logic
document.getElementById('predict-btn').addEventListener('click', async () => {
    const sequence = document.getElementById('sequence').value.trim().toUpperCase();
    if (!sequence) {
        alert('SYSTEM ERROR: Sequence missing.');
        return;
    }
    // Clean sequence
    const cleanSeq = sequence.replace(/[^ACGUN]/g, '');

    const btnText = document.querySelector('.btn-text');
    const loader = document.getElementById('btn-loader');
    
    btnText.style.display = 'none';
    loader.style.display = 'block';
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sequence: cleanSeq })
        });
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Analysis failed');
        }
        
        const data = await response.json();
        document.getElementById('viz-placeholder').style.display = 'none';
        updateDashboard(data);
        
    } catch (error) {
        alert("ERROR: " + error.message);
    } finally {
        btnText.style.display = 'block';
        loader.style.display = 'none';
    }
});

function updateDashboard(data) {
    const { contact_map, length, pairs, sequence } = data;
    
    // 1. Update Readout
    document.getElementById('res-length').textContent = `${length} nt`;
    document.getElementById('res-pairs-count').textContent = pairs.length;
    
    // Generate simple dot-bracket notation
    let dotBracket = Array(length).fill('.');
    // simple greedy approach for brackets
    let used = new Set();
    // Sort by confidence
    let sortedPairs = [...pairs].sort((a,b) => b[2] - a[2]);
    for(let p of sortedPairs) {
        let [i, j, prob] = p;
        if (!used.has(i) && !used.has(j)) {
            dotBracket[Math.min(i,j)] = '(';
            dotBracket[Math.max(i,j)] = ')';
            used.add(i); used.add(j);
        }
    }
    document.getElementById('dot-bracket-out').textContent = dotBracket.join('');

    // 2. Heatmap Plotly
    const plotData = [{
        z: contact_map,
        type: 'heatmap',
        colorscale: [
            [0, '#050914'],
            [0.5, '#1d2d50'],
            [1, '#00ffcc']
        ],
        hoverinfo: 'x+y+z'
    }];
    
    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 20, b: 40, l: 40, r: 20 },
        font: { color: '#8b9bb4', family: 'Fira Code' },
        xaxis: { title: 'Position j', gridcolor: '#1d2d50', zerolinecolor: '#1d2d50' },
        yaxis: { title: 'Position i', gridcolor: '#1d2d50', zerolinecolor: '#1d2d50', autorange: 'reversed' }
    };
    
    Plotly.newPlot('heatmap', plotData, layout, {responsive: true, displayModeBar: false});
    
    // 3. Update Table
    const tbody = document.getElementById('pairs-tbody');
    tbody.innerHTML = '';
    sortedPairs.forEach(p => {
        let tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${p[0] + 1}</td>
            <td style="color: ${getNucColor(sequence[p[0]])}">${sequence[p[0]]}</td>
            <td>${p[1] + 1}</td>
            <td style="color: ${getNucColor(sequence[p[1]])}">${sequence[p[1]]}</td>
            <td style="color: #00ffcc;">${(p[2] * 100).toFixed(2)}%</td>
        `;
        tbody.appendChild(tr);
    });

    // 4. Draw D3 2D Force Graph
    drawRNAForceGraph(sequence, sortedPairs);
}

function getNucColor(base) {
    if(base==='A') return '#ff0055';
    if(base==='U' || base==='T') return '#00d4ff';
    if(base==='G') return '#00ff66';
    if(base==='C') return '#ffaa00';
    return '#8b9bb4';
}

function drawRNAForceGraph(sequence, pairs) {
    const container = document.getElementById('rna-viz-container');
    container.innerHTML = '';
    
    const width = container.clientWidth;
    const height = container.clientHeight || 500;
    
    // Create nodes
    const nodes = sequence.split('').map((base, i) => ({ id: i, base: base }));
    
    // Create links
    const links = [];
    // Backbone
    for(let i=0; i<nodes.length-1; i++) {
        links.push({ source: i, target: i+1, type: 'backbone', distance: 15 });
    }
    // Pairs
    let usedPairs = new Set();
    pairs.forEach(p => {
        if (!usedPairs.has(p[0]) && !usedPairs.has(p[1]) && p[2] > 0.6) {
            links.push({ source: p[0], target: p[1], type: 'pair', distance: 30, prob: p[2] });
            usedPairs.add(p[0]); usedPairs.add(p[1]);
        }
    });

    const svg = d3.select("#rna-viz-container").append("svg")
        .attr("width", width)
        .attr("height", height);

    // Zoom container
    const g = svg.append("g");
    svg.call(d3.zoom().on("zoom", (e) => g.attr("transform", e.transform)));

    const simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id(d => d.id).distance(d => d.distance).strength(d => d.type === 'backbone' ? 1.5 : 0.8))
        .force("charge", d3.forceManyBody().strength(-40))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collide", d3.forceCollide().radius(8));

    // Draw lines
    const link = g.append("g")
        .selectAll("line")
        .data(links)
        .join("line")
        .attr("stroke", d => d.type === 'backbone' ? "#3a86ff" : "#00ffcc")
        .attr("stroke-width", d => d.type === 'backbone' ? 3 : (d.prob * 3))
        .attr("stroke-dasharray", d => d.type === 'backbone' ? "none" : "5,5")
        .attr("opacity", d => d.type === 'backbone' ? 0.7 : 0.9);

    // Draw nodes
    const node = g.append("g")
        .selectAll("circle")
        .data(nodes)
        .join("circle")
        .attr("r", 6)
        .attr("fill", d => getNucColor(d.base))
        .attr("stroke", "#0a1128")
        .attr("stroke-width", 1.5)
        .call(drag(simulation));
        
    // Tooltips
    node.append("title").text(d => `Position: ${d.id + 1}\nNucleotide: ${d.base}`);

    simulation.on("tick", () => {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);
    });
}

function drag(simulation) {
  function dragstarted(event) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    event.subject.fx = event.subject.x;
    event.subject.fy = event.subject.y;
  }
  function dragged(event) {
    event.subject.fx = event.x;
    event.subject.fy = event.y;
  }
  function dragended(event) {
    if (!event.active) simulation.alphaTarget(0);
    event.subject.fx = null;
    event.subject.fy = null;
  }
  return d3.drag()
      .on("start", dragstarted)
      .on("drag", dragged)
      .on("end", dragended);
}
