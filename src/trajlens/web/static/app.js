"use strict";

var SEVERITY_ORDER = ["FAIL", "WARN", "ERROR", "INFO"];
var GRADE_COLOR = { PASS: "#34d399", WARN: "#fbbf24", FAIL: "#f87171", ERROR: "#f87171" };

function gradeLabel(grade) {
  if (grade === "FAIL") return "FAIL — unsafe to train on";
  if (grade === "WARN") return "WARN — usable with caution";
  return grade;
}

function renderDetails(details) {
  if (!details || Object.keys(details).length === 0) return "";
  try {
    return JSON.stringify(details, null, 2);
  } catch (e) {
    return String(details);
  }
}

function render(data) {
  document.getElementById("loading").classList.add("hidden");
  document.getElementById("report").classList.remove("hidden");

  document.getElementById("title").textContent = "trajlens dashboard: " + data.ref;
  document.getElementById("meta").textContent =
    "Version: " + data.version +
    "  |  Episodes: " + data.num_episodes +
    "  |  Frames: " + (data.num_frames === null ? "unknown" : data.num_frames);

  var score = data.trust_score === null ? 0 : data.trust_score;
  document.getElementById("score").textContent = data.trust_score === null ? "—" : score + "/100";
  document.getElementById("score-label").textContent =
    "Trust score (formula v" + data.score_formula_version + ")";
  var fill = document.getElementById("gauge-fill");
  // Dynamic, per-report values (percentage, grade color) -- not
  // representable as a static CSS class, so this stays a CSSOM property
  // assignment from external JS, not an inline style="" HTML attribute or
  // el.setAttribute('style', ...). style-src-attr blocks the latter two;
  // it does not block .style.* writes from a script-src 'self' file.
  fill.style.width = score + "%";
  fill.style.background = GRADE_COLOR[data.grade] || "#334155";

  var gradeEl = document.getElementById("grade");
  gradeEl.textContent = gradeLabel(data.grade);
  gradeEl.className = "grade grade-" + data.grade;

  var counts = { FAIL: 0, WARN: 0, ERROR: 0, INFO: 0 };
  data.results.forEach(function (r) {
    if (counts[r.severity] !== undefined) counts[r.severity]++;
  });
  var countsBody = document.getElementById("counts-body");
  countsBody.innerHTML = "";
  SEVERITY_ORDER.forEach(function (sev) {
    var tr = document.createElement("tr");
    var tdSev = document.createElement("td");
    tdSev.className = "sev-" + sev;
    tdSev.textContent = sev;
    var tdCount = document.createElement("td");
    tdCount.textContent = String(counts[sev]);
    tr.appendChild(tdSev);
    tr.appendChild(tdCount);
    countsBody.appendChild(tr);
  });

  var resultsBody = document.getElementById("results-body");
  resultsBody.innerHTML = "";
  data.results.forEach(function (r, idx) {
    var row = document.createElement("tr");
    row.className = "result-row";

    var tdSev = document.createElement("td");
    tdSev.className = "sev-" + r.severity;
    tdSev.textContent = r.severity;

    var tdId = document.createElement("td");
    tdId.textContent = r.check_id;

    var tdMsg = document.createElement("td");
    tdMsg.textContent = r.message;

    row.appendChild(tdSev);
    row.appendChild(tdId);
    row.appendChild(tdMsg);
    resultsBody.appendChild(row);

    var detailsText = renderDetails(r.details);
    if (detailsText) {
      var detailRow = document.createElement("tr");
      var detailCell = document.createElement("td");
      detailCell.colSpan = 3;
      var detailBox = document.createElement("div");
      detailBox.className = "details";
      detailBox.id = "details-" + idx;
      detailBox.textContent = detailsText;
      detailCell.appendChild(detailBox);
      detailRow.appendChild(detailCell);
      resultsBody.appendChild(detailRow);

      row.addEventListener("click", function () {
        detailBox.classList.toggle("open");
      });
    }
  });

  renderEpisodes(data.episodes);
}

function renderEpisodes(episodes) {
  var section = document.getElementById("episodes-section");
  if (!episodes || !episodes.worst || episodes.worst.length === 0) {
    section.classList.add("hidden");
    return;
  }
  section.classList.remove("hidden");

  var body = document.getElementById("episodes-body");
  body.innerHTML = "";
  episodes.worst.forEach(function (ep) {
    var row = document.createElement("tr");

    var tdIdx = document.createElement("td");
    tdIdx.textContent = String(ep.episode_index);

    var tdContribution = document.createElement("td");
    tdContribution.textContent = String(ep.trust_contribution);

    var tdCount = document.createElement("td");
    tdCount.textContent = String(ep.finding_count);

    var tdByCheck = document.createElement("td");
    var byCheckParts = [];
    Object.keys(ep.finding_counts_by_check || {}).forEach(function (checkId) {
      byCheckParts.push(checkId + ": " + ep.finding_counts_by_check[checkId]);
    });
    tdByCheck.textContent = byCheckParts.join(", ");

    row.appendChild(tdIdx);
    row.appendChild(tdContribution);
    row.appendChild(tdCount);
    row.appendChild(tdByCheck);
    body.appendChild(row);
  });
}

function renderError(message) {
  document.getElementById("loading").classList.add("hidden");
  var errEl = document.getElementById("error");
  errEl.classList.remove("hidden");
  errEl.textContent = "Failed to load report: " + message;
}

fetch("/api/report")
  .then(function (resp) {
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    return resp.json();
  })
  .then(render)
  .catch(function (err) {
    renderError(err.message);
  });
