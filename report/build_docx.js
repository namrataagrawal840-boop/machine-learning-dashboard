// Builds the written report from report/assets/content.json and the figures in report/assets/.
// Usage (from repo root): python -m report.build_assets && node report/build_docx.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, LevelFormat, BorderStyle, PageBreak, Footer, PageNumber, ExternalHyperlink,
} = require("docx");

const A = path.join(__dirname, "assets");
const C = JSON.parse(fs.readFileSync(path.join(A, "content.json"), "utf8"));
const OUT = path.join(__dirname, "Extended_Report_19_pages.docx");

const W = 9638; // content width (A4, 2 cm margins) in DXA
const FONT = "Arial";
const BLUE = "3D5F87", TEXT = "263548", MUTED = "6B788A";

const n = (x) => Number(x).toLocaleString("en-IN");
const f1 = (x) => Number(x).toFixed(1);
const inr = (x) => "₹" + Math.round(x).toLocaleString("en-IN");
const pct = (o) => `${f1(o.pct)}% (${n(o.k)} of ${n(o.n)})`;
const pv = (p) => (p < 0.001 ? "p < 0.001" : `p = ${p.toFixed(3)}`);

// ---------------------------------------------------------------- primitives
function runs(text, base = {}) {
  // **bold** markup
  return text.split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((t) =>
    t.startsWith("**") ? new TextRun({ text: t.slice(2, -2), bold: true, ...base }) : new TextRun({ text: t, ...base }));
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text, opts.run || {}), spacing: { after: 120, line: 288 }, ...opts.para });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const B = (text, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, children: runs(text), spacing: { after: 60, line: 276 } });
const NUM = (text, ref = "numbers") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(text), spacing: { after: 60, line: 276 } });
const NUM2 = (text) => NUM(text, "numbers2");
const Caption = (t) => new Paragraph({ children: [new TextRun({ text: t, italics: true, size: 17, color: MUTED })], spacing: { after: 160 }, alignment: AlignmentType.CENTER });

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function Img(name, widthPx = 620) {
  const file = path.join(A, name + ".png");
  const s = pngSize(file);
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 80, after: 40 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: widthPx, height: Math.round(widthPx * s.h / s.w) },
      altText: { title: name, description: name, name } })],
  });
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "D5DDE6" };
const borders = { top: border, bottom: border, left: border, right: border };
function cell(text, width, opts = {}) {
  return new TableCell({
    borders, width: { size: width, type: WidthType.DXA },
    shading: opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: String(text).split("\n").map((line) => new Paragraph({ children: runs(line, { size: opts.size || 17, bold: opts.bold, color: opts.color }), spacing: { after: 20 } })),
  });
}
function Tbl(header, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, widths[i], { fill: "EAF2FC", bold: true })) }),
      ...rows.map((r) => new TableRow({ children: r.map((v, i) => cell(v, widths[i])) })),
    ],
  });
}
// Explanation box used under every chart
function Explain({ q, measure, shows, implication, caveat }) {
  const w1 = 2100, w2 = W - w1;
  const row = (k, v, fill) => new TableRow({ children: [cell(k, w1, { fill, bold: true }), cell(v, w2, { fill })] });
  return new Table({
    width: { size: W, type: WidthType.DXA }, columnWidths: [w1, w2],
    rows: [row("Business question", q, "F7F9FC"), row("What is measured", measure, "FFFFFF"),
      row("What the data shows", shows, "EAF2FC"), row("Business implication", implication, "EAF4EF"),
      row("Limitations", caveat, "FFF2E8")],
  });
}
const gap = () => new Paragraph({ children: [], spacing: { after: 120 } });
const SYN = "Synthetic data (n = " + n(C.n_clean) + "); stated answers, not observed behaviour.";

// ---------------------------------------------------------------- content
const body = [];
const add = (...x) => body.push(...x);

// Title page
add(
  new Paragraph({ spacing: { before: 2400, after: 120 }, children: [new TextRun({ text: "INDIVIDUAL ASSIGNMENT — DATA ANALYTICS FOR BUSINESS STRATEGY", size: 18, color: MUTED, bold: true })] }),
  new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("Validating an India-First Luggage-Storage Marketplace with Synthetic Data, Data Preparation and Descriptive & Diagnostic Analytics")] }),
  new Paragraph({ spacing: { before: 240, after: 120 }, children: runs("Business idea: a platform connecting travellers who have checked out but not yet departed with verified local storage partners (hotels, cafés, retail, dedicated facilities), with scheduled pickup and delivery to airports and stations as a later phase.", { size: 21, color: TEXT }) }),
  new Paragraph({ spacing: { before: 600 }, children: [new TextRun({ text: "Student: ______________________    Roll no.: ____________    Date: 10 November 2026", size: 20 })] }),
  new Paragraph({ spacing: { before: 120 }, children: [new TextRun({ text: "Deliverables: this report · dataset (data/) · working code and interactive Streamlit dashboard (code/)", size: 18, color: MUTED })] }),
  new Paragraph({ spacing: { before: 120 }, children: [new TextRun({ text: "Important: all consumer and partner survey data in this project are SYNTHETIC (simulated). Competitor and transport-node data come from sourced desk research. No real customer has been surveyed and no booking has been made.", size: 18, color: "8A4E26", bold: true })] }),
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({ children: [new TextRun({ text: "Contents", bold: true, size: 28, font: FONT })], spacing: { after: 120 } }),
  ...["1. Executive summary", "2. Business idea and research objectives", "3. Synthetic data generation (Step 1)",
    "4. Data preparation: cleaning and transformation (Step 2)", "5. Descriptive analytics (Step 3)", "6. Diagnostic analytics",
    "7. Correlation analysis", "8. Market evidence and extended analytics", "9. Business strategy and sales pipeline",
    "10. Limitations", "Appendix A — Submission contents and how to run the dashboard", "Appendix B — Key sources for market research",
  ].map((t) => new Paragraph({ children: [new TextRun({ text: t, size: 21 })], spacing: { after: 100 } })),
  new Paragraph({ children: [new PageBreak()] }),
);

// 1. Executive summary
add(H1("1. Executive summary"),
  P(`A traveller who checks out at 11:00 but departs at 17:00 must do something with their bags for six hours. This project tests, with a fully specified synthetic survey of ${n(C.n_clean)} cleaned responses (500 planned per city in Goa, Jaipur, Delhi and Mumbai), whether that gap creates an unmet need large enough to support a verified luggage-storage marketplace, and what the first product should be.`),
  B(`**The problem is common, the unmet need is narrower.** ${f1(C.long_gap.pct)}% of applicable respondents had a checkout-to-departure gap of 3 hours or more, but only ${f1(C.unmet_need.pct)}% meet the stricter "unmet need" definition (long gap, personally managed luggage, and no accessible storage option). Many travellers already solve the problem with hotel storage.`),
  B(`**Pain predicts stated demand.** Inconvenience and booking intent are moderately correlated (Spearman ρ = ${C.d7.rho}); high stated intent rises from ${C.d7_hi["1"]}% among those rating inconvenience 1 to ${C.d7_hi["5"]}% among those rating it 5.`),
  B(`**Short-term rentals and homestays are the clearest gap.** Only ${f1(C.d4.find(r=>r.accommodation_type__label.startsWith("Airbnb")).held)}% of Airbnb guests and ${f1(C.d4.find(r=>r.accommodation_type__label==="Homestays").held)}% of homestay guests had their accommodation hold luggage, versus ${f1(C.d4.find(r=>r.accommodation_type__label==="Hotels").held)}% of hotel guests.`),
  B(`**Storage first, pickup later.** ${f1(C.storage_first)}% would first try a storage-only option versus ${f1(C.pickup_first)}% a pickup/delivery option. Median stated WTP is ${inr(C["storage_wtp_inr__value"].median)} for 3-hour storage of one suitcase (IQR ${inr(C["storage_wtp_inr__value"].q1)}–${inr(C["storage_wtp_inr__value"].q3)}).`),
  B(`**Trust is a launch requirement.** CCTV (${C.security[0][1]}%), loss & damage protection terms (${C.security[1][1]}%) and verified partner addresses (${C.security[2][1]}%) are the most-cited essential features; "concern about loss or damage" is the single most common reason not to use the service.`),
  B(`**Market evidence.** Desk research found ${C.competitors.length} existing storage options in the four cities — official airport/railway cloakrooms (cheap but fixed-location and bureaucratic) and global platforms (Bounce, Radical Storage, LuggageHero) charging roughly ₹250–400 or US$3–5 per bag per day. Jaipur airport reports no cloakroom.`),
  P(`**Recommendation:** pilot a storage-only, trust-heavy service (verified partners, seals, photo handover, protection terms) in one microzone, targeting short-term-rental and homestay guests with long post-checkout gaps; test prices around the stated interquartile range; add pickup only once paid storage bookings and courier costs are measured. All findings are **indicative**, because the data are synthetic.`),
  new Paragraph({ children: [new PageBreak()] }));

// 2. Business idea
add(H1("2. Business idea and research objectives"),
  H2("2.1 The problem and proposed solution"),
  P("Accommodation checkout is typically around 11:00, but flights and trains often depart in the afternoon or evening. In that window travellers either ask the hotel to hold bags (not possible for many rentals and homestays), use a fixed cloakroom at a station or airport, or carry their luggage. The proposed platform lets travellers book storage by the hour at verified local businesses near their route, with transparent pricing, a digital record and seal for each bag, and documented handover. A later service collects bags from the accommodation and stores them or delivers them to the departure point."),
  H2("2.2 Phased product strategy"),
  Tbl(["Phase", "Scope", "Gate to advance"], [
    ["1 · Verified storage", "Online booking at partner venues; transparent prices; digital handover records", "Unmet need confirmed; 5–10 verified partners in one microzone; positive contribution per booking"],
    ["2 · Pickup + storage", "Scheduled collection from accommodation", "Paid Phase-1 bookings; pickup add-on uptake; courier cost below pickup price"],
    ["3 · Scheduled delivery", "Delivery to airport / station at a set time", "Reliable on-time pickup; liability and insurance terms"],
    ["4 · Multi-city network", "Consistent standards across cities", "Repeatable unit economics in the first city"],
  ], [2000, 3800, 3838]),
  gap(),
  H2("2.3 Research objectives"),
  NUM("**Is there an underserved customer segment?** Who experiences the checkout gap and cannot solve it with existing options?"),
  NUM("**Is pickup and delivery worth introducing?** Do travellers value it enough to pay the additional charge?"),
  NUM("**Can adequate partner density be created?** Are enough local establishments willing and able to store bags?"),
  NUM("**Will customers and local businesses accept the model?** Trust requirements on both sides."),
  P("These objectives define the end-to-end validation path from awareness to repeat purchase (Section 9). The individual-stage analysis (synthetic data, cleaning, descriptive and diagnostic analytics) answers mainly objectives 1 and 4; the extended analytics (Section 8) support the group stage."),
  H2("2.4 Rationale for each analytics technique"),
  Tbl(["Technique", "Applied to", "Why it fits this dataset and objective"], [
    ["Synthetic data generation", "30-question consumer survey; 14-question partner survey", "No real survey exists yet; a rule-based simulation with documented assumptions lets the full analytical and sales-pipeline logic be built and tested before fieldwork, and shows which data the real survey must capture."],
    ["Data cleaning & transformation", "Raw export with injected defects", "Real survey exports contain duplicates, screen-outs, typos and impossible values; ordinal and multi-select answers must be encoded before analysis without inventing data."],
    ["Descriptive analytics", "Distributions of profile, problem, alternatives, trust, demand, price", "Sizes the problem and describes who has it — the first test of whether a market exists."],
    ["Diagnostic analytics", "Cross-tabs and grouped comparisons for 10 questions", "Explains why need and intent differ between travellers (e.g. accommodation type), which drives targeting."],
    ["Correlation (Pearson / Spearman / Cramér's V)", "Continuous ₹ answers / ordered scales / nominal categories", "Each measure matches the data type; Pearson on coded categories would be meaningless."],
    ["Classification (group stage)", "High_Intent (Q23 ≥ 8) from Q2–Q22", "Identifies who to target with outreach; leakage from Q23–Q30 excluded."],
    ["Clustering (group stage)", "K-means on pre-intent behaviour", "Finds segments without imposing personas; intent overlaid afterwards."],
    ["Association rules (group stage)", "Multi-select features, venues, trust needs", "Reveals feature combinations for product bundles."],
    ["Regression (group stage)", "Storage and pickup WTP", "Identifies price drivers for pricing experiments."],
  ], [2200, 2600, 4838]),
  new Paragraph({ children: [new PageBreak()] }));

// 3. Synthetic data
const cityTxt = Object.entries(C.by_city).map(([k, v]) => `${k} ${n(v)}`).join(", ");
add(H1("3. Synthetic data generation (Step 1)"),
  P(`The dataset was generated by a reproducible Python program (analysis/synthetic.py, fixed random seed) rather than typed by hand. Each simulated respondent answers every question of the 30-question India Traveller Luggage Storage survey, with answer options copied word-for-word from the questionnaire. The design follows the business brief:`),
  B("**Sample design:** 2,000 respondents, 500 per city (Goa, Jaipur, Delhi, Mumbai). This balanced allocation is an experimental design to compare cities, not an estimate of market size."),
  B("**Geography:** four research microzones per city (e.g. New Delhi Railway Station / Paharganj, CSMT / Fort, Hawa Mahal / Pink City, Calangute & Candolim) — investigation zones, not confirmed markets."),
  B("**Hidden traveller types** (business, leisure, backpacker, family, visiting-family/pilgrim) shift every answer, with city-specific mixes (e.g. Goa leisure-heavy, Mumbai business-heavy). The type is not saved, so later clustering must rediscover structure."),
  B("**Built-in, documented relationships** (assumptions): hotels usually hold bags while rentals rarely do; flights and trains create longer gaps than private cars; time managing luggage never exceeds the gap; inconvenience rises with time carried and bag count; awareness of storage is lower for international and infrequent travellers; intent rises with inconvenience, difficulty, acceptance of partner venues and trust; WTP is log-normal with a ~11% zero share and a few legitimate extremes (₹1,500–5,000)."),
  B(`**Not Applicable is distinct from missing:** about 9% are "planning-only" travellers (${n(C.planning_only)} after cleaning) whose past-trip answers are "Not applicable", flagged by past_trip_experience.`),
  B("**Realistic defects** were injected into the raw export to make the cleaning step meaningful: 30 duplicate submissions, 12 screened-out respondents, 25 abandoned surveys, inconsistent capitalisation, WTP typed as text (“₹200”, “200 rs”), negative amounts, out-of-range intent scores and scattered non-response."),
  B("**Partner survey:** 240 anonymous establishments (15 per microzone) answering P1–P14 with linked answers (space → capacity → willingness; establishment type → opening hours). None is marked verified, because no field check has occurred."),
  P(`The resulting raw file has ${n(C.n_raw)} rows; after cleaning, ${n(C.n_clean)} respondents remain (${cityTxt}). The same structure feeds the end-to-end sales-pipeline logic: each funnel stage (problem → unmet need → acceptance → intent → pilot interest) is a measurable survey condition that will later be replaced by real website, booking and handover events (Section 9).`),
  H2("3.1 Desk research added to the synthetic data"),
  P(`Competitor and map data were **not** simulated (the brief forbids invented competitors or coordinates). Instead ${C.competitors.length} existing storage options and ${C.n_nodes} transport/tourism nodes were collected from public sources on 10 Nov 2026, each with a source URL and evidence status (company-reported or third-party reported; none field-verified). Prices and hours not published are left blank.`),
  new Paragraph({ children: [new PageBreak()] }));

// 4. Cleaning
add(H1("4. Data preparation: cleaning and transformation (Step 2)"),
  P(`Cleaning (analysis/cleaning.py) reduces the raw file from ${n(C.n_raw)} to ${n(C.n_clean)} rows and records every decision in data_cleaning_log.csv. Principles: never impute stated answers, keep legitimate extremes (flag, don't delete), keep ₹0 as a valid answer, and keep "Not applicable" separate from missing. Empty cells fell from ${n(C.raw_cells_missing)} to ${n(C.clean_cells_missing)}; the remaining gaps are genuine item non-response or skip logic.`),
  Tbl(["#", "Column", "Action", "Rows", "Reason"], C.log.map((r) => [String(r.step), r.column, r.action, n(r.rows_affected), r.reason]), [450, 1700, 3300, 700, 3488]),
  gap(),
  H2("4.1 Transformations for analysis"),
  B("**Ordinal encoding:** banded answers (gap length, luggage time, trips, group size, bags, frequency, search difficulty, detour, past spend, age) are mapped to ordered ranks; special answers (Not applicable, Unaware, Did not search, Prefer not to say) are kept as separate categories and excluded from rank statistics."),
  B("**Multi-select binarisation:** the five 'select all that apply' questions (Q15, Q19, Q21, Q22, Q28) become one 0/1 indicator per option; a respondent who skipped the question is missing, not 0."),
  B("**Derived indicators (documented working definitions):** Long gap = Q11 ≥ 3 h; Experiences problem = long gap AND managed luggage ≥ 1 h (Q12); Unmet need = problem AND (storage hard to find/unaware OR an access barrier in Q19) AND NOT 'free hotel storage was sufficient'; Accepts partner venues = Q21 includes a verified hotel, café/restaurant, retail store or registered business; High_Intent = Q23 of 8–10."),
  B("**Numeric cleaning:** WTP text converted to rupees; outliers flagged with the IQR rule but retained."),
  P("These steps are implemented once and reused by both the dashboard and the models, so every figure in this report can be traced to the same definitions."),
  new Paragraph({ children: [new PageBreak()] }));

// 5. Descriptive
const W_ = C["storage_wtp_inr__value"], PW = C["pickup_delivery_wtp_inr__value"];
add(H1("5. Descriptive analytics (Step 3)"),
  P("Each figure below is followed by a structured explanation. All percentages exclude missing and not-applicable answers; denominators are shown. The interactive dashboard reproduces every chart with filters by city, microzone and traveller profile."),
  H2("5.1 Size of the checkout gap"),
  Img("fig_gap"), Caption("Figure 1. Distribution of checkout-to-departure gap and of time spent personally managing luggage."),
  Explain({ q: "How long is the window between checkout and departure, and how long do travellers actually carry their bags?",
    measure: "Share of respondents in each Q11 and Q12 band (planning-only respondents shown as Not applicable).",
    shows: `${pct(C.long_gap)} of applicable respondents had a gap of 3 hours or more; 3–4 h is the most common band. Most respondents who had to manage luggage did so for 1–4 hours.`,
    implication: "A multi-hour window is the norm rather than the exception, so a time-based storage product has a natural use case; hourly or 3–4-hour pricing matches the typical gap.",
    caveat: SYN }),
  gap(),
  H2("5.2 From problem to unmet need"),
  Img("fig_cascade", 560), Caption("Figure 2. Successively stricter definitions of the customer problem."),
  Explain({ q: "How many travellers have a problem the market does not already solve?",
    measure: "Share meeting each working definition (Section 4.1); each bar has its own valid-answer denominator.",
    shows: `Gap ≥ 3 h: ${pct(C.long_gap)}; long gap and managed luggage ≥ 1 h: ${pct(C.has_problem)}; storage hard to find or unaware: ${pct(C.search_difficulty)}; unmet need: ${pct(C.unmet_need)}.`,
    implication: "Experiencing the gap is not the same as needing a new service. The addressable core is roughly one traveller in four — large enough to test, but the business must win this group rather than all travellers.",
    caveat: "The unmet-need rule is a provisional definition; changing it changes the size of the segment. " + SYN }),
  gap(),
  H2("5.3 Current alternatives and trust requirements"),
  Img("fig_methods_security"), Caption("Figure 3. Current luggage-management methods (Q15) and essential trust features (Q22); multi-select."),
  Explain({ q: "What do travellers use today, and what would they need before trusting a new service?",
    measure: "Percentage of answering respondents selecting each option (totals exceed 100%).",
    shows: `${C.methods[0][0]} (${C.methods[0][1]}%) and ${C.methods[1][0]} (${C.methods[1][1]}%) dominate current behaviour. The most-cited essential features are ${C.security.slice(0, 3).map((x) => `${x[0]} (${x[1]}%)`).join(", ")}.`,
    implication: "The main competitor is 'doing nothing' (carrying bags) and free hotel storage. Trust features are entry tickets, not upsells: CCTV-equipped partners, published protection terms and verified addresses must be in the Phase-1 product.",
    caveat: SYN }),
  gap(),
  H2("5.4 Inconvenience, intent and willingness to pay"),
  Img("fig_inconv_intent_wtp"), Caption("Figure 4. Inconvenience score, booking intent and storage WTP distributions."),
  Explain({ q: "How painful is the problem, how likely are people to book, and what would they pay?",
    measure: "Q17 (1–5), Q23 (0–10; gold = High_Intent 8–10) and Q26 storage WTP in rupees.",
    shows: `${f1(C.inconv_45)}% rate inconvenience 4–5. High stated intent: ${pct(C.high_intent)}. Storage WTP: median ${inr(W_.median)}, mean ${inr(W_.mean)} (right-skewed), IQR ${inr(W_.q1)}–${inr(W_.q3)}, ${f1(W_.zero_share)}% say ₹0. Extra pickup/delivery WTP: median ${inr(PW.median)}, ${f1(PW.zero_share)}% say ₹0.`,
    implication: `Price tests should use the stated interquartile range rather than the mean. A quarter of respondents would pay nothing extra for pickup, supporting a storage-first launch. First-choice service: ${C.first_service.slice(0, 2).map((x) => `${x[0].toLowerCase()} (${x[1]}%)`).join("; ")}.`,
    caveat: "Stated intent and hypothetical WTP typically overstate real purchase behaviour. " + SYN }),
  new Paragraph({ children: [new PageBreak()] }));

// 6. Diagnostic
const d4 = (k) => C.d4.find((r) => r.accommodation_type__label.startsWith(k));
add(H1("6. Diagnostic analytics"),
  P("Diagnostic analysis asks why need and intent differ between travellers. Ten questions are analysed interactively in the dashboard; the five most decision-relevant are reported here."),
  H2("6.1 Does a longer gap mean more time carrying luggage?"),
  Img("fig_d1", 540), Caption("Figure 5. Row percentages of luggage-management time within each gap band."),
  Explain({ q: "Are long gaps being absorbed by existing storage, or do travellers end up carrying bags?",
    measure: "Cross-tabulation of Q11 × Q12 (row %); Spearman rank correlation because both are ordered bands.",
    shows: `Spearman ρ = ${C.d1.rho} (${pv(C.d1.p)}, n = ${n(C.d1.n)}): a moderate positive association — the longer the gap, the longer travellers personally manage luggage.`,
    implication: "Long gaps are not fully solved by existing options; the longest-gap travellers are the natural early adopters.",
    caveat: "Association, not causation; both answers are recalled from one trip. " + SYN }),
  gap(),
  H2("6.2 Accommodation type vs availability of storage"),
  Img("fig_d4", 600), Caption("Figure 6. Storage outcomes by accommodation type (groups with n ≥ 30)."),
  Explain({ q: "Which accommodation types leave travellers without a storage option?",
    measure: "Per accommodation type: % whose accommodation held luggage (Q15), % finding storage hard or unaware (Q18), % with unmet need.",
    shows: `Hotel guests: ${f1(d4("Hotels").held)}% held luggage, ${f1(d4("Hotels").unmet)}% unmet need. Airbnb/short-term rentals: ${f1(d4("Airbnb").held)}% held, ${f1(d4("Airbnb").difficulty)}% found storage hard/unaware, ${f1(d4("Airbnb").unmet)}% unmet need. Homestays: ${f1(d4("Homestays").held)}% held, ${f1(d4("Homestays").unmet)}% unmet need.`,
    implication: "Short-term rentals and homestays are the clearest underserved segment. Rental hosts and homestay owners are promising referral partners, and marketing should target them rather than hotel guests.",
    caveat: "Groups overlap with travel purpose (Cramér's V purpose × accommodation = 0.27), so part of the difference may be driven by who books rentals. " + SYN }),
  gap(),
  H2("6.3 Inconvenience vs likelihood of using storage"),
  Img("fig_d7", 520), Caption("Figure 7. Booking intent by inconvenience score; gold line = % with High_Intent."),
  Explain({ q: "Do travellers who suffer more say they are more likely to book?",
    measure: "Box plots of Q23 by Q17; Spearman correlation; share with High_Intent per level.",
    shows: `Spearman ρ = ${C.d7.rho} (${pv(C.d7.p)}, n = ${n(C.d7.n)}). High intent rises steadily: ${Object.entries(C.d7_hi).map(([k, v]) => `${k} → ${v}%`).join(", ")}.`,
    implication: "Messaging should target moments of pain (late checkout, long layover, several bags) rather than general travel audiences.",
    caveat: "Both measures are self-reported in the same survey, which can inflate the association. " + SYN }),
  gap(),
  H2("6.4 Transport mode vs checkout gap; bags vs demand; frequency vs WTP"),
  Tbl(["Question", "Method", "Result", "Business reading"], [
    ["Transport mode vs gap", "Cramér's V (nominal × ordinal)", `V = ${C.d5.v} (${pv(C.d5.p)}); gap ≥ 3 h: aeroplane ${C.d5.long_by_mode.Aeroplane}%, train ${C.d5.long_by_mode.Train}%, personal vehicle ${C.d5.long_by_mode["Personal vehicle / rental car"]}%`, "Weak but clear: air and rail travellers drive demand, so the first partners should sit on routes to airports and main stations."],
    ["Bags vs demand", "Spearman (bags × intent)", `ρ = ${C.d6.rho} (${pv(C.d6.p)}); high intent by bags: ${Object.entries(C.d6_hi).map(([k, v]) => `${k}: ${v}%`).join(", ")}`, "No meaningful relationship — multi-bag discounts are not needed to win larger parties; per-bag pricing is defensible."],
    ["Frequency vs WTP", "Spearman (trips × WTP)", `ρ = ${C.d10.rho} (${pv(C.d10.p)}); median WTP by trips: ${Object.entries(C.d10_med).map(([k, v]) => `${k}: ₹${v}`).join(", ")}`, "Only the most frequent travellers (7+ trips) state notably higher WTP — a candidate for a premium or subscription test."],
  ], [1800, 1800, 3000, 3038]),
  gap(),
  P(`City medians for storage WTP: ${Object.entries(C.wtp_city).map(([k, v]) => `${k} ₹${v}`).join(", ")} — consistent with Mumbai's higher price level. Unmet need is similar across cities (${C.city_tbl.map((r) => `${r.city} ${r.unmet}%`).join(", ")}), so city choice should be driven by partner density and competition rather than demand alone.`),
  new Paragraph({ children: [new PageBreak()] }));

// 7. Correlation
add(H1("7. Correlation analysis"),
  P("Three measures are used, matched to the type of variable: **Pearson r** only for genuinely continuous rupee amounts, **Spearman ρ** for ordered answer bands and 0–10 / 1–5 scales (it uses ranks and does not assume equal spacing), and bias-corrected **Cramér's V** for unordered categories. Converting nominal categories to numbers and computing Pearson correlations would be meaningless, so this is never done."),
  Img("fig_spearman", 520), Caption("Figure 8. Spearman correlation matrix of ordinal variables (pairwise complete)."),
  Explain({ q: "Which ordered answers move together?",
    measure: "Spearman ρ between ranks; special answers excluded.",
    shows: `Strongest pairs: ${C.spearman_top.map((t) => `${t[0]} × ${t[1]} (${t[2]})`).join("; ")}. Pearson r between storage and pickup WTP = ${C.pearson_wtp.r} (n = ${n(C.pearson_wtp.n)}).`,
    implication: "Time spent carrying luggage is the strongest driver of inconvenience, which in turn is the strongest correlate of intent — the value proposition should be framed as 'hours of freedom'. Travellers who value storage also value pickup, supporting a later bundle.",
    caveat: "Many pairs are tested, so a few small correlations will be significant by chance; correlations are not causal. " + SYN }),
  gap(),
  Img("fig_cramers", 440), Caption("Figure 9. Cramér's V between categorical variables."),
  Explain({ q: "Which categorical answers are associated?",
    measure: "Bias-corrected Cramér's V from chi-square tests (0 = none, 1 = perfect; no direction).",
    shows: `Strongest associations: ${C.cramers_top.map((t) => `${t[0]} × ${t[1]} (${t[2]})`).join("; ")}. All are weak-to-moderate.`,
    implication: "High intent is associated with the first service chosen and accommodation type — useful targeting variables. Purpose and accommodation are linked, so their effects should not be interpreted separately.",
    caveat: SYN }),
  new Paragraph({ children: [new PageBreak()] }));

// 8. Market + extended analytics
const compRows = Object.entries(C.comp_summary).map(([city, o]) => [city, String(o["Official airport / railway storage"] || 0), String(o["Aggregator network"] || 0), String((o["Smart locker operator"] || 0) + (o["Other short-term storage"] || 0))]);
add(H1("8. Market evidence and extended analytics"),
  H2("8.1 Existing supply (desk research)"),
  Tbl(["City", "Official airport / rail / bus cloakrooms", "Platform partner listings", "Lockers / other"], compRows, [1600, 3000, 2700, 2338]),
  gap(),
  B("Official cloakrooms are cheap (railway reports ₹20–60 per bag per day) but fixed at stations/airports, require locked bags and receipts, and Mumbai airport charges ₹200–800 per bag by size and duration (operator tariff). Jaipur airport's FAQ states it has no cloak room."),
  B("Global platforms (Bounce, Radical Storage, LuggageHero) list partner venues near New Delhi station, Paharganj, Bandra and Mumbai airport at roughly ₹250–400 or US$3–5 per bag per day — a price anchor close to the survey's upper quartile."),
  B("Implication: the opportunity is not 'no storage exists' but storage that is near the traveller's route, bookable, and trusted — especially in tourist zones away from stations (e.g. Hawa Mahal / Pink City, Calangute) and for rental/homestay guests."),
  P(`Partner survey (synthetic): ${f1(C.partners.willing[0])}% of 240 establishments are willing to store non-customers' bags (dedicated facilities ${C.partners.by_type["Dedicated storage facility"]}%, hotels ${C.partners.by_type["Hotel / hostel"]}%, restaurants/cafés ${C.partners.by_type["Restaurant / café"]}%), offering ${n(C.partners.capacity_willing)} bags of simultaneous capacity; ${f1(C.partners.digital[0])}% accept digital handover and only ${f1(C.partners.courier[0])}% can support courier pickup — another reason to delay Phase 2.`),
  H2("8.2 Group-stage analytics (summary)"),
  P("These methods go beyond the individual requirement and are reported briefly; full interactive results are in dashboard pages 6–9."),
  H3("Classification — predicting High_Intent (Q23 ≥ 8) from Q2–Q22 only (test set)"),
  Tbl(["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"], C.cls.map((r) => [r.model, r.accuracy.toFixed(3), r.precision.toFixed(3), r.recall.toFixed(3), r.f1.toFixed(3), r.roc_auc.toFixed(3), r.pr_auc.toFixed(3)]), [2400, 1200, 1200, 1200, 1200, 1219, 1219]),
  P(`Ensemble models rank respondents best (ROC-AUC ≈ 0.79), but recall at the default threshold is low because only ${f1(C.high_intent.pct)}% are high-intent. A false positive wastes outreach spend; a false negative misses an interested traveller — the dashboard's threshold slider shows this trade-off.`),
  H3("Clustering — K-means on pre-intent behaviour"),
  P(`Silhouette scores (${C.clusters.silhouette}) select k = ${C.clusters.k}; separation is weak, as expected from overlapping traveller types. The decisive segment is visible after overlaying outcomes:`),
  Tbl(["Cluster", "n", "High intent %", "Unmet need %", "Mean gap rank", "Mean inconvenience", "Median WTP"], C.cluster_overlay.map((r) => [r.cluster, String(r.n), f1(r.high_intent), f1(r.unmet), f1(r.gap), f1(r.inconv), "₹" + r.wtp]), [1400, 900, 1400, 1400, 1500, 1600, 1438]),
  P(`One cluster combines long gaps, high inconvenience, ${f1(Math.max(...C.cluster_overlay.map((r) => r.unmet)))}% unmet need and ${f1(Math.max(...C.cluster_overlay.map((r) => r.high_intent)))}% high intent — the launch target.`),
  H3("Association rules and regression"),
  P(`Apriori found ${C.rules_n} rules with lift > 1, all modest (maximum lift ${Math.max(...C.rules_top.map((r) => r.lift)).toFixed(2)}): e.g. respondents requiring loss & damage terms and real-time status also tend to require 24/7 support — supporting a single 'protection bundle'. Ridge regression explains ${f1(C.reg.find((r) => r.model.startsWith("Ridge") && r.target === "storage_wtp_inr").r2 * 100)}% of storage-WTP variance; the largest positive coefficient is business travel and the largest negatives are hostel stays and relying on friends — WTP is mostly individual, so pricing should be tested experimentally.`),
  new Paragraph({ children: [new PageBreak()] }));

// 9. Strategy
add(H1("9. Business strategy and sales pipeline"),
  H2("9.1 Survey-intent funnel"),
  Img("fig_funnel", 540), Caption("Figure 10. Respondents meeting each successive condition (survey intent, not sales)."),
  P(`Of ${n(C.funnel[0][1])} respondents, ${n(C.funnel[C.funnel.length - 1][1])} (${f1(C.funnel[C.funnel.length - 1][1] / C.funnel[0][1] * 100)}%) meet every condition up to 'would definitely join a pilot'. This is an upper bound on an early-adopter pool, not a conversion rate.`),
  H2("9.2 End-to-end validation of the sales pipeline"),
  Tbl(["Funnel stage", "Metric to instrument", "Current evidence"], [
    ["Awareness", "Reach / impressions in target zones", "Not measured"],
    ["Website visit", "Unique visitors", "Not measured"],
    ["Location search", "Searches with a location", "Not measured"],
    ["Availability check", "Searches returning a slot", "Partner capacity (synthetic) only"],
    ["Booking initiation", "Checkouts started", "Stated intent only"],
    ["Paid booking", "Paid bookings; price-test conversion", "Not measured"],
    ["Completed handover", "Bags dropped & collected without incident", "Not measured"],
    ["Repeat purchase", "Customers with ≥ 2 bookings", "Not measured"],
  ], [2400, 3800, 3438]),
  gap(),
  P("**Contribution margin per booking** = booking revenue − partner payment − transport cost − payment fees − refunds/claims − other variable costs. High stated WTP does not prove profitability; the dashboard includes a scenario calculator where every input is labelled as an assumption until pilot data exist."),
  H2("9.3 Recommendations"),
  NUM2("**Target first:** travellers in short-term rentals and homestays with gaps ≥ 3 h — the cluster with the highest unmet need and intent."),
  NUM2("**First service:** storage only (Phase 1). Pickup is preferred by a minority, a quarter would pay nothing extra for it, and few partners can support couriers."),
  NUM2("**Where:** use the dashboard's opportunity score to choose one microzone for field verification; favour tourist zones away from existing station cloakrooms and with willing hotel/dedicated-storage partners."),
  NUM2(`**Trust package:** verified partner addresses, CCTV, tamper-evident seals, photo handover and clear loss & damage terms — the top requirements in Q22 and the top barrier in Q29.`),
  NUM2(`**Pricing experiments:** A/B test ${inr(W_.q1)}, ${inr(W_.median)} and ${inr(W_.q3)} per bag (3 h) and compare 'hourly per bag' with 'fixed 3–4 h' structures; measure paid conversion.`),
  NUM2("**Before launch:** collect 600–800 real survey responses (150–200 per city), 40–60 partner audits per city, verify competitor locations on foot, and run a 5–10-partner storage pilot targeting the first 50–100 paid bookings."),
  gap());

// 10. Limitations + appendix
add(H1("10. Limitations"),
  B("**Synthetic data:** results reflect the assumptions coded into the simulation (documented in docs/SYNTHETIC_DATA.md), not real traveller behaviour. They validate the analysis pipeline and illustrate likely patterns; they do not validate the business."),
  B("**Stated vs revealed preference:** intent and WTP are hypothetical and usually overstate real purchases."),
  B("**Equal city samples:** 500 per city is a design choice; city totals are not market sizes."),
  B("**Desk research:** competitor data are company- or third-party-reported and unverified; platform partner addresses are hidden until booking, so they are not mapped."),
  B("**Associations are not causal**, and many comparisons increase the chance of spurious significance."),
  H1("Appendix A — Submission contents and how to run the dashboard"),
  Tbl(["Folder / file", "Contents"], [
    ["report/", "This report (Word and PDF)"],
    ["dataset/", "consumer_survey_raw_synthetic.csv, consumer_survey_clean.csv, data_cleaning_log.csv, partner_survey.csv, competitor_locations.csv, geographic_market_audit.csv, model outputs, data dictionary"],
    ["code/", "Complete GitHub repository: Streamlit dashboard (streamlit_app.py, dashboard/), data generation, cleaning and model pipeline (analysis/), report builder (report/), tests"],
  ], [2400, 7238]),
  gap(),
  P("**Run locally:** pip install -r requirements.txt, then streamlit run streamlit_app.py (opens http://localhost:8501). **Regenerate data and models:** pip install -r requirements-analysis.txt, then python -m analysis.run_pipeline. **Live code:** https://github.com/namrataagrawal840-boop/machine-learning-dashboard"),
  H1("Appendix B — Key sources for market research"),
  ...C.competitors.filter((r, i, arr) => arr.findIndex((x) => x.source_url === r.source_url) === i).slice(0, 14).map((r) =>
    new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 40 }, children: [
      new TextRun({ text: `${r.competitor_name} (${r.evidence_status}): `, size: 17 }),
      new ExternalHyperlink({ link: r.source_url, children: [new TextRun({ text: r.source_url, style: "Hyperlink", size: 17 })] })] })),
  P("Coordinates of transport nodes are listed with their source URLs in dataset/geographic_market_audit.csv."),
);

// ---------------------------------------------------------------- document
const doc = new Document({
  creator: "Luggage Marketplace Analytics", title: "Luggage Marketplace — Data Analytics Report",
  styles: {
    default: { document: { run: { font: FONT, size: 20, color: TEXT } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 44, bold: true, color: TEXT }, paragraph: { spacing: { before: 0, after: 200 }, keepNext: true } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 30, bold: true, color: BLUE }, paragraph: { spacing: { before: 240, after: 140 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 24, bold: true, color: TEXT }, paragraph: { spacing: { before: 200, after: 100 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 21, bold: true, color: TEXT }, paragraph: { spacing: { before: 160, after: 80 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: "numbers2", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [
      new TextRun({ text: "Luggage marketplace analytics · synthetic data · page ", size: 16, color: MUTED }),
      new TextRun({ children: [PageNumber.CURRENT], size: 16, color: MUTED })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT); });
