// 5-page submission report. Usage: python -m report.build_assets && node report/build_docx_short.js
const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, LevelFormat, BorderStyle, Footer, PageNumber, HeadingLevel } = require("docx");

const A = path.join(__dirname, "assets");
const C = JSON.parse(fs.readFileSync(path.join(A, "content.json"), "utf8"));
const OUT = path.join(__dirname, "Luggage_Marketplace_Report.docx");
const MARGIN = 1020, W = 11906 - 2 * MARGIN; // 1.4 cm margins
const FONT = "Arial", BLUE = "3D5F87", TEXT = "263548", MUTED = "6B788A", SZ = 20;

const n = (x) => Number(x).toLocaleString("en-IN");
const f1 = (x) => Number(x).toFixed(1);
const inr = (x) => "₹" + Math.round(x).toLocaleString("en-IN");
const WTP = C["storage_wtp_inr__value"], PWTP = C["pickup_delivery_wtp_inr__value"];
const acc = (k) => C.d4.find((r) => r.accommodation_type__label.startsWith(k));
const cls = (m) => C.cls.find((r) => r.model === m);
const bestAuc = C.cls.reduce((a, b) => (b.roc_auc > a.roc_auc ? b : a));
const ridgeS = C.reg.find((r) => r.model.startsWith("Ridge") && r.target === "storage_wtp_inr");
const topCl = C.cluster_overlay.reduce((a, b) => (b.unmet > a.unmet ? b : a));
const lastF = C.funnel[C.funnel.length - 1];

function runs(text, base = {}) {
  return text.split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((t) =>
    t.startsWith("**") ? new TextRun({ text: t.slice(2, -2), bold: true, size: SZ, ...base }) : new TextRun({ text: t, size: SZ, ...base }));
}
const P = (t, after = 80) => new Paragraph({ children: runs(t), spacing: { after, line: 264 } });
const H = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const B = (t) => new Paragraph({ numbering: { reference: "b", level: 0 }, children: runs(t), spacing: { after: 50, line: 252 } });
const Cap = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: t, italics: true, size: 15, color: MUTED })] });
function Img(name, widthPx) {
  const file = path.join(A, name + ".png"), b = fs.readFileSync(file);
  const w = b.readUInt32BE(16), h = b.readUInt32BE(20);
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40, after: 20 }, children: [new ImageRun({ type: "png", data: b,
    transformation: { width: widthPx, height: Math.round(widthPx * h / w) }, altText: { title: name, description: name, name } })] });
}
const bd = { style: BorderStyle.SINGLE, size: 4, color: "D5DDE6" };
const borders = { top: bd, bottom: bd, left: bd, right: bd };
const cell = (t, w, o = {}) => new TableCell({ borders, width: { size: w, type: WidthType.DXA },
  shading: o.fill ? { fill: o.fill, type: ShadingType.CLEAR, color: "auto" } : undefined,
  margins: { top: 40, bottom: 40, left: 80, right: 80 },
  children: [new Paragraph({ spacing: { after: 0, line: 228 }, children: runs(String(t), { size: 17, bold: o.bold, color: o.color }) })] });
function Tbl(head, rows, frac, fills = {}) {
  const widths = frac.map((f) => Math.round(W * f));
  widths[widths.length - 1] = W - widths.slice(0, -1).reduce((a, b) => a + b, 0);
  return new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: widths, rows: [
    new TableRow({ tableHeader: true, children: head.map((h, i) => cell(h, widths[i], { fill: "EAF2FC", bold: true })) }),
    ...rows.map((r) => new TableRow({ children: r.map((v, i) => cell(v, widths[i], { fill: fills[i] ? fills[i](v) : undefined, bold: fills[i] ? true : false })) })),
  ] });
}
const verdictFill = (v) => (v.startsWith("Yes") ? "EAF4EF" : v.startsWith("Not yet") ? "FFF2E8" : "FFF8E8");
const sp = () => new Paragraph({ spacing: { after: 60 }, children: [] });

const body = [
  new Paragraph({ spacing: { after: 40 }, children: [new TextRun({ text: "INDIVIDUAL ASSIGNMENT · DATA ANALYTICS FOR BUSINESS STRATEGY", size: 15, bold: true, color: MUTED })] }),
  new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("Validating an India-first luggage-storage marketplace with synthetic data and analytics")] }),
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: "Student: ______________________   Roll no.: __________   Date: 10 Nov 2026   ·   Dashboard & code: github.com/namrataagrawal840-boop/machine-learning-dashboard", size: 16, color: MUTED })] }),
  P("**Business idea.** Hotel checkout is ~11:00 but flights and trains often leave in the evening, so travellers spend hours carrying bags. The platform lets them book hourly storage at verified local partners (hotels, cafés, shops, storage facilities) near their route; scheduled pickup and airport/station delivery come later. **All survey data are synthetic** (simulated to validate the idea); competitor and map data are real desk research with sources."),

  H("1. How the idea is validated"),
  P("The idea is broken into six testable claims. Each is tested on the data with a stated measure, so the conclusion follows from evidence rather than opinion:"),
  Tbl(["Claim to validate", "Test on the data", "Result", "Verdict"], [
    ["The checkout-gap problem exists", "Q11: gap ≥ 3 h", `${f1(C.long_gap.pct)}% (${n(C.long_gap.k)}/${n(C.long_gap.n)})`, "Yes — common"],
    ["An underserved segment exists", "Unmet need = long gap + carried bags + no accessible storage, excluding 'hotel storage sufficient'", `${f1(C.unmet_need.pct)}% overall; Airbnb ${f1(acc("Airbnb").unmet)}%, homestays ${f1(acc("Homestays").unmet)}% vs hotels ${f1(acc("Hotels").unmet)}%`, "Yes — narrower, rental/homestay guests"],
    ["Customers accept partner-based storage", "Q21 accepts a verified hotel/café/shop/registered business", `${f1(C.accepts_partner.pct)}%; top requirements CCTV ${C.security[0][1]}%, loss-damage terms ${C.security[1][1]}%`, "Conditional — on trust features"],
    ["Customers will pay", "Q26 stated WTP vs market prices", `Median ${inr(WTP.median)} (IQR ${inr(WTP.q1)}–${inr(WTP.q3)}); platforms charge ~₹250–400/day, rail cloakrooms ₹20–60`, "Plausible — needs price test"],
    ["Enough partners will join", "P8 willingness, P6 capacity", `${f1(C.partners.willing[0])}% willing; ${n(C.partners.capacity_willing)} bags capacity`, "Indicative (synthetic)"],
    ["Pickup/delivery should launch now", "Q24 first choice, Q27 WTP, P12 courier support", `${f1(C.pickup_first)}% choose pickup first; ${f1(PWTP.zero_share)}% pay ₹0 extra; ${f1(C.partners.courier[0])}% of partners can support couriers`, "Not yet — Phase 2"],
  ], [0.2, 0.28, 0.34, 0.18], { 3: verdictFill }),
  P(`**Sales-pipeline validation.** The same data form a survey-intent funnel: ${C.funnel.map((f) => `${f[0].replace("All respondents", "all")} ${n(f[1])}`).join(" → ")} (${f1(lastF[1] / C.funnel[0][1] * 100)}% of respondents). Each survey stage maps to a real funnel stage to be measured in a pilot — awareness → website visit → location search → availability check → booking → paid booking → handover → repeat purchase.`, 40),
  Img("fig_funnel", 430), Cap("Survey-intent funnel: an upper bound on the early-adopter pool, not a sales conversion rate."),

  H("2. Step 1 — Synthetic data generation"),
  P("A reproducible Python generator (fixed seed) answers every one of the 30 consumer questions and 14 partner questions using the exact questionnaire options. Design choices and why:"),
  B("**2,000 respondents, 500 per city** (Goa, Jaipur, Delhi, Mumbai) across 16 research microzones — balanced to compare cities (not a market-size estimate)."),
  B("**Hidden traveller types** (business, leisure, backpacker, family, pilgrim/visiting family) with city-specific mixes create realistic correlated answers; the type is not saved, so clustering must rediscover it."),
  B("**Plausible built-in rules:** hotels hold bags, rentals rarely do; flights/trains create longer gaps; inconvenience rises with time carried and bags; intent rises with pain, difficulty and trust; WTP is log-normal with ~11% ₹0 and a few legitimate extremes."),
  B(`**Realistic survey problems** were injected into the raw file (duplicates, screen-outs, abandoned surveys, typos, "₹200"-style text, negative and out-of-range values) so cleaning is meaningful. About 9% are planning-only travellers whose past-trip answers are "Not applicable".`),
  B("**240 anonymous partner responses** and **real desk research**: 22 existing storage options and 21 airports/stations with published coordinates, each with a source URL."),

  H("3. Step 2 — Data cleaning and transformation"),
  P(`Raw ${n(C.n_raw)} rows → clean ${n(C.n_clean)} rows; every decision is logged (data_cleaning_log.csv). Rules: never impute answers, keep ₹0, flag (not delete) extremes, keep "Not applicable" separate from missing.`),
  Tbl(["Problem found", "Rows", "Treatment and reason"], [
    ["Duplicate submissions", "30", "Removed — would double-count respondents"],
    ["Screened out at Q1 ('Neither')", "12", "Removed — outside target population"],
    ["Abandoned before Q23", "25", "Removed — no demand or price answers"],
    ["Inconsistent text (e.g. 'TRAIN', ' hotels ')", "160", "Standardised to questionnaire wording"],
    ["WTP typed as text / negative", "65 / 8", "Converted to ₹ / set to missing (impossible value)"],
    ["Intent outside 0–10", "10", "Set to missing"],
    ["Extreme WTP (> Q3 + 3×IQR)", "25", "Kept and flagged — legitimate high payers"],
  ], [0.36, 0.1, 0.54]),
  P("**Transformations:** ordered bands → ranks (special answers excluded); each multi-select question → one 0/1 column per option (skipped = missing, not 0); derived indicators long_gap, unmet_need, accepts_partner and High_Intent (Q23 ≥ 8) with written definitions, reused by the dashboard and models.", 40),

  H("4. Step 3 — Descriptive and diagnostic analytics (EDA)"),
  Img("short_fig1", 680), Cap("Figure 1. A: distribution of the gap; B: from problem to unmet need; C: storage outcomes by accommodation (n ≥ 30 each)."),
  B(`**A – The problem is common.** 3–4 h is the most frequent gap and ${f1(C.long_gap.pct)}% of applicable travellers have ≥ 3 h to fill. **So:** a time-based (hourly / 3–4 h) storage product fits the typical need.`),
  B(`**B – But the unmet need is smaller.** Only ${f1(C.unmet_need.pct)}% lack a usable option, because many use hotel storage. **So:** the business must target this quarter of travellers, not everyone.`),
  B(`**C – Diagnostic: accommodation explains the gap.** Hotels held bags for ${f1(acc("Hotels").held)}% of guests; Airbnb ${f1(acc("Airbnb").held)}% and homestays ${f1(acc("Homestays").held)}%, where unmet need is highest. **So:** target rental/homestay guests and recruit hosts as referral partners.`),
  Img("fig_methods_security", 600), Cap("Figure 2. Current luggage-management methods (Q15) and features required before trusting the service (Q22); multi-select."),
  B(`**Alternatives and trust.** ${C.methods[0][1]}% carried their bags and ${C.methods[1][1]}% relied on accommodation storage — "doing nothing" and free hotel storage are the real competitors. The most-required features are ${C.security.slice(0, 3).map((x) => x[0] + " (" + x[1] + "%)").join(", ")}, and loss/damage worry is the top reason not to use the service. **So:** trust features are part of the minimum product, not optional extras.`),
  Img("short_fig2", 690), Cap("Figure 3. D: High_Intent share by inconvenience; E: share willing to pay at least each price; F: Spearman correlations."),
  B(`**D – Pain drives demand.** High stated intent rises from ${C.d7_hi["1"]}% (inconvenience 1) to ${C.d7_hi["5"]}% (5); Spearman ρ = ${C.d7.rho}, p < 0.001. **So:** market at moments of pain (late flight, many bags), not to all tourists.`),
  B(`**E – Pricing.** Three-quarters state a WTP of at least ${inr(WTP.q1)}, half at least ${inr(WTP.median)} and a quarter at least ${inr(WTP.q3)} for 3 hours; ${f1(WTP.zero_share)}% say ₹0. **So:** A/B-test those three prices; the mean (${inr(WTP.mean)}) is inflated by extremes.`),
  B(`**F – Correlation (method matched to data type).** Spearman is used for ordered scales: luggage time × inconvenience ${C.spearman_top[0][2]}, inconvenience × intent ${C.d7.rho}; Pearson only for ₹ values: storage × pickup WTP r = ${C.pearson_wtp.r}; Cramér's V for categories: purpose × accommodation ${C.cramers_top[0][2]}. Bags × intent is ~0 (ρ = ${C.d6.rho}). **So:** sell "hours of freedom"; no multi-bag discount is needed; storage and pickup appeal to the same people, supporting a later bundle.`),
  P(`Other diagnostics (dashboard page 3): air and rail travellers have the longest gaps (≥ 3 h: aeroplane ${C.d5.long_by_mode.Aeroplane}%, train ${C.d5.long_by_mode.Train}%, car ${C.d5.long_by_mode["Personal vehicle / rental car"]}%; Cramér's V = ${C.d5.v}) → place partners on routes to airports and main stations. Only 7+-trip travellers state notably higher WTP (median ₹${C.d10_med["7+"]}) → candidates for a subscription. Associations are not causal.`, 40),

  H("5. How effectively can algorithms be applied?"),
  P("Each algorithm was actually run on the cleaned data (analysis/models.py) with leakage controls; results are judged on whether they change a business decision:"),
  Tbl(["Algorithm", "Why it suits this data", "Result (test set)", "Meaningful for the business?"], [
    ["Classification (LR, KNN, DT, RF, GB)", "Binary target High_Intent; predictors Q2–Q22 only (Q23–Q30 excluded to avoid leakage)", `Best ROC-AUC ${bestAuc.roc_auc.toFixed(2)} (${bestAuc.model}); LR F1 ${cls("Logistic Regression").f1.toFixed(2)}; recall low at 0.5 threshold (base rate ${f1(C.high_intent.pct)}%)`, "Yes — ranks who to target; lower the threshold if outreach is cheap"],
    ["K-means clustering", "Pre-intent behaviour scales; intent overlaid afterwards", `k = ${C.clusters.k} (silhouette 0.14, weak). ${topCl.cluster}: ${f1(topCl.unmet)}% unmet need, ${f1(topCl.high_intent)}% high intent`, "Yes — isolates the launch segment (long gaps, high pain)"],
    ["Association rules (Apriori)", "Multi-select features, venues and trust needs", `${C.rules_n} rules, max lift ${Math.max(...C.rules_top.map((r) => r.lift)).toFixed(2)} (e.g. loss-damage terms + real-time status → 24/7 support)`, "Partly — supports one 'protection bundle'; associations are weak"],
    ["Linear / Ridge / Lasso regression", "Continuous WTP targets", `Ridge R² ${ridgeS.r2.toFixed(2)} (storage), ~0.05 (pickup); top driver: business travel (+₹${C.reg_coef[0].coefficient})`, "Limited — WTP is individual; use price experiments, not models"],
  ], [0.17, 0.27, 0.33, 0.23]),
  P("**Is the dataset meaningful?** Yes, for decisions: it identifies a target segment, a first product, trust requirements and price test points, and it proves the full pipeline (data → cleaning → EDA → models → dashboard) works. It cannot prove demand: synthetic data contain the relationships the generator was given, so the same code must be re-run on real responses.", 40),

  H("6. Conclusion and strategy"),
  B("**Verdict:** the idea is **indicatively validated for a storage-first launch** — the problem is common, a clear underserved segment exists (rental/homestay guests with long gaps), and acceptance depends on trust. Pickup/delivery is **not yet justified**."),
  B(`**Phase 1:** storage only, at verified partners near stations/tourist zones, with CCTV, seals, photo handover and clear loss terms; price tests at ${inr(WTP.q1)} / ${inr(WTP.median)} / ${inr(WTP.q3)} per bag (3 h).`),
  B("**Phase 2–4:** add pickup only after paid storage bookings and real courier costs show a positive contribution margin (revenue − partner payout − transport − fees − refunds); then delivery; then multi-city."),
  B("**Before launch:** 600–800 real survey responses (150–200 per city), 40–60 partner audits per city, on-foot competitor checks, and a 5–10-partner pilot targeting the first 50–100 paid bookings."),
  P("**Limitations:** synthetic data; stated (not revealed) intent and WTP; equal city samples are a design choice; competitor data are unverified desk research. **Submission:** this report · dataset/ (CSVs, data dictionary, generation assumptions) · code/ (Streamlit dashboard with interactive, explained charts for every step; run with streamlit run streamlit_app.py).", 0),
];

const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: SZ, color: TEXT } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 34, bold: true, color: TEXT }, paragraph: { spacing: { before: 0, after: 80 }, keepNext: true } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 25, bold: true, color: BLUE }, paragraph: { spacing: { before: 200, after: 80 }, keepNext: true, outlineLevel: 0 } },
    ],
  },
  numbering: { config: [{ reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 220 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN, footer: 400 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [
      new TextRun({ text: "Luggage marketplace analytics · synthetic survey data · page ", size: 14, color: MUTED }), new TextRun({ children: [PageNumber.CURRENT], size: 14, color: MUTED })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log("wrote", OUT); });
