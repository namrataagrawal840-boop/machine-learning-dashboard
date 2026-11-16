// Simple, plain-language submission report with graphs and dashboard screenshots.
// Usage: python -m report.build_assets && node report/build_docx_simple.js
const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, LevelFormat, BorderStyle, Footer, PageNumber, HeadingLevel } = require("docx");

const A = path.join(__dirname, "assets"), S = path.join(__dirname, "screens");
const C = JSON.parse(fs.readFileSync(path.join(A, "content.json"), "utf8"));
const OUT = path.join(__dirname, "Luggage_Storage_Project_Report.docx");
const MARGIN = 1020, W = 11906 - 2 * MARGIN;
const FONT = "Arial", BLUE = "3D5F87", TEXT = "263548", MUTED = "6B788A", SZ = 21;

const n = (x) => Number(x).toLocaleString("en-IN");
const r0 = (x) => Math.round(Number(x));
const inr = (x) => "₹" + Math.round(x).toLocaleString("en-IN");
const WTP = C["storage_wtp_inr__value"], PWTP = C["pickup_delivery_wtp_inr__value"];
const acc = (k) => C.d4.find((r) => r.accommodation_type__label.startsWith(k));
const best = C.cls.reduce((a, b) => (b.roc_auc > a.roc_auc ? b : a));
const topCl = C.cluster_overlay.reduce((a, b) => (b.unmet > a.unmet ? b : a));
const lastF = C.funnel[C.funnel.length - 1];

function runs(text, base = {}) {
  return text.split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((t) =>
    t.startsWith("**") ? new TextRun({ text: t.slice(2, -2), bold: true, size: SZ, ...base }) : new TextRun({ text: t, size: SZ, ...base }));
}
const P = (t, after = 100) => new Paragraph({ children: runs(t), spacing: { after, line: 276 } });
const H = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const B = (t) => new Paragraph({ numbering: { reference: "b", level: 0 }, children: runs(t), spacing: { after: 50, line: 264 } });
const Cap = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 }, children: [new TextRun({ text: t, italics: true, size: 16, color: MUTED })] });
function Img(file, widthPx) {
  const b = fs.readFileSync(file), w = b.readUInt32BE(16), h = b.readUInt32BE(20);
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 30 }, keepNext: true, children: [new ImageRun({ type: "png", data: b,
    transformation: { width: widthPx, height: Math.round(widthPx * h / w) }, altText: { title: path.basename(file), description: path.basename(file), name: path.basename(file) } })] });
}
const fig = (name, w) => Img(path.join(A, name + ".png"), w);
const shot = (name, w) => Img(path.join(S, name + ".png"), w);
const bd = { style: BorderStyle.SINGLE, size: 4, color: "D5DDE6" };
const borders = { top: bd, bottom: bd, left: bd, right: bd };
const cell = (t, w, o = {}) => new TableCell({ borders, width: { size: w, type: WidthType.DXA },
  shading: o.fill ? { fill: o.fill, type: ShadingType.CLEAR, color: "auto" } : undefined, margins: { top: 50, bottom: 50, left: 90, right: 90 },
  children: [new Paragraph({ spacing: { after: 0, line: 240 }, children: runs(String(t), { size: 18, bold: o.bold }) })] });
function Tbl(head, rows, frac) {
  const widths = frac.map((f) => Math.round(W * f));
  widths[widths.length - 1] = W - widths.slice(0, -1).reduce((a, b) => a + b, 0);
  return new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: widths, rows: [
    new TableRow({ tableHeader: true, children: head.map((h, i) => cell(h, widths[i], { fill: "EAF2FC", bold: true })) }),
    ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((v, i) => cell(v, widths[i])) }))] });
}

const body = [
  new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("Can a luggage-storage service work for travellers in India?")] }),
  new Paragraph({ spacing: { after: 160 }, children: [new TextRun({ text: "Individual assignment · Name: ____________________   Roll no.: __________   Date: 10 November 2026", size: 17, color: MUTED })] }),

  H("1. The idea"),
  P("Most hotels ask you to check out at around 11 in the morning, but a lot of flights and trains leave in the evening. That leaves travellers with a few hours and a suitcase they have to drag around. My idea is an app where a traveller can book a few hours of storage at a nearby verified shop, café, hotel or storage room, drop the bag, enjoy the city, and pick it up before leaving. Later the service could also collect the bag from the hotel and deliver it to the airport or railway station."),
  P("Before spending money on a business like this, I wanted to check four things: is this a real problem, who has it the most, would people trust and pay for it, and should pickup and delivery be offered from day one or later."),

  H("2. Step 1: Creating the data"),
  P("**What I did.** I did not have time or budget to survey real travellers, so I created a synthetic (made-up but realistic) dataset. I first wrote a 30-question traveller survey and a 14-question survey for shop and hotel owners. Then I used a Python program to fill in answers for 2,000 travellers, 500 each in Goa, Jaipur, Delhi and Mumbai, spread over 16 areas such as Paharganj, CSMT and Calangute, plus 240 shop and hotel owners."),
  P("**How I made it realistic.** I did not use random answers. I used common-sense rules so the answers connect the way they would in real life. For example, hotel guests can usually leave bags at the hotel but Airbnb guests usually cannot, people flying or taking trains tend to have longer waits than people driving, and people who carry their bags for longer feel more inconvenienced. I also added the kind of mistakes real surveys have, like duplicate entries, half-filled forms, spelling differences and prices typed as text, so that I could practise cleaning them."),
  P("**Real market information.** Competitors and locations should not be invented, so for these I did desk research. I found 22 existing storage options (railway and airport cloakrooms, and apps like Bounce, Radical Storage and LuggageHero) and 21 airports, stations and tourist spots with their map locations, each with a website source."),
  P("**Why.** Synthetic data let me build and test the whole process (data, cleaning, analysis, dashboard) before doing a real survey. The same steps can be re-run later on real answers."),

  H("3. Step 2: Cleaning the data"),
  P(`**What I did.** The raw file had ${n(C.n_raw)} rows. I removed 30 duplicates, 12 people who said they don't travel (they should not have continued the survey), and 25 people who quit halfway. I fixed answers written in different ways (for example "TRAIN" and "train"), turned prices like "₹200" and "200 rs" into numbers, and removed impossible values like a negative price or a score of 15 out of 10. That left **${n(C.n_clean)} clean responses**. I wrote down every change in a cleaning log so anyone can check it.`),
  P("**Why these choices.** I did not guess missing answers, because guessing would mean inventing opinions. I kept people who said they would pay ₹0, because that is a real answer. I also kept very high prices and only flagged them, because some business travellers really might pay more. People who had only planned a trip were kept but marked \"not applicable\" for questions about past trips."),
  shot("s_quality", 600), Cap("Screenshot 1. Data quality summary from my dashboard (records, duplicates, completeness, invalid answers)."),
  P("**Getting it ready for analysis.** Some questions had answers like \"1–2 hours\" or \"3–4 hours\", so I put these in order. Questions where people could tick many boxes were split into yes/no columns, one for each option. I also created a few simple labels, like \"long wait\" (3 hours or more) and \"unmet need\" (a long wait, had to carry the bags, and no easy storage option nearby)."),

  H("4. Step 3: What the data shows"),
  P("I built an interactive dashboard in Streamlit so all the charts can be filtered by city, area and type of traveller. Each chart has a short note on what it shows and what it means for the business. Here are the main findings."),
  shot("s_overview_kpis_c", 640), Cap("Screenshot 2. The main numbers on the dashboard's overview page."),
  fig("short_fig1", 650), Cap("Graph 1. A: how long people wait after checkout. B: from having the problem to having no solution. C: storage by type of accommodation."),
  B(`**The problem is real (A).** The most common wait is 3–4 hours, and ${r0(C.long_gap.pct)}% of travellers had 3 hours or more to fill.`),
  B(`**But not everyone needs a new service (B).** Many people just leave bags at their hotel. Only about ${r0(C.unmet_need.pct)}% have a real unmet need, so this is the group the business has to win.`),
  B(`**Why some people struggle more (C).** ${r0(acc("Hotels").held)}% of hotel guests could leave bags at the hotel, but only ${r0(acc("Airbnb").held)}% of Airbnb guests and ${r0(acc("Homestays").held)}% of homestay guests could. That is where the unmet need is highest, so they are the best first customers.`),
  fig("fig_methods_security", 600), Cap("Graph 2. What people do today with their bags, and what they need before they would trust the service."),
  B(`**The real competition is "doing nothing".** ${r0(C.methods[0][1])}% simply carried their bags and ${r0(C.methods[1][1])}% left them at the hotel.`),
  B(`**Trust matters most.** The top requirements are CCTV (${r0(C.security[0][1])}%), clear loss and damage cover (${r0(C.security[1][1])}%) and verified shop addresses (${r0(C.security[2][1])}%). Fear of loss or damage was also the most common reason people gave for not using the service. So these features have to be there from the start.`),
  fig("fig_d7", 470), Cap("Graph 3. How likely people are to book, compared with how inconvenient the wait was for them."),
  B(`**The more annoying the wait, the more likely they are to book.** Among people who rated the inconvenience 1 out of 5, only ${r0(C.d7_hi["1"])}% were very likely to book. Among those who rated it 5 out of 5, it was ${r0(C.d7_hi["5"])}%. So the marketing should reach people at the moment they feel the problem, for example on late-flight days.`),
  B(`**Price.** Half of the respondents would pay at least ${inr(WTP.median)} to store one suitcase for 3 hours, and most answers were between ${inr(WTP.q1)} and ${inr(WTP.q3)}. Existing apps charge around ₹250–400 a day. Many people (${r0(PWTP.zero_share)}%) would not pay anything extra for pickup and delivery.`),
  P(`**Checking which things move together.** I checked how strongly different answers are linked. The strongest link was between the time spent carrying bags and how inconvenient people found it. The number of bags made almost no difference to whether people would book, so a special multi-bag discount is not needed at launch. I used methods suited to each kind of answer. I did not just turn categories like city names into numbers, because that would give misleading results.`),

  H("5. Step 4: Using algorithms"),
  P("To go a step further, I tried four common machine-learning methods on the clean data. Here is what each one does in plain words and what it told me."),
  Tbl(["Method", "What it does (simply)", "What I found", "Useful?"], [
    ["Classification", "Learns from answers to guess who is likely to book", `It guessed correctly about ${r0(best.accuracy * 100)}% of the time, and ranks likely customers fairly well (score ${best.roc_auc.toFixed(2)} out of 1). I only used answers given before the booking question, so it could not "cheat".`, "Yes, to decide who to target"],
    ["Clustering", "Groups similar travellers together without being told the groups", `It found ${C.clusters.k} groups. One group (${n(topCl.n)} people) had long waits, a lot of inconvenience, ${r0(topCl.unmet)}% unmet need and ${r0(topCl.high_intent)}% high intent.`, "Yes, this is the launch target"],
    ["Association rules", "Finds features people tend to choose together", "People who want loss and damage cover also want live tracking and 24/7 support.", "Partly, suggests a 'safety bundle'"],
    ["Regression", "Tries to predict how much someone will pay", "Business travellers and Mumbai visitors would pay a bit more, but overall price is hard to predict.", "Limited, real price tests are better"],
  ], [0.15, 0.25, 0.42, 0.18]),
  shot("s_segments_c", 560), Cap("Screenshot 3. The four traveller groups found by clustering. Cluster 2 stands out with high intent and high unmet need."),

  H("6. How I reached my conclusion"),
  P("I did not decide based on one chart. I went back to my four questions and checked each one against the evidence:"),
  Tbl(["My question", "What the data said", "My answer"], [
    ["Is this a real problem?", `${r0(C.long_gap.pct)}% wait 3+ hours after checkout`, "Yes"],
    ["Who needs it most?", "Airbnb and homestay guests with long waits", "A clear target group exists"],
    ["Would people trust and pay?", `${r0(C.accepts_partner.pct)}% would use verified shops/hotels if safety features exist; median price ${inr(WTP.median)}`, "Yes, if trust is built in"],
    ["Pickup and delivery from day one?", `Only ${r0(C.pickup_first)}% would try it first, ${r0(PWTP.zero_share)}% would not pay extra, and few shops can handle couriers`, "Not yet, add it later"],
  ], [0.28, 0.5, 0.22]),
  P(`I also followed the "sales path" in the survey: out of ${n(C.funnel[0][1])} people, ${n(C.funnel[1][1])} had a long wait, ${n(C.funnel[2][1])} had no good option, ${n(C.funnel[4][1])} were very likely to book and ${n(lastF[1])} said they would definitely try a trial version. In a real launch I would track the same path with real website visits and bookings.`),

  H("7. Conclusion and next steps"),
  P("Based on this analysis, the idea looks worth testing, but in a careful way. I would start with **storage only**, in one area, with a small number of checked partner shops and hotels. CCTV, sealed bags, photos at drop-off and clear loss cover would be part of the service from the start. I would mainly target Airbnb and homestay guests with long waits. I would test three prices (" + inr(WTP.q1) + ", " + inr(WTP.median) + " and " + inr(WTP.q3) + " for 3 hours) and add pickup and delivery only once real bookings show it can make a profit."),
  P("**Limitations.** The traveller and shop-owner data are synthetic, so the results show how the process works and what to expect, not proof that the business will succeed. The next step is a real survey of 600–800 travellers, visits to local shops, and a small paid trial."),
  P("**What I am submitting:** this report, the dataset (raw and clean files, cleaning log, data dictionary) and the working code, including the interactive dashboard (run with \"streamlit run streamlit_app.py\").", 0),
];

const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: SZ, color: TEXT } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 34, bold: true, color: TEXT }, paragraph: { spacing: { before: 0, after: 80 }, keepNext: true } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 25, bold: true, color: BLUE }, paragraph: { spacing: { before: 220, after: 90 }, keepNext: true, outlineLevel: 0 } },
    ],
  },
  numbering: { config: [{ reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 220 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN, footer: 420 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "Page ", size: 15, color: MUTED }), new TextRun({ children: [PageNumber.CURRENT], size: 15, color: MUTED })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log("wrote", OUT); });
