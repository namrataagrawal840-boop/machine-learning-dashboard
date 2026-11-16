"""Secondary research: market size, competitors and public review signals.

Every figure below was taken from a published source (listed with it) during desk research
on 10 November 2026. Estimates from different research firms disagree because they define
"e-commerce" differently, so the dashboard shows the source next to each number and never
mixes them in one total. Nothing here is synthetic.
"""
from __future__ import annotations

import pandas as pd

MARKET_SIZE = [
    # region, year, value_usd_bn, kind, source, url
    ("UAE", 2020, 4.8, "Actual", "EZDubai / Euromonitor e-commerce report (via Hi Dubai)", "https://focus.hidubai.com/uae-e-commerce-market-size-reaches-aed42-2-billion-in-2025/"),
    ("UAE", 2025, 11.5, "Actual", "EZDubai / Euromonitor e-commerce report (via Hi Dubai)", "https://focus.hidubai.com/uae-e-commerce-market-size-reaches-aed42-2-billion-in-2025/"),
    ("UAE", 2030, 18.3, "Forecast", "EZDubai forecast, 9.9% CAGR 2026–2030 (via GDN Online)", "https://www.gdnonline.com/Details/1407905/UAE-e-commerce-expected-to-grow-to-$183bn-by-2030"),
    ("MENASA", 2020, 63.2, "Actual", "EZDubai / Euromonitor (via Gulf Today)", "https://www.gulftoday.ae/business/2026/09/28/uae-e-commerce-market-size-reaches-dhs422-billion-in-2025"),
    ("MENASA", 2025, 141.4, "Actual", "EZDubai / Euromonitor (via Gulf Today)", "https://www.gulftoday.ae/business/2026/09/28/uae-e-commerce-market-size-reaches-dhs422-billion-in-2025"),
    ("MENASA", 2030, 247.6, "Forecast", "EZDubai / Euromonitor (via Gulf Today)", "https://www.gulftoday.ae/business/2026/09/28/uae-e-commerce-market-size-reaches-dhs422-billion-in-2025"),
    ("Saudi Arabia", 2025, 24.6, "Estimate", "Ken Research (2025 base year)", "https://www.kenresearch.com/industry-reports/saudi-arabia-e-commerce-market.md"),
    ("Saudi Arabia", 2032, 54.4, "Forecast", "Ken Research, 12% CAGR", "https://www.kenresearch.com/industry-reports/saudi-arabia-e-commerce-market.md"),
]

MARKET_FACTS = [
    ("UAE online retail passed AED 50 bn in 2024; online penetration above 15%", "Redseer", "https://redseer.com/articles/uaes-next-frontiers-of-online-retail-growth/"),
    ("Quick retail grew 5× faster than the rest of UAE online retail in 2024", "Redseer", "https://redseer.com/articles/uaes-next-frontiers-of-online-retail-growth/"),
    ("Abu Dhabi and Sharjah hold 40%+ of the UAE population but only ~12% online penetration", "Redseer", "https://redseer.com/articles/uaes-next-frontiers-of-online-retail-growth/"),
    ("Talabat and noon now share the lead in UAE quick retail", "Redseer", "https://redseer.com/articles/quick-retail-surges-ninja-leads-in-ksa-talabat-and-noon-share-the-lead-in-uae/"),
    ("Top two online retailers hold roughly 30% of GCC online retail (estimate)", "Gulf News analysis", "https://gulfnews.com/business/analysis/in-ecommerce-uaes-big-retail-groups-are-slowly-gaining-market-share-1.94364920"),
    ("UAE cart abandonment rate was about 78% in 2025", "ECDB", "https://ecdb.com/resources/sample-data/market/ae/all"),
    ("Saudi Mada-card e-commerce sales were SAR 69.3 bn in Q1 2025, up 56% year on year", "Economy Middle East", "https://economymiddleeast.com/news/saudi-arabia-e-commerce-sales-hit-18-5-billion-in-q1-2025-up-56-percent-yoy"),
]

NOON_FACTS = [
    ("Founded 2016, launched in the UAE on 1 Oct 2017; HQ in Dubai with a base in Saudi Arabia", "Gulf News / Arab News", "https://gulfnews.com/companies/noon"),
    ("Operates marketplace, food delivery and grocery in Saudi Arabia, the UAE and Egypt", "Wamda", "https://www.wamda.com/en/2025/12/noon-raises-500-million-pif-backed-investors-ahead-potential-ipo"),
    ("Raised USD 500 m in Dec 2025 from backers including PIF and Mohamed Alabbar, ahead of a possible IPO", "Wamda / Semafor", "https://www.semafor.com/article/12/12/2025/mideast-amazon-rival-noon-raises-500-million"),
    ("Acquired fashion platform Namshi from Emaar for AED 1.23 bn (USD 335 m); completed Feb 2023", "Gulf News / Images RetailME", "https://www.imagesretailme.com/noon-com-completes-acquisition-of-namshi/"),
    ("Partnered with ADNOC Distribution (Apr 2025) to put noon Minutes hubs in service stations", "ADNOC Distribution press release", "https://www.adnocdistribution.ae/media-folder/news-listing/2025/04/adnoc-distribution-and-noon-enter-strategic-partnership-to-redefine-quick-commerce-convenience"),
    ("Ecosystem includes noon Minutes, noon Food, noon Grocery and the noon One subscription", "Gulf Business", "https://gulfbusiness.com/en/2026/lists/meet-the-30-companies-shaping-the-gulfs-future/noon"),
    ("Return window: 15 days for eligible items; inspection 2–5 working days; refund can take 7–14 business days", "noon return policy (GCC page)", "https://www.noon.com/bahrain-en/return-policy/"),
    ("Trustpilot TrustScore about 1.2 / 5 from ~5,000 reviews; replies to ~97–98% of negative reviews", "Trustpilot (noon.com)", "https://jp.trustpilot.com/review/noon.com"),
]

# Competitor positioning. 'Evidence' fields are short paraphrases of the cited source.
COMPETITORS = [
    {"competitor": "Amazon.ae", "type": "Global marketplace", "fast_delivery": "Amazon Now: 15-minute essentials; 2-hour delivery on 30+ categories",
     "membership": "Prime (free Amazon Now delivery over AED 25)", "returns_signal": "Standard retail returns; Amazon Bazaar items 15-day returns",
     "review_signal": "Trustpilot reviews of Amazon.ae are mostly negative about delivery and support",
     "threat_to_noon": "High", "source": "https://www.aboutamazon.com/news/innovation/amazon-now-delivery-uae"},
    {"competitor": "Namshi (owned by noon)", "type": "Fashion marketplace", "fast_delivery": "Uses noon's logistics and fleet",
     "membership": "Shares noon ecosystem", "returns_signal": "Reviews repeatedly mention refund delays",
     "review_signal": "~450 Trustpilot reviews, mixed to negative", "threat_to_noon": "Sister brand",
     "source": "https://www.jasminedirectory.com/shopping-ecommerce/clothing/womens/detail,433105390,namshi.html"},
    {"competitor": "Talabat (Talabat Mart)", "type": "Food + quick commerce", "fast_delivery": "Dark stores, ~20–30 min in dense areas",
     "membership": "Talabat Pro (from ~AED 19/month)", "returns_signal": "Mostly groceries; few returns",
     "review_signal": "Shares the UAE quick-retail lead with noon (Redseer)", "threat_to_noon": "High (grocery, essentials)",
     "source": "https://redseer.com/articles/quick-retail-surges-ninja-leads-in-ksa-talabat-and-noon-share-the-lead-in-uae/"},
    {"competitor": "Careem Quik", "type": "Super-app quick commerce", "fast_delivery": "15-minute dark-store groceries and electronics",
     "membership": "Careem Plus", "returns_signal": "-",
     "review_signal": "'Dirhams for Delays' paid AED 1 per late minute; Dh709k paid out in 2023",
     "threat_to_noon": "Medium", "source": "https://thenationalnews.com/uae/2024/02/01/careem-paid-out-dh709000-last-year-to-reimburse-customers-for-late-food-orders-in-dubai"},
    {"competitor": "Carrefour UAE (MAF)", "type": "Omnichannel grocer", "fast_delivery": "Express and scheduled grocery slots",
     "membership": "SHARE loyalty", "returns_signal": "Store network for returns",
     "review_signal": "Big retail groups slowly gaining share by going omnichannel", "threat_to_noon": "Medium (grocery)",
     "source": "https://gulfnews.com/business/analysis/in-ecommerce-uaes-big-retail-groups-are-slowly-gaining-market-share-1.94364920"},
    {"competitor": "Temu", "type": "Cross-border value marketplace", "fast_delivery": "Launched local-to-local sellers for faster delivery",
     "membership": "-", "returns_signal": "-", "review_signal": "Top-ranked UAE shopping app on Android in early 2024 (SimilarWeb)",
     "threat_to_noon": "Medium (price)", "source": "https://menastartupdigest.com/?p=25793"},
    {"competitor": "SHEIN", "type": "Cross-border fast fashion", "fast_delivery": "UAE-based warehouse",
     "membership": "-", "returns_signal": "-", "review_signal": "Led the cross-border fashion surge in UAE and KSA",
     "threat_to_noon": "Medium (fashion)", "source": "https://redseer.com/articles/cross-border-players-grow-but-trade-shifts-local/"},
    {"competitor": "Trendyol", "type": "Turkish marketplace", "fast_delivery": "Local warehouses in UAE and KSA",
     "membership": "-", "returns_signal": "-", "review_signal": "Entered GCC with heavy promotions in 2023",
     "threat_to_noon": "Medium (fashion, home)", "source": "https://redseer.com/articles/cross-border-players-grow-but-trade-shifts-local/"},
]

# What each rival already does on the three problems (benchmarks for the solutions).
BENCHMARKS = [
    ("Delivery promise & recovery", "Careem", "Paid customers AED 1 per minute late ('Dirhams for Delays'); drivers not penalised and cannot see the ETA",
     "Shows a public late-delivery guarantee works as marketing. Our Solution 1 is cheaper: credit only persuadable customers.",
     "https://thenationalnews.com/uae/2024/02/01/careem-paid-out-dh709000-last-year-to-reimburse-customers-for-late-food-orders-in-dubai"),
    ("Delivery promise & recovery", "Amazon.ae", "15-minute Amazon Now from neighbourhood micro-fulfilment centres plus 2-hour delivery",
     "Raises customer expectations for speed, so an honest, kept promise matters even more.",
     "https://www.aboutamazon.com/news/innovation/amazon-now-delivery-uae"),
    ("Refunds", "noon today", "Inspection 2–5 working days after the item reaches the warehouse; refund 7–14 business days to card",
     "Long, invisible waits are what Solution 2's trust tiers and refund tracker remove for low-risk customers.",
     "https://www.noon.com/bahrain-en/return-policy/"),
    ("Refunds", "noon Egypt", "Refunds to noon store credit within 24 hours after inspection",
     "noon already runs a wallet-first refund in one market, so instant wallet refunds are operationally possible.",
     "https://helpegypt.noon.com/portal/en/kb/articles/what-is-the-noon-refund-process-6-3-2024"),
    ("Customer service", "noon today", "Trustpilot replies ask customers to email care@noon.com with a subject line; ~97–98% of negative reviews answered",
     "noon replies to almost every review, but with a template. Solution 3 is about one owner and a fix, not more replies.",
     "https://jp.trustpilot.com/review/noon.com"),
]


def market_tables() -> dict[str, pd.DataFrame]:
    return {
        "market_size": pd.DataFrame(MARKET_SIZE, columns=["region", "year", "value_usd_bn", "kind", "source", "url"]),
        "market_facts": pd.DataFrame(MARKET_FACTS, columns=["fact", "source", "url"]),
        "noon_facts": pd.DataFrame(NOON_FACTS, columns=["fact", "source", "url"]),
        "competitors": pd.DataFrame(COMPETITORS),
        "benchmarks": pd.DataFrame(BENCHMARKS, columns=["problem", "who", "what_they_do", "lesson_for_noon", "url"]),
    }
