"""Desk research on existing luggage-storage supply and travel nodes (researched 10 Nov 2026).

Writes data/competitor_locations.csv and data/geographic_market_audit.csv.

Evidence rules (from the project brief):
* Nothing here has been field-verified, so no record is marked "Verified".
* "Company-reported" = stated by the operator / platform on its own website.
* "Third-party reported" = travel guides, aggregators or blogs.
* Unknown prices, hours or locations are left blank — never estimated.
* Coordinates are only given for transport/tourism nodes with published coordinates (Wikipedia infobox,
  airport databases, latlong.net). Facilities *inside* a station or airport carry that node's coordinates
  (flagged in `location_basis`). Platform partners whose address is revealed only after booking get no
  coordinates.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
RESEARCHED = "2026-11-10"

# ---------------------------------------------------------------- travel nodes with published coordinates
NODES = [
    # city, microzone, node_name, node_type, lat, lon, coord_source
    ("Delhi", "New Delhi Railway Station / Paharganj", "New Delhi railway station (NDLS)", "Railway station",
     28.6417, 77.2207, "https://www.wikipedia.org/wiki/New_Delhi_railway_station"),
    ("Delhi", "New Delhi Railway Station / Paharganj", "Paharganj", "Tourism / budget-hotel district",
     28.642361, 77.217110, "https://www.latlong.net/place/paharganj-delhi-india-1767.html"),
    ("Delhi", "Old Delhi Railway Station", "Delhi Junction / Old Delhi railway station (DLI)", "Railway station",
     28.6610, 77.2277, "https://wiki.krisyotam.com/content/wikipedia_en_all_nopic_2026-03/Delhi_Junction"),
    ("Delhi", "Connaught Place", "Connaught Place (Rajiv Chowk)", "Commercial / tourism district",
     28.63278, 77.21972, "https://id.wikipedia.com/wiki/Connaught_Place,_New_Delhi"),
    ("Delhi", "IGI Airport / Aerocity", "Indira Gandhi International Airport (DEL)", "Airport",
     28.566500, 77.103088, "https://airport-data.com/world-airports/VIDP-DEL/"),
    ("Delhi", "IGI Airport / Aerocity", "Aerocity", "Airport hotel district",
     28.550421, 77.121765, "https://www.latlong.net/place/aerocity-indira-gandhi-international-airport-new-delhi-delhi-india-22460.html"),
    ("Mumbai", "CSMT / Fort", "Chhatrapati Shivaji Maharaj Terminus (CSMT)", "Railway station",
     18.9398, 72.8355, "https://wikiwand.com/en/Mumbai_CST"),
    ("Mumbai", "Churchgate / Marine Drive", "Churchgate railway station", "Railway station",
     18.9348861, 72.8271639, "https://wiki.krisyotam.com/content/wikipedia_en_all_nopic_2026-03/Churchgate_railway_station"),
    ("Mumbai", "Bandra", "Bandra railway station", "Railway station",
     19.054444, 72.840556, "https://bafybeiemxf5abjwjbikoz4mc3a3dla6ual3jsgpdr4cjr3oz3evfyavhwq.ipfs.4everland.io/wiki/Bandra_railway_station.html"),
    ("Mumbai", "Bandra", "Bandra Terminus (BDTS)", "Railway station",
     19.0626444, 72.8410889, "https://ww-article-cache-1.s3.amazonaws.com/en/Bandra_Terminus"),
    ("Mumbai", "Andheri / airport corridor", "Andheri railway station", "Railway station",
     19.119167, 72.846944, "https://en.wikipedia.org/wiki/Andheri_railway_station"),
    ("Mumbai", "Andheri / airport corridor", "Chhatrapati Shivaji Maharaj International Airport (BOM)", "Airport",
     19.0887, 72.8679, "https://airfield.directory/airfield/VABB"),
    ("Jaipur", "Jaipur Junction", "Jaipur Junction railway station (JP)", "Railway station",
     26.9208, 75.7866, "https://en.wikipedia.org/wiki/Jaipur_Junction"),
    ("Jaipur", "Sindhi Camp", "Sindhi Camp metro station (beside the inter-state bus terminal)", "Bus terminal",
     26.922563, 75.799747, "https://en.wikipedia.org/wiki/Sindhi_Camp_metro_station"),
    ("Jaipur", "Hawa Mahal / Pink City", "Hawa Mahal", "Tourist attraction",
     26.923892, 75.826607, "https://www.findlatitudeandlongitude.com/l/hawa+mahal%2C+India/545296/"),
    ("Jaipur", "Jaipur Junction", "Jaipur International Airport (JAI)", "Airport",
     26.8242, 75.8122, "https://www.planemapper.com/airports/JAI"),
    ("Goa", "Mopa & Dabolim airport corridors", "Dabolim Airport (GOI)", "Airport",
     15.38083, 73.83139, "https://www.Wikipedia.org/wiki/Goa_Airport"),
    ("Goa", "Mopa & Dabolim airport corridors", "Manohar International Airport, Mopa (GOX)", "Airport",
     15.7322, 73.8680, "https://Www.wikipedia.org/wiki/Mopa_Airport"),
    ("Goa", "Anjuna & Mapusa", "Thivim railway station (for Mapusa)", "Railway station",
     15.6307, 73.8769, "https://en.wikipedia.org/wiki/Thivim_railway_station"),
    ("Goa", "Anjuna & Mapusa", "Mapusa town", "Town / market",
     15.591976, 73.808931, "https://indiamapia.com/North_Goa/Mapusa.html"),
    ("Goa", "Calangute & Candolim", "Calangute", "Beach / tourism",
     15.54390, 73.75530, "https://latitude.to/map/in/india/cities/calangute"),
]

# ---------------------------------------------------------------- existing storage supply
# (name, city, microzone, category, service_type, node for coordinates or None, location_basis,
#  hourly_price, daily_price, price_note, hours, pickup, delivery, evidence, source_url)
C = [
    ("IGI Airport left luggage / cloakroom (T3 & T1)", "Delhi", "IGI Airport / Aerocity",
     "Official airport / railway storage", "Airport cloakroom", "Indira Gandhi International Airport (DEL)",
     "inside airport — node coordinates", None, None,
     "Reported ₹100–200 per bag for up to 2 h; ₹300–600 per bag per day by size; ID required", "24/7 (reported)",
     None, None, "Third-party reported", "https://sleepinginairports.net/guides/new-delhi-airport-guide.htm"),
    ("New Delhi railway station cloakroom (platforms 1 and 16)", "Delhi", "New Delhi Railway Station / Paharganj",
     "Official airport / railway storage", "Railway cloakroom", "New Delhi railway station (NDLS)",
     "inside station — node coordinates", None, None,
     "Railway cloakrooms reported at ₹20–50 per bag per 24 h; locked bags and receipt required", None,
     None, None, "Third-party reported", "https://vertoe.com/luggage-storage/delhi"),
    ("Old Delhi railway station cloakroom (Fatehpuri-side exit)", "Delhi", "Old Delhi Railway Station",
     "Official airport / railway storage", "Railway cloakroom", "Delhi Junction / Old Delhi railway station (DLI)",
     "inside station — node coordinates", None, None, None, None, None, None, "Third-party reported",
     "https://vertoe.com/luggage-storage/delhi"),
    ("Bounce partner – Arakashan Road, Paharganj (opposite NDLS)", "Delhi", "New Delhi Railway Station / Paharganj",
     "Aggregator network", "Partner-venue marketplace", None, "address on listing; not geocoded", None, 260.0,
     "From ₹260 per bag per 24 h (company-reported); Bounce lists 2 Delhi locations", None, None, None,
     "Company-reported", "https://bounce.com/city/delhi"),
    ("Radical Storage – New Delhi Railway Station (hotel partner)", "Delhi", "New Delhi Railway Station / Paharganj",
     "Aggregator network", "Partner-venue marketplace", None, "address revealed after booking", None, None,
     "Listed as a verified hotel partner, open 24 h; online booking only", "24 hours", None, None,
     "Company-reported",
     "https://radicalstorage.com/luggage-storage/new-delhi/new-delhi-train-station/luggage-storage-new-delhi-railway-station-4"),
    ("Radical Storage – Arya Nagar (6 min from Paharganj Market)", "Delhi", "New Delhi Railway Station / Paharganj",
     "Aggregator network", "Partner-venue marketplace", None, "address revealed after booking", None, None,
     "US$3.90 per bag per day shown on network pages", None, None, None, "Company-reported",
     "https://japancheapo.radicalstorage.com/luggage-storage/new-delhi/new-delhi-train-station/luggage-storage-arya-nagar"),
    ("LuggageHero – New Delhi Railway Station area", "Delhi", "New Delhi Railway Station / Paharganj",
     "Aggregator network", "Partner-venue marketplace", None, "location on app map; not geocoded", None, None,
     "From US$1.49 per hour or US$4.90 per day (company-reported)", None, None, None, "Company-reported",
     "https://luggagehero.com/delhi/new-delhi-railway-station/"),
    ("Radical Storage partners – Connaught Place", "Delhi", "Connaught Place", "Aggregator network",
     "Partner-venue marketplace", None, "addresses revealed after booking", None, None,
     "Network pricing reported ₹200–350 per bag per day", None, None, None, "Third-party reported",
     "https://airconnect.live/en/airports/del/luggage-storage"),
    ("Mumbai Airport left luggage – Terminal 1 (arrival hall exit ramp)", "Mumbai", "Andheri / airport corridor",
     "Official airport / railway storage", "Airport cloakroom",
     "Chhatrapati Shivaji Maharaj International Airport (BOM)", "inside airport — node coordinates", None, None,
     "₹200 (cabin) / ₹270 (check-in bag) up to 6 h; ₹350 / ₹480 up to 12 h; ₹650 / ₹800 up to 18 h; store up to 3 months",
     None, None, "Inter-terminal transfer ₹250–350 per bag", "Company-reported",
     "https://csmia.adaniairports.com/left-luggage.aspx"),
    ("Mumbai Airport left luggage – Terminal 2 (Level 1, P4)", "Mumbai", "Andheri / airport corridor",
     "Official airport / railway storage", "Airport cloakroom",
     "Chhatrapati Shivaji Maharaj International Airport (BOM)", "inside airport — node coordinates", None, None,
     "Same tariff as T1 (₹200–800 by bag size and duration)", None, None, "Inter-terminal transfer ₹250–350 per bag",
     "Company-reported", "https://csmia.adaniairports.com/left-luggage.aspx"),
    ("CSMT railway cloakroom", "Mumbai", "CSMT / Fort", "Official airport / railway storage", "Railway cloakroom",
     "Chhatrapati Shivaji Maharaj Terminus (CSMT)", "inside station — node coordinates", None, None,
     "Reported ₹20–60 per bag per 24 h; padlocked bags; receipt", None, None, None, "Third-party reported",
     "https://airconnect.live/en/luggage-storage/mumbai"),
    ("Bounce partner – near Bandra Station West", "Mumbai", "Bandra", "Aggregator network",
     "Partner-venue marketplace", None, "near Bandra station; not geocoded", None, None,
     "From US$2.95 per day (company-reported)", None, None, None, "Company-reported",
     "https://bounce.com/luggage-storage/mumbai"),
    ("Bounce partner – near Bal Gandharva Rang Mandir (Bandra)", "Mumbai", "Bandra", "Aggregator network",
     "Partner-venue marketplace", None, "Pali Mala Road; not geocoded", None, None,
     "From US$2.95 per day (company-reported)", None, None, None, "Company-reported",
     "https://bounce.com/luggage-storage/mumbai"),
    ("Bounce partner – near Mumbai Airport Terminal 2", "Mumbai", "Andheri / airport corridor", "Aggregator network",
     "Partner-venue marketplace", None, "near T2; not geocoded", None, None, "From US$2.95 per day (company-reported)",
     None, None, None, "Company-reported", "https://bounce.com/luggage-storage/mumbai"),
    ("Bounce partner – near Mumbai Airport Terminal 1", "Mumbai", "Andheri / airport corridor", "Aggregator network",
     "Partner-venue marketplace", None, "near T1; not geocoded", None, None, None, None, None, None,
     "Company-reported", "https://bounce.com/luggage-storage/mumbai"),
    ("SafeStorage – Mumbai (pickup to Panvel warehouse)", "Mumbai", "Outside research zones",
     "Other short-term storage", "Warehouse storage with pickup", None, "warehouse in Panvel, Navi Mumbai",
     None, None, "From ₹99 per month (long-term orientation)", None, "Yes", None, "Company-reported",
     "https://safestorage.in/mumbai"),
    ("Jaipur Junction railway cloakroom (platform 1)", "Jaipur", "Jaipur Junction",
     "Official airport / railway storage", "Railway cloakroom", "Jaipur Junction railway station (JP)",
     "inside station — node coordinates", None, None, None, None, None, None, "Third-party reported",
     "https://vertoe.com/luggage-storage/jaipur"),
    ("Sindhi Camp bus terminal cloakroom (platform 1)", "Jaipur", "Sindhi Camp",
     "Official airport / railway storage", "Bus-terminal cloakroom",
     "Sindhi Camp metro station (beside the inter-state bus terminal)", "beside terminal — node coordinates", None,
     None, "Reported ₹20–80 per day", None, None, None, "Third-party reported", "https://jaipurlove.com/?p=1803"),
    ("SafeStorage – Jaipur (hotel pickup; airport delivery offered)", "Jaipur", "Outside research zones",
     "Other short-term storage", "Warehouse storage with pickup", None, "facility location not published", None,
     None, "From ₹99 per bag per month", "24/7 monitored (company-reported)", "Yes", "Yes (to airport/hotel)",
     "Company-reported", "https://safestorage.in/jaipur-luggage-storage"),
    ("LuggageHero – Jaipur partner network", "Jaipur", "Not specified", "Aggregator network",
     "Partner-venue marketplace", None, "locations on app map; not geocoded", None, None,
     "From US$1.49 per hour (company-reported)", None, None, None, "Company-reported",
     "https://luggagehero.com/jaipur/"),
    ("SafeCloak smart digital lockers – Manohar International Airport (forecourt)", "Goa",
     "Mopa & Dabolim airport corridors", "Smart locker operator", "Smart lockers",
     "Manohar International Airport, Mopa (GOX)", "airport forecourt — node coordinates", None, None, None,
     None, None, None, "Third-party reported", "https://miagoaairport.com/luggage-storage"),
    ("Bounce (ex-Eelway) partner network – Calangute", "Goa", "Calangute & Candolim", "Aggregator network",
     "Partner-venue marketplace", None, "partner shops and hotels; not geocoded", None, None, None,
     "Listed 7 days, 24 h (company-reported)", None, None, "Company-reported",
     "https://www.eelway.com/en/luggage-storage/calangute"),
]

GEO_NEAREST = {  # node -> option found at / attached to that node
    "Indira Gandhi International Airport (DEL)": "IGI Airport left luggage (T3 & T1)",
    "New Delhi railway station (NDLS)": "Station cloakroom (platforms 1 & 16)",
    "Delhi Junction / Old Delhi railway station (DLI)": "Station cloakroom",
    "Chhatrapati Shivaji Maharaj International Airport (BOM)": "CSMIA left luggage (T1 & T2)",
    "Chhatrapati Shivaji Maharaj Terminus (CSMT)": "Station cloakroom",
    "Jaipur Junction railway station (JP)": "Station cloakroom (platform 1)",
    "Sindhi Camp metro station (beside the inter-state bus terminal)": "Bus-terminal cloakroom",
    "Jaipur International Airport (JAI)": "None at airport — operator FAQ states no cloak room",
    "Manohar International Airport, Mopa (GOX)": "SafeCloak smart lockers (forecourt)",
}
GEO_NEAREST_SRC = {"Jaipur International Airport (JAI)": "https://adani.com/jaipur-airport/faq"}


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    node = {n[2]: n for n in NODES}
    rows = []
    for i, c in enumerate(C, 1):
        (name, city, zone, cat, stype, nd, basis, hp, dp, note, hours, pick, deliv, ev, url) = c
        lat = lon = None
        if nd:
            lat, lon = node[nd][4], node[nd][5]
        rows.append({"competitor_id": f"C{i:03d}", "competitor_name": name, "city": city, "microzone": zone,
                     "competitor_category": cat, "service_type": stype, "latitude": lat, "longitude": lon,
                     "location_basis": basis, "hourly_price": hp, "daily_price": dp, "price_note": note,
                     "operating_hours": hours, "pickup_available": pick, "delivery_available": deliv,
                     "evidence_status": ev, "verification_date": RESEARCHED, "source_url": url,
                     "data_source": "secondary research (desk, not field-verified)"})
    comp = pd.DataFrame(rows)
    geo = pd.DataFrame([{"city": n[0], "microzone": n[1], "node_name": n[2], "node_type": n[3], "latitude": n[4],
                         "longitude": n[5], "nearest_storage_option": GEO_NEAREST.get(n[2]),
                         "nearest_storage_distance_m": None,
                         "distance_type": "facility at node (no walking distance measured)" if n[2] in GEO_NEAREST else None,
                         "evidence_status": "Third-party reported", "source_url": n[6],
                         "storage_info_source": GEO_NEAREST_SRC.get(n[2], ""), "verification_date": RESEARCHED}
                        for n in NODES])
    return comp, geo


def main() -> None:
    comp, geo = build()
    comp.to_csv(DATA / "competitor_locations.csv", index=False)
    geo.to_csv(DATA / "geographic_market_audit.csv", index=False)
    print(f"competitors: {len(comp)} ({int(comp['latitude'].notna().sum())} with node coordinates); nodes: {len(geo)}")


if __name__ == "__main__":
    main()
