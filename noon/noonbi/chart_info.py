"""One explanation per chart, shown behind the ⓘ icon above every chart and reused in the written guide.

Each entry: title, what (what it shows), read (how to read it), made (data + method), why (what it means for noon),
careful (limits). Keys are the chart keys passed to charts.show(); live/simulator keys are matched by prefix.
"""
from __future__ import annotations

INFO: dict[str, dict[str, str]] = {
    # ------------------------------------------------------------------ 1 · Overview
    "ov_chain": dict(
        title="How the three problems feed each other (flow chart)",
        what="Customers flowing from what went wrong (late delivery, failed delivery, slow refund, rejected refund, other issues) "
             "to whether they raised a support ticket, and then to whether they kept buying or stopped within 90 days.",
        read="Band width = number of customers or tickets. Follow a band from left to right. The thick 'No ticket' band shows "
             "most late orders do not become tickets; the red band on the right shows tickets that ended with a lost customer.",
        made="Counts from the synthetic orders and tickets tables. A ticket's trigger (late order, failed delivery, slow or "
             "rejected refund) is recorded when it is created; churn = no purchase in the 90 days after the ticket.",
        why="Shows the three problems are one chain: delivery and refund failures create tickets, and tickets create churn. "
            "That is why the three solutions share one data system.",
        careful="Synthetic data shaped by public reviews. Real band sizes will differ; the method stays the same."),
    # ------------------------------------------------------------------ 2 · Market
    "mk_size": dict(
        title="Market size by region (USD bn)",
        what="Total e-commerce sales value for the UAE, MENASA or Saudi Arabia, by year, from published research.",
        read="Pick a region with the buttons. Dark bars = reported actuals, light bars = forecasts, teal = estimates. Hover for the source.",
        made="Secondary research (EZDubai/Euromonitor, Ken Research), each figure linked on the page and in Sources.",
        why="The UAE market more than doubled from 2020 to 2025 and still grows about 10% a year. As growth slows, keeping "
            "customers after a bad experience matters more than winning new ones.",
        careful="Research firms define e-commerce differently, so figures from different firms are never added together."),
    # ------------------------------------------------------------------ 3 · Data
    "dq_log_chart": dict(
        title="Cleaning log: rows affected by each fix",
        what="Every data problem found in the raw exports and how many rows it touched, per dataset.",
        read="Longer bar = more rows fixed. Use the dataset pills to filter. The table below gives the action and the reason.",
        made="The raw files were generated with deliberate errors (duplicates, two date formats, 'DXB' for Dubai, prices typed "
             "as text, negative times, CSAT of 7). The cleaning code fixes each one and writes a log row.",
        why="Shows the analysis is built on clean data, and that no missing value was guessed.",
        careful="Missing zones stay 'Unknown' and rejected refunds have no refund time, on purpose."),
    # ------------------------------------------------------------------ 4 · Customers
    "cu_seg_scatter": dict(
        title="Customer segments (K-Means on RFM + returns)",
        what="A sample of customers placed by days since last order (x) and 12-month spend (y, log scale), coloured by segment.",
        read="Each dot is a customer. Left = bought recently; higher = spends more. Colours are the five groups the algorithm found.",
        made="K-Means clustering on five standardised measures: recency, log order count, log spend, return rate and rejected "
             "refunds. Five groups chosen with the silhouette score. Names are our reading of each group's averages.",
        why="Different groups need different policies: instant refunds for loyal customers, checks for high-risk returners, "
            "win-back timing for lapsing customers.",
        careful="Only a sample of 2,500 customers is drawn to keep the chart fast."),
    "cu_seg_k": dict(
        title="Choosing the number of customer groups (silhouette)",
        what="The silhouette score for 3 to 7 groups. The dark bar is the number we kept.",
        read="Higher = groups are more clearly separated. We look for a high score that still gives groups with a clear business meaning.",
        made="K-Means run for each k on the same standardised data; silhouette computed on up to 4,000 customers.",
        why="Shows the number of segments was chosen with evidence, not picked by hand.",
        careful="Silhouette is one guide; business usefulness also matters."),
    "cu_clv_scatter": dict(
        title="Predicted vs actual next-12-month profit (CLV)",
        what="For customers the model did not train on, the profit they actually brought next year (x) against what the model predicted (y).",
        read="Dots on the dotted line are perfect predictions. Dots above were over-predicted, below under-predicted.",
        made="XGBoost regression on orders, spend, recency, tenure, late orders, failed deliveries, returns, rejected refunds, "
             "tickets and repeat contacts. 70/30 split. R² about 0.75.",
        why="CLV decides how much noon should spend to keep someone: who gets a credit, an instant refund or a senior agent.",
        careful="Profit assumes an 11% margin. The biggest spenders are the hardest to predict exactly."),
    "cu_clv_tiers": dict(
        title="CLV tiers",
        what="Average predicted next-year profit for the Low (bottom 50%), Mid (next 30%) and High (top 20%) customer tiers.",
        read="Taller bar = more valuable tier.",
        made="Customers ranked by predicted CLV and cut at the 50th and 80th percentiles.",
        why="The High tier is where retention spend pays back most.",
        careful="Tiers come from predictions, not actual future profit."),
    "cu_churn_roc": dict(
        title="Customer churn model: ROC curve",
        what="How well each model ranks customers from most to least likely to stop buying in the next 90 days.",
        read="A curve hugging the top-left corner is better. The dotted diagonal is random guessing. AUC in the legend: 0.5 = guessing, 1 = perfect.",
        made="Logistic regression and XGBoost on the same customer features as the CLV model; 70/30 split stratified on churn.",
        why="Customers with a bad experience and a high churn score are the ones recovery actions should reach first.",
        careful="Synthetic data; on real data the curve will be lower."),
    "cu_churn_imp": dict(
        title="What drives customer churn",
        what="The features the XGBoost churn model relies on most, as a share of total importance.",
        read="Longer bar = the model uses it more. It shows influence, not direction (it does not say 'more of X = more churn').",
        made="XGBoost feature importance (gain), top 12 features.",
        why="Recency and engagement matter most, but late orders, rejected refunds and repeat contacts also appear, linking churn to the three problems.",
        careful="Importance is not cause; correlated features share credit."),
    "cu_npd": dict(
        title="Next-purchase window (confusion matrix)",
        what="For active customers, the actual window of their next order (rows) against the model's prediction (columns).",
        read="The diagonal is correct predictions. Off-diagonal cells are mistakes, such as predicting '0–14 days' for someone who waited 46+.",
        made="Random forest classifier, three classes (0–14, 15–45, 46+ days), 70/30 split.",
        why="A follow-up message works best just before a customer would normally buy again.",
        careful="Accuracy is modest (about 61% vs 51% by guessing the most common window). Use it to time messages, not for big spending decisions."),
    # ------------------------------------------------------------------ 5 · Problem 1
    "p1_month": dict(
        title="Late-delivery rate by month",
        what="The share of orders delivered after the promised date each month (line), with order volume behind it (bars).",
        read="Left axis = late %. Right axis = orders. Watch the line jump when the bars jump.",
        made="Synthetic orders, Nov 2025 – Oct 2026; late = actual delivery day after the promised day. Follows the sidebar filters.",
        why="Lateness spikes in sale periods like Yellow Friday, so one fixed 'tomorrow' promise cannot be right in both quiet and peak weeks.",
        careful="Sale dates are approximate."),
    "p1_factor_chart": dict(
        title="What makes an order late?",
        what="The late rate split by the factor you pick: who ships it, courier, item size, address quality, order time, zone type, sale event or emirate.",
        read="Longer bar = more often late. Hover to see late orders and total orders for each group.",
        made="Grouped averages on the synthetic orders; follows the sidebar filters.",
        why="Points to the causes operations can act on: sellers who ship themselves, 3PL couriers, bulky items, weak addresses.",
        careful="Each bar looks at one factor at a time; the Apriori rules on the next tab show combinations."),
    "p1_map": dict(
        title="Where late orders are concentrated (map)",
        what="22 UAE neighbourhoods. Bubble size = orders, colour = late rate (green low, red high).",
        read="Big red bubbles are busy zones with many late orders. Hover for the zone and its group.",
        made="Zone late rates from the synthetic orders; zone centres are approximate. Use 'Simple plot (offline)' if map tiles do not load.",
        why="Remote and villa zones need different capacity or promises than dense urban areas.",
        careful="Coordinates are approximate area centres, for display only."),
    "p1_update": dict(
        title="Silence makes it worse",
        what="For late orders, the share of customers who bought again within 60 days: with an early update vs without.",
        read="Compare the two bars. The boxes beside show how rarely updates are sent today.",
        made="Synthetic late orders split by whether a proactive update was sent.",
        why="Telling the customer first is cheap and protects repeat business. It is step one of Solution 1.",
        careful="This is an observational comparison; the A/B test on tab 4 is the proper proof."),
    "p1_sellers": dict(
        title="Seller tiers (K-Means)",
        what="Each seller placed by average hours to hand a parcel to the courier (x) and late rate (y), coloured by tier.",
        read="Bubble size = orders. Red dots far right and high are the poor sellers: slow handover and mostly late.",
        made="K-Means with 3 groups on standardised handover time, late rate and return rate, for sellers with 20+ orders.",
        why="Poor sellers lose the express badge and get a longer promised date until they improve.",
        careful="Three groups were kept because they are easy to act on."),
    "p1_sk": dict(
        title="Choosing the number of seller tiers",
        what="Silhouette score for 2 to 6 seller groups; the dark bar is the number kept.",
        read="Higher = clearer separation.",
        made="K-Means on the seller features above, for each k.",
        why="Evidence for three tiers.",
        careful="Business meaning also decided the choice."),
    "p1_zones_bar": dict(
        title="Problem zones (K-Means)",
        what="Late rate for each zone, with its group (Reliable, Watch, Chronic problem) in brackets.",
        read="Longest bars are the worst zones. Hover for order counts.",
        made="K-Means with 3 groups on zone late rate, average days late, failed-attempt rate and volume.",
        why="Chronic zones get extra capacity, earlier cut-offs or a longer honest promise.",
        careful="Small zones can move a lot from month to month."),
    "p1_rules_chart": dict(
        title="Conditions that happen together before a delay (Apriori)",
        what="Each bubble is an 'IF these conditions, THEN late' rule.",
        read="Right = the pattern is more common (support). Higher = when the IF happens, the order is late more often (confidence). "
             "Bigger and darker = stronger than chance (lift). The table beside lists the rules.",
        made="Apriori on yes/no items per order: sale day, Yellow Friday, bulky, evening, partial address, ships directly, poor seller, "
             "3PL courier, villa/remote zone. Rules kept only if lift > 1 and clearly stronger than any shorter rule inside them.",
        why="Combinations matter more than single factors, e.g. a poor seller who ships directly with a 3PL courier is late about 90% of the time.",
        careful="Association shows what happens together, not proof of cause."),
    "p1_roc": dict(
        title="Late-delivery model: ROC curve",
        what="How well three models rank orders from most to least likely to be late, on unseen months.",
        read="Closer to the top-left = better. AUC in the legend (0.5 = guessing, 1 = perfect).",
        made="Logistic regression, random forest and XGBoost trained on Nov 2025 – Jul 2026 and tested on Aug – Oct 2026. "
             "Seller history comes from the training months only.",
        why="Boosted trees reach AUC about 0.86, good enough to act on orders before they are late.",
        careful="Real accuracy will differ; the comparison method stays the same."),
    "p1_imp": dict(
        title="What drives late deliveries",
        what="The features the XGBoost late-order model uses most.",
        read="Longer bar = bigger influence on the prediction.",
        made="XGBoost feature importance, top 12.",
        why="Seller handover history, who ships the item and the courier dominate, so they are where operations should start.",
        careful="Importance shows influence, not direction or cause."),
    "p1_cm": dict(
        title="Late-delivery model: confusion matrix",
        what="On unseen orders: how many on-time and late orders the model predicted correctly and wrongly.",
        read="Top-left = on time and predicted on time. Bottom-right = late and caught. Bottom-left = late orders missed. Top-right = false alarms.",
        made="XGBoost with a 0.4 threshold (an order is flagged if its late risk is 40% or more).",
        why="0.4 rather than 0.5 catches more late orders, because missing one costs more than a false alarm.",
        careful="Moving the threshold trades missed late orders for false alarms."),
    "p1_reg": dict(
        title="An honest delivery date: predicted vs actual days",
        what="For unseen orders, predicted delivery days (x) against actual days (y).",
        read="Dots on the dotted line are perfect. The spread shows how uncertain delivery time is.",
        made="XGBoost regression predicts the average time; a second quantile model predicts P90, the time by which 90% of similar orders arrive.",
        why="Promising the average means being late about half the time. Promising P90 means 9 in 10 orders arrive by the date.",
        careful="A sample of 1,500 orders is plotted."),
    "p1_kept": dict(
        title="Promise kept: today vs P90 date, by emirate",
        what="The share of orders delivered by the promised date, today and if the P90 date were promised.",
        read="Grey = today, green = P90 promise. A taller green bar = more promises kept.",
        made="Test months Aug – Oct 2026. Today's promise is 1 day (2 outside Dubai, Sharjah and Ajman, +1 for bulky).",
        why="Promise kept rises from about 70% to 96% overall.",
        careful="The P90 date is longer (about 2.2 vs 1.4 days). Whether that costs sales must be tested."),
    "p1_addr": dict(
        title="Failed first attempts by address quality and zone type",
        what="The share of orders where the driver could not deliver at the first try, by address type (columns) and zone type (rows).",
        read="Darker cell = more failures. Partial addresses in villa communities are the worst.",
        made="Synthetic orders; a logistic-regression model on the same inputs scores each order's failure risk (AUC shown above).",
        why="Those customers get a WhatsApp request for a location pin before dispatch, and drivers can only mark 'unavailable' with GPS proof.",
        careful="Synthetic data."),
    "p1_ab": dict(
        title="A/B test: early message + credit for likely-late orders",
        what="Repeat purchase within 60 days for late orders in a randomised test (Apr – Jun 2026): control (today) vs treatment (message + AED 15 credit).",
        read="Compare the two bars. The boxes beside give the uplift, its 95% confidence interval and the p-value.",
        made="Late orders were split 50/50 at random; two-proportion z-test.",
        why="+6.4 points repeat buying with p ≈ 0.006 means the effect is very unlikely to be chance.",
        careful="One 3-month test; effects can fade and should be re-tested."),
    "p1_uplift": dict(
        title="Who should get the credit? (uplift)",
        what="Customers in the test ranked by predicted uplift into five groups, with the real extra repeat buying the credit caused in each.",
        read="Green bars = the credit helped; red = it did not. The top groups gain most.",
        made="Two-model uplift: one random forest predicts repeat buying with a credit, another without. Each customer is scored by a model trained on the other half.",
        why="Giving credits only to the top groups gets most of the benefit for well under half the cost.",
        careful="Estimated from one test; re-check every few months."),
    "p1_demand": dict(
        title="Daily order forecast: Seasonal ARIMA vs regression",
        what="Daily orders across all regions (dark line), each model's fit on training data (dotted), its forecast for the test months and the next 45 days (thick), and an 80% range (shaded).",
        read="Left of the first dotted line = training. Between the lines = test months forecast 92 days ahead without seeing them. "
             "Right of the second line = the future forecast. Switch models with the buttons; drag or double-click to zoom.",
        made="Seasonal ARIMA (0,1,1)(0,1,1)[7] on log orders with yearly waves and sale-event flags as inputs, and a regression with "
             "trend, weekday, yearly waves and events. Both trained to July 2026.",
        why="Both are about 2.4% off on unseen months and both see Yellow Friday more than doubling orders, so couriers and agents can be booked weeks ahead.",
        careful="Synthetic series; the range widens the further ahead it looks."),
    "p1_sarima_acf": dict(
        title="Residual check for Seasonal ARIMA",
        what="The correlation of the model's leftover errors with themselves 1 to 21 days earlier.",
        read="Bars inside the dotted lines = no pattern left. A bar outside (red) would mean the model missed something at that lag.",
        made="Autocorrelation of the training residuals; the Ljung-Box test below checks the first 14 lags together.",
        why="A p-value above 0.05 (here about 0.26) means the errors look like random noise, so the model has captured the pattern.",
        careful="Passing this test does not guarantee future accuracy."),
    "p1_events": dict(
        title="Sale-day effects",
        what="How many extra orders each sale event brings compared with a normal day.",
        read="Longer bar = bigger surge. Yellow Friday is the largest.",
        made="Coefficients of the event flags in the regression model, converted to percent.",
        why="Tells planners how much extra courier and support capacity to book for each event.",
        careful="Synthetic series; real effects depend on each year's promotions."),
    # ------------------------------------------------------------------ 6 · Problem 2
    "p2_reasons": dict(
        title="Why customers return",
        what="The share of returns for each reason.",
        read="Longer bar = more common reason.",
        made="Return reasons from the synthetic returns, after cleaning mapped different spellings to six labels. Follows the sidebar filters.",
        why="Size and fit dominate, which points to fixes at the source like size guides.",
        careful="Customers sometimes pick a reason that gets the fastest refund."),
    "p2_cat": dict(
        title="Return rate by category",
        what="Returns as a share of orders in each category.",
        read="Longer bar = returned more often. Hover for counts.",
        made="Returns divided by orders per category.",
        why="Fashion is returned far more than anything else, mostly for size.",
        careful="Synthetic data."),
    "p2_wait": dict(
        title="Slow refunds cost repeat business",
        what="The share of customers who bought again within 90 days, grouped by how long their refund took, plus those whose refund was rejected.",
        read="Bars from left to right get slower. The last bar is rejected refunds, the worst outcome.",
        made="Synthetic returns grouped by refund days.",
        why="The faster the refund, the more likely the customer comes back. This is the core reason for Solution 2.",
        careful="Observational; the randomised test on tab 6 measures the real effect."),
    "p2_seg": dict(
        title="Who returns: customer segments",
        what="The five customer groups placed by returns per order (x) and average next-year profit (y).",
        read="Bubble size = number of customers. Top-left (loyal low-returners) is the group to protect. Right-hand bubbles are where returns come from.",
        made="K-Means on recency, frequency, spend, return rate and rejected refunds, five groups.",
        why="Most returners are loyal and honest; a size-swapper group needs size help, and a small disputed group (about 4%) is where possible abuse sits.",
        careful="Segment names are our reading of each group's averages."),
    "p2_k": dict(
        title="Choosing the number of return segments",
        what="Silhouette score for 3 to 7 groups; dark bar = number kept.",
        read="Higher = clearer groups.",
        made="K-Means for each k on the same standardised features.",
        why="Evidence for five segments.",
        careful="Business meaning also decided the choice."),
    "p2_roc": dict(
        title="Return-fails-QC model: ROC curve",
        what="How well three models rank returns from safest to riskiest, on returns they did not train on.",
        read="Closer to the top-left = better ranking. AUC: 0.5 = guessing, 1 = perfect.",
        made="Logistic regression, random forest and XGBoost on customer history, item, reason, timing and photos; 70/30 split.",
        why="Random forest (AUC about 0.79) is good at ranking, which is how trust tiers use it.",
        careful="Only about 8% of returns fail, so ranking (AUC) matters more than accuracy."),
    "p2_imp": dict(
        title="What predicts a failed return",
        what="The features the random forest relies on most.",
        read="Longer bar = bigger influence.",
        made="Random forest feature importance.",
        why="Prior returns, account age, item value and how fast the item came back matter most, so a rejection can be explained to the customer.",
        careful="Importance is not cause."),
    "p2_cm": dict(
        title="Return-fails-QC model: confusion matrix",
        what="On 1,464 unseen returns: 1,284 passes predicted correctly, 59 honest returns wrongly flagged, 73 failures missed, 48 failures caught.",
        read="Bottom-right = failures caught. Bottom-left = failures missed. Top-right = honest customers wrongly flagged.",
        made="Random forest with a 0.5 threshold.",
        why="As a yes/no judge it is weak (catches 40% of failures; right 45% of the time when it flags). Saying 'always passes' would be 91.7% accurate. "
            "So it is used to rank returns into tiers, never to reject a return on its own.",
        careful="Accuracy alone (91%) is misleading because failures are rare."),
    "p2_rules_chart": dict(
        title="What goes with a return (Apriori)",
        what="Each bubble is an 'IF these happen, THEN the order is returned' rule.",
        read="Right = more common. Higher = more reliable. Bigger and darker = stronger than chance (lift).",
        made="Apriori on yes/no items per order: fashion, electronics, two sizes ordered, poor-quality seller, late delivery, sale day, seller ships directly, value AED 500+.",
        why="Ordering two sizes is followed by a return every time (lift about 6); fashion from weak sellers or delivered late comes back about 2.5× more often. "
            "Fixes: size guide, seller checks, honest delivery dates.",
        careful="The 100% is from synthetic data; real data would be lower but still high."),
    "p2_coef": dict(
        title="What makes a refund slow (regression)",
        what="Extra refund days each factor adds compared with a wallet refund for an item fulfilled by noon.",
        read="Bars to the right add days; bars to the left save days.",
        made="Linear regression of refund days on refund method, fulfilment, category, emirate and item value, on standard-process returns.",
        why="Card refunds and seller-shipped items add the most days, so wallet refunds become the default and every customer sees a guaranteed date.",
        careful="Linear model; individual refunds vary."),
    "p2_keep": dict(
        title="'Keep it' partial refund",
        what="For each category: average item value, predicted cost of taking it back, and the keep-it offer (80% of that cost).",
        read="Where the peach bar (cost) is close to or above the grey bar (value), taking the item back is not worth it.",
        made="Linear regression of return cost on item size, category and value; return cost = two-way shipping + handling + write-down.",
        why="For cheap or damaged items, letting the customer keep it for a partial refund saves money for both sides.",
        careful="Return costs use assumed shipping and handling rates."),
    "p2_ab": dict(
        title="A/B test: instant vs standard refund",
        what="Repeat purchase within 90 days for low-risk returns in a randomised test (May – Oct 2026).",
        read="Compare the two bars. Boxes beside: uplift, 95% confidence interval, p-value.",
        made="Low-risk returns split 50/50 at random; two-proportion z-test.",
        why="+5.9 points, p ≈ 0.009: instant refunds make customers come back more.",
        careful="The test used a simple low-risk rule; 4.7% of instant refunds later failed the check, which is why the stricter model-based Green tier is used."),
    # ------------------------------------------------------------------ 7 · Problem 3
    "p3_topics": dict(
        title="What customers complain about",
        what="The share of support tickets by topic.",
        read="Longer bar = more tickets.",
        made="Synthetic tickets created by late orders, failed deliveries, slow or rejected refunds, defects and payment issues. Follows the emirate and month filters.",
        why="Delivery delays create the most tickets, so fixing Problem 1 also shrinks Problem 3.",
        careful="Synthetic data."),
    "p3_res_box": dict(
        title="Hours to resolve, by topic",
        what="The spread of resolution times for each ticket topic.",
        read="The box covers the middle half of tickets; the line inside is the median; whiskers show the usual range.",
        made="Synthetic resolution hours.",
        why="Warranty and rejected-refund cases take days; they need an owner and a deadline most.",
        careful="Extreme cases are hidden to keep the chart readable."),
    "p3_csat": dict(
        title="Template replies and repeat contacts drag satisfaction down",
        what="Average satisfaction (CSAT, 1–5) by reply type and whether the customer had to contact again.",
        read="Taller = happier. The lowest bar is template reply + had to contact again.",
        made="Synthetic tickets grouped by template_reply and repeat_contact.",
        why="The problem is ownership and fixing things first time, not the number of agents.",
        careful="Synthetic data."),
    "p3_cm": dict(
        title="Reading tickets automatically (confusion matrix)",
        what="For unseen tickets: the actual topic (rows) against the topic the text model predicted (columns), as % of each actual topic.",
        read="A strong diagonal = correct. Off-diagonal cells show which topics get mixed up.",
        made="TF-IDF turns text into numbers (rare, telling words weigh more); logistic regression learns which words go with which topic. 75/25 split.",
        why="About 97.5% correct, so tickets go straight to the right team. Mistakes happen mainly when a ticket mentions two issues.",
        careful="Synthetic tickets use simple wording; real accuracy will be lower."),
    "p3_themes": dict(
        title="Complaint themes found without labels",
        what="Seven groups of tickets that use similar words, with their top words and share of all tickets.",
        read="Longer bar = bigger theme. The label shows the theme's top three words.",
        made="TF-IDF → reduced to 60 dimensions (SVD) → K-Means with 7 groups. The algorithm never sees the real topic.",
        why="Managers see what customers actually say, including unplanned themes such as complaints about 'copy paste' replies.",
        careful="One theme mixes refund and delivery wording; mixed themes are reviewed by a person."),
    "p3_k": dict(
        title="Choosing the number of complaint themes",
        what="Silhouette score for 4 to 10 text groups; dark bar = number kept.",
        read="Higher = clearer groups. It keeps rising slowly, so we stopped where themes are still easy to name.",
        made="K-Means for each k on the reduced TF-IDF vectors.",
        why="Seven themes can each be handed to a team.",
        careful="Text clusters are rarely perfectly separated."),
    "p3_month": dict(
        title="Monthly complaint volume by topic",
        what="Tickets per month, stacked by topic.",
        read="Bar height = all tickets that month; colours = topics.",
        made="Synthetic tickets by creation month (incomplete November excluded).",
        why="Shared weekly with operations so the team that owns the cause sees its trend.",
        careful="Synthetic data."),
    "p3_coef": dict(
        title="A realistic deadline: what adds hours to a case",
        what="Extra resolution hours each factor adds compared with a chat ticket about a late delivery.",
        read="Bars to the right add hours.",
        made="Linear regression of resolution hours on topic, channel, number of contacts and template replies.",
        why="Gives each case a deadline the customer can see ('resolved by Thursday').",
        careful="Linear model; individual cases vary."),
    "p3_roc": dict(
        title="Churn after a complaint: ROC curve",
        what="How well three models rank customers who complained by their chance of stopping buying within 90 days.",
        read="Closer to the top-left = better. AUC around 0.70 here.",
        made="Logistic regression, random forest and XGBoost on topic, channel, contacts, hours, reopened, CSAT, template, tenure, order value and anger.",
        why="High churn risk plus high CLV routes the case to a senior agent with a 24-hour target.",
        careful="Churn depends on much more than one ticket, so moderate accuracy is expected."),
    "p3_imp": dict(
        title="What drives churn after a complaint",
        what="The features the XGBoost churn model uses most.",
        read="Longer bar = bigger influence.",
        made="XGBoost feature importance.",
        why="Repeat contacts, long resolution, reopened tickets and low CSAT push churn up.",
        careful="Importance is not cause."),
    "p3_bc": dict(
        title="Churn by number of contacts",
        what="The share of customers who stopped buying within 90 days, by how many times they contacted support on the same issue.",
        read="Bars rise from left to right.",
        made="Synthetic tickets grouped by contacts. A market-response regression puts each extra contact at about +9 points of churn.",
        why="Puts a price on slow service: fixing a case at the second contact is cheaper than losing the customer.",
        careful="Linear approximation on synthetic data."),
    "p3_br": dict(
        title="Churn by time to resolve",
        what="The share of customers who stopped buying within 90 days, by how long their case took.",
        read="Bars rise as cases take longer.",
        made="Synthetic tickets grouped by resolution time. Each extra day adds about 3 points of churn.",
        why="Visible deadlines and priority routing directly protect revenue.",
        careful="Synthetic data."),
    # ------------------------------------------------------------------ 8 · Simulator
    "sim1_d1": dict(
        title="Simulator · Problem 1: late rate by who ships",
        what="In the 7,592 simulation orders, the late rate for items fulfilled by noon vs shipped by the seller.",
        read="Longer bar = more often late.",
        made="Aug – Oct 2026 orders the models never trained on.",
        why="Seller-shipped items are late far more often.",
        careful="Descriptive only."),
    "sim1_d2": dict(
        title="Simulator · Problem 1: late rate by courier",
        what="Late rate for the in-house fleet and two 3PL partners.",
        read="Longer bar = more often late.",
        made="Same simulation orders.",
        why="3PL partners are slower, so risky orders go to the in-house fleet.",
        careful="Descriptive only."),
    "sim1_d3": dict(
        title="Simulator · Problem 1: risk tiers vs reality",
        what="For each risk tier from the late-order model, the share of orders that really were late.",
        read="High-risk orders were late about 82% of the time, low-risk orders rarely.",
        made="Model risk scores on unseen orders.",
        why="Shows the model's tiers line up with what happened, so acting on them is justified.",
        careful="Tier cut-offs are 0.25 and 0.55."),
    "sim1_live": dict(
        title="Simulator · Problem 1: late orders as the run progresses",
        what="Cumulative late orders while the records are replayed: today's process (grey) vs the solution (green).",
        read="The gap between the lines is the late orders the solution prevents.",
        made="Each order replayed twice with the same random draw; only the process differs (P90 promise, priority courier, pin request).",
        why="Shows the effect building up order by order.",
        careful="Uses the assumptions in the expander above."),
    "sim1_res_em": dict(
        title="Simulator · Problem 1 result: late rate by emirate",
        what="Late rate today vs with the solution, for each emirate.",
        read="Grey = today, green = solution.",
        made="Result of the replay.",
        why="Every emirate improves, with the biggest gains where promises were most over-optimistic.",
        careful="Does not model customers who might not order because of a longer date."),
    "sim2_d1": dict(
        title="Simulator · Problem 2: why customers return",
        what="Return reasons in the 1,198 simulation returns.",
        read="Longer bar = more common.",
        made="Test-split returns the model never trained on.",
        why="Context before the run.",
        careful="Descriptive only."),
    "sim2_d2": dict(
        title="Simulator · Problem 2: model risk vs real failures",
        what="Returns grouped by predicted fail risk, with the share that really failed the check.",
        read="Low-risk returns almost never fail; high-risk returns often do.",
        made="Random forest scores on unseen returns.",
        why="Shows instant refunds for low-risk returns are safe.",
        careful="Bands use the default cut-offs 0.15 and 0.50."),
    "sim2_live": dict(
        title="Simulator · Problem 2: refund-chase tickets as the run progresses",
        what="Expected cumulative refund-chase tickets: today vs with trust tiers.",
        read="The gap is the complaints the solution prevents.",
        made="Each return replayed twice with the same random draw; Green refunded in about 1 day, Amber in 3, Red after a full check.",
        why="Faster refunds mean fewer customers chasing support.",
        careful="Uses the assumptions in the expander."),
    "sim2_res": dict(
        title="Simulator · Problem 2 result: median refund days by tier",
        what="Median refund time today vs with the solution, for Green, Amber and Red returns.",
        read="Grey = today, green = solution. Red stays the same because it is still fully checked.",
        made="Result of the replay.",
        why="The honest majority are refunded in about a day.",
        careful="Change the Green cut-off in the assumptions to trade speed for risk."),
    "sim3_d1": dict(
        title="Simulator · Problem 3: ticket topics",
        what="Topics of the 2,129 simulation tickets.",
        read="Longer bar = more tickets.",
        made="Test-split tickets.",
        why="Context before the run.",
        careful="Descriptive only."),
    "sim3_d2": dict(
        title="Simulator · Problem 3: churn by contacts",
        what="The share of customers who stopped buying, by contacts on the same issue.",
        read="Bars rise with more contacts.",
        made="Test-split tickets.",
        why="Shows why cutting repeat contacts matters.",
        careful="Descriptive only."),
    "sim3_live": dict(
        title="Simulator · Problem 3: customers lost as the run progresses",
        what="Cumulative customers who stop buying: today vs with the Case Ownership Engine.",
        read="The gap is the customers the solution keeps.",
        made="Each ticket replayed twice with the same random draw; owner, deadline, priority routing and suggested fix applied.",
        why="Shows service fixes protect revenue.",
        careful="Effects of ownership and fixes are assumptions from the proposal."),
    "sim3_res": dict(
        title="Simulator · Problem 3 result: churn by topic",
        what="Churn rate today vs with the solution for each ticket topic.",
        read="Grey = today, green = solution.",
        made="Result of the replay.",
        why="Rejected refunds and warranty cases improve most because they had the most repeat contacts.",
        careful="Assumption-driven; the real test is a 3-arm A/B."),
    # ------------------------------------------------------------------ 9 · Live platform
    "live_late": dict(
        title="Live platform: late-delivery rate",
        what="The share of orders due so far that arrived late, over simulated time, in both copies of the shop.",
        read="Grey = today's process, green = with solutions. It updates every 1.5 seconds while the platform runs.",
        made="Computed from the live database after every step. Both copies receive exactly the same customers and orders.",
        why="Shows the delivery solution working in real time, especially during the Yellow Friday surge.",
        careful="Event rules are simplified versions of the models."),
    "live_tickets": dict(
        title="Live platform: open complaint tickets",
        what="The number of unresolved tickets over simulated time in both copies.",
        read="Lower is better. The gap grows as unhandled problems pile up today.",
        made="Live database; tickets open from late orders, failed deliveries and slow or rejected refunds, and close at their resolution time.",
        why="Shows how fixing delivery and refunds, plus case ownership, keeps the support queue small.",
        careful="Simplified rules."),
    "live_buckets": dict(
        title="Live platform: category buckets",
        what="Sales value and order count per category in the last 24 simulated hours (solution copy).",
        read="Longer bar = more money in that category bucket.",
        made="Live order table summed by category.",
        why="What a category manager would watch during a sale.",
        careful="Both copies have the same sales; only the process differs."),
    # ------------------------------------------------------------------ 10 · Business case
    "bc_total": dict(
        title="Net gain per month by solution",
        what="The monthly net gain of each solution using the inputs typed into the calculator.",
        read="Taller bar = more value. Change any input above and the bars update.",
        made="Extra customers kept × value per customer − cost (credits or extra fraud), plus contact cost saved for Solution 3.",
        why="Puts all three solutions on one money scale.",
        careful="Default inputs are assumptions from the group's proposal, not noon's real figures."),
}

# keys passed to the shared helpers in views/common.py that reuse the same chart type
INFO["p1_ab_chart"] = INFO["p1_ab"]
INFO["p2_ab_chart"] = INFO["p2_ab"]


# --- explanations updated for the realistic data; {placeholders} are filled from the results at run time
_UPDATES = [
    ('cu_clv_scatter', 'made', 'XGBoost regression on orders, spend, recency, tenure, late orders, failed deliveries, returns, rejected refunds, tickets and repeat contacts. 70/30 split. R² about {clv_r2}.'),
    ('cu_clv_scatter', 'careful', "Next year's profit comes from a separate simulated future (hidden buying rates, people moving away), so a large part is unpredictable from history. Profit assumes an 11% margin."),
    ('cu_churn_imp', 'why', 'Recency and order frequency matter most, but late orders, rejected refunds and repeat contacts also appear, linking churn to the three problems.'),
    ('cu_churn_roc', 'careful', 'AUC about {cust_churn_auc}: churn depends on life events no data shows, so moderate accuracy is expected.'),
    ('cu_npd', 'careful', 'Honest result: accuracy {npd_acc} vs {npd_base} by always guessing the most common window — no better than guessing. Next-purchase timing is not predictable from these features, so it is not used for decisions.'),
    ('p1_rules_chart', 'why', 'Combinations matter more than single factors, e.g. a poor seller who ships directly with a 3PL courier is late about {p1_top_rule} of the time.'),
    ('p1_roc', 'why', 'All three models reach AUC about {p1_auc}; part of every delay (driver, traffic, depot load) cannot be seen at checkout, so no model is perfect.'),
    ('p1_roc', 'careful', 'The test months include a courier-contract change and 40 new sellers (drift); page 11 shows month-by-month cross-validation.'),
    ('p1_cm', 'why', '0.4 rather than 0.5 catches more late orders, because missing one costs more than a false alarm; page 11 finds the cost-minimising threshold.'),
    ('p1_kept', 'why', 'Promise kept rises from about {p1_kept_today} to {p1_kept_p90} overall.'),
    ('p1_kept', 'careful', 'The P90 date is longer (about {p1_p90_days} vs {p1_today_days} days). Whether that costs sales must be tested.'),
    ('p1_ab', 'why', '+{p1_ab_up} points repeat buying with p ≈ {p1_ab_p}: unlikely to be chance, but the 95% range is wide ({p1_ab_lo} to {p1_ab_hi} points).'),
    ('p1_uplift', 'read', 'Green bars = the credit helped; red = it did not. If the bars do not fall steadily from left to right, the model is not ranking reliably.'),
    ('p1_uplift', 'why', 'Targeting only pays if ranking beats random; the Qini curve below checks this. With one small test it does not reliably, so credits go to all likely-late orders until a larger test.'),
    ('p1_demand', 'made', 'Seasonal ARIMA {sarima_order} on log orders with yearly waves, sale events, the 14 Jun outage and the new Abu Dhabi hub as inputs, and a regression with trend, weekday, yearly waves and the same flags. Both trained to July 2026.'),
    ('p1_demand', 'why', 'On unseen months SARIMA is about {sarima_mape} off and the regression {reg_mape} (repeating last week: {naive_mape}). Both see Yellow Friday more than doubling orders, so couriers and agents can be booked weeks ahead.'),
    ('p1_sarima_acf', 'why', 'A p-value above 0.05 (here about {ljung}) means the errors look like random noise, so the model has captured the pattern.'),
    ('p2_roc', 'why', 'The models rank reasonably (AUC about {p2_auc}), which is how trust tiers use them; 5-fold cross-validation then picked {p2_model}.'),
    ('p2_imp', 'why', 'Prior returns, account age, how fast the item came back and missing photos matter most, so a slower refund can be explained to the customer.'),
    ('p2_cm', 'what', 'On {p2_n} unseen returns, flagging the riskiest (score ≥ {p2_red}, the Red cut-off): {p2_tn} passes correctly left alone, {p2_fp} honest returns sent to inspection, {p2_fn} failures missed, {p2_tp} failures caught.'),
    ('p2_cm', 'made', 'The cross-validated, calibrated model at the Red cut-off (top 5% of risk scores).'),
    ('p2_cm', 'why', "As a yes/no judge it is weak (catches {p2_recall} of failures). Saying 'always passes' would be {p2_always} accurate. So it only decides how FAST to refund, never whether to refund: Red means 'inspect first'."),
    ('p2_cm', 'careful', 'Accuracy alone is misleading because failures are rare.'),
    ('p2_rules_chart', 'why', 'Ordering two sizes is followed by a return {p2_two_conf} of the time (lift about {p2_two_lift}); fashion from weak sellers or delivered late comes back about 2.5× more often. Fixes: size guide, seller checks, honest delivery dates.'),
    ('p2_rules_chart', 'careful', 'The two-size flag is recorded on every order (not only returned ones), so the rule is not circular.'),
    ('p2_ab', 'why', '+{p2_ab_up} points, p ≈ {p2_ab_p}: instant refunds make customers come back more (95% range {p2_ab_lo} to {p2_ab_hi} points).'),
    ('p2_ab', 'careful', 'The test used a simple low-risk rule; {p2_guard} of instant refunds later failed the check, which is why the stricter model-based Green tier is used.'),
    ('p3_cm', 'why', 'About {p3_acc} correct on messy text (typos, Arabizi, two issues in one message, 6% wrong agent tags), so most tickets go straight to the right team; unclear ones go to a person.'),
    ('p3_cm', 'careful', "Some 'mistakes' are the agent's tag being wrong, not the model."),
    ('p3_themes', 'careful', 'One large theme mixes many short, general messages; mixed themes are reviewed by a person. Greeting words are removed first.'),
    ('p3_roc', 'read', 'Closer to the top-left = better. AUC around {p3_auc} here.'),
    ('p3_bc', 'made', 'Synthetic tickets grouped by contacts. A market-response regression puts each extra contact at about +{p3_contact} points of churn.'),
    ('p3_br', 'made', 'Synthetic tickets grouped by resolution time. Each extra day adds about {p3_day} points of churn.'),
    ('sim1_d1', 'what', 'In the {sim1_n} simulation orders, the late rate for items fulfilled by noon vs shipped by the seller.'),
    ('sim1_d3', 'read', 'High-risk orders were late about {p1_high_late} of the time, low-risk orders about {p1_low_late}.'),
    ('sim2_d1', 'what', 'Return reasons in the {sim2_n} simulation returns.'),
    ('sim2_d2', 'read', 'Green returns rarely fail; Red returns fail several times more often than average.'),
    ('sim2_d2', 'made', 'Cross-validated, calibrated model scores on unseen returns.'),
    ('sim2_d2', 'careful', 'Bands use the default cut-offs {p2_green} and {p2_red}.'),
]
for _k, _f, _v in _UPDATES:
    if _k in INFO:
        INFO[_k] = dict(INFO[_k], **{_f: _v})
INFO["p1_ab_chart"] = INFO["p1_ab"]
INFO["p2_ab_chart"] = INFO["p2_ab"]


class _Keep(dict):
    def __missing__(self, k):
        return "{" + k + "}"


def _values() -> dict:
    """Numbers quoted in the explanations, read from the results so the text never goes stale."""
    try:
        from noonbi import data as D
        S = D.summary()
        pc = lambda x, d=0: f"{x * 100:.{d}f}%"  # noqa: E731
        p1, p2, p3 = D.table("p1_cls_metrics"), D.table("p2_cls_metrics"), D.table("p3_churn_metrics")
        ab = D.table("ab_tests")
        cm = D.table("p2_confusion").iloc[0]
        pr = D.table("p1_promise_compare")
        rt = D.table("p1_risk_tiers").set_index("risk_tier")
        r2 = D.table("p2_rules")
        two = r2[r2["if"] == "Ordered two sizes"]
        fc = D.table("p1_forecast_compare").set_index("model")["test_mape"]
        n = cm["true_neg"] + cm["false_pos"] + cm["false_neg"] + cm["true_pos"]
        return dict(
            clv_r2=f"{D.table('cust_clv_metrics')['r2'].max():.2f}", cust_churn_auc=f"{D.table('cust_churn_metrics')['roc_auc'].max():.2f}",
            npd_acc=pc(S["customers"]["npd_accuracy"]), npd_base=pc(S["customers"]["npd_baseline"]),
            p1_top_rule=pc(D.table("p1_rules")["confidence"].iloc[0]), p1_auc=f"{p1['roc_auc'].max():.2f}",
            p1_kept_today=pc(pr["kept_rate"].iloc[0]), p1_kept_p90=pc(pr["kept_rate"].iloc[1]),
            p1_today_days=f"{pr['avg_promised_days'].iloc[0]:.1f}", p1_p90_days=f"{pr['avg_promised_days'].iloc[1]:.1f}",
            p1_ab_up=f"{ab['uplift_pts'].iloc[0]:.1f}", p1_ab_p=f"{ab['p_value'].iloc[0]:.3f}",
            p1_ab_lo=f"{ab['ci_low_pts'].iloc[0]:.1f}", p1_ab_hi=f"{ab['ci_high_pts'].iloc[0]:.1f}",
            p2_ab_up=f"{ab['uplift_pts'].iloc[1]:.1f}", p2_ab_p=f"{ab['p_value'].iloc[1]:.3f}",
            p2_ab_lo=f"{ab['ci_low_pts'].iloc[1]:.1f}", p2_ab_hi=f"{ab['ci_high_pts'].iloc[1]:.1f}", p2_guard=pc(ab["guardrail_fail_rate"].iloc[1], 1),
            sarima_order=S["demand"]["sarima_order"], sarima_mape=pc(S["demand"]["sarima_mape_test"], 1),
            reg_mape=pc(fc.get("Time-series regression", S["demand"]["mape_test"]), 1),
            naive_mape=pc(S["demand"]["naive_mape_test"], 1), ljung=f"{S['demand']['sarima_ljungbox_p']:.2f}",
            p2_auc=f"{p2['roc_auc'].max():.2f}", p2_model=S["p2"].get("chosen_model", "the best model"),
            p2_n=f"{int(n):,}", p2_tn=f"{int(cm['true_neg']):,}", p2_fp=f"{int(cm['false_pos']):,}", p2_fn=f"{int(cm['false_neg']):,}",
            p2_tp=f"{int(cm['true_pos']):,}", p2_recall=pc(cm["true_pos"] / max(cm["true_pos"] + cm["false_neg"], 1)),
            p2_always=pc((cm["true_neg"] + cm["false_pos"]) / n, 1), p2_red=f"{S['p2']['red_cutoff']:.2f}", p2_green=f"{S['p2']['green_cutoff']:.3f}",
            p2_two_conf=pc(two["confidence"].iloc[0]) if len(two) else "often", p2_two_lift=f"{two['lift'].iloc[0]:.1f}" if len(two) else "high",
            p3_acc=pc(S["p3"]["topic_accuracy"]), p3_auc=f"{p3['roc_auc'].max():.2f}",
            p3_contact=f"{S['p3']['churn_per_contact_pts']:.0f}", p3_day=f"{S['p3']['churn_per_day_pts']:.0f}",
            sim1_n=f"{len(D.table('p1_orders')):,}", sim2_n=f"{len(D.table('p2_returns')):,}",
            p1_high_late=pc(rt.loc["High risk", "actual_late_rate"]), p1_low_late=pc(rt.loc["Low risk", "actual_late_rate"]),
        )
    except Exception:  # pragma: no cover - explanations still render, with placeholders
        return {}


def get(key: str) -> dict | None:
    inf = INFO.get(key)
    if inf is None:
        for prefix in ("sim1_live", "sim2_live", "sim3_live"):
            if key.startswith(prefix):
                inf = INFO[prefix]
    if inf is None:
        return None
    vals = _Keep(_values())
    return {k: (v.format_map(vals) if isinstance(v, str) else v) for k, v in inf.items()}


def markdown(info: dict) -> str:
    return (f"**{info['title']}**\n\n"
            f"**What it shows:** {info['what']}\n\n"
            f"**How to read it:** {info['read']}\n\n"
            f"**How it is made:** {info['made']}\n\n"
            f"**Why it matters:** {info['why']}\n\n"
            f"**Be careful:** {info['careful']}")


# --- charts added with the validation work
INFO.update({
    "p2_calib": dict(
        title="Can the refund-risk scores be read as real chances? (calibration)",
        what="Unseen returns split into ten groups by predicted fail risk: the average prediction (x) against the share that really failed (y).",
        read="Points on the dotted line = a score of 10% really means 1 in 10 fail. Above the line = the model under-states risk.",
        made="Model scores and Platt-calibrated scores (sigmoid fitted by 5-fold cross-validation). The Brier score in the legend is the average squared error.",
        why="The fraud cap is set in money, so the Green cut-off needs honest probabilities, not just a good ranking.",
        careful="Each point is about 140 returns, so points wobble a little."),
    "p2_cutoff_curve": dict(
        title="Speed vs risk: choosing the Green cut-off",
        what="For every possible Green cut-off: the share of returns refunded instantly (green) and the money paid early on items that later fail inspection, as a share of all refund value (red).",
        read="Moving right gives more instant refunds but more money at risk. The dotted red line is the 0.5% cap; the dashed line is your slider.",
        made="Out-of-fold scores on the training returns chose the default ({p2_green}); the chart checks it on unseen returns.",
        why="Turns a policy (fraud budget) into a number the operations team can set and defend.",
        careful="Not every failed check is fraud and some value is recovered, so this is an upper bound."),
    "p1_qini": dict(
        title="Does ranking by the uplift model beat random? (Qini curve)",
        what="Extra repeat buyers gained when credits go to the top x% by predicted uplift (green) vs to a random x% (dotted).",
        read="Green above the dotted line = the model finds persuadable customers. Touching or below = it does not.",
        made="Two-model uplift scored out-of-fold on the randomised test (about 1,750 late orders).",
        why="Decides whether to target credits or simply give them to every likely-late order.",
        careful="Uplift needs much more data than an average effect; a larger test is needed before targeting."),
    "val_p1_time": dict(
        title="Late-delivery model, month by month (time-series cross-validation)",
        what="ROC-AUC when each model is trained on all earlier months and tested on the next month, June to October 2026.",
        read="A flat line = stable model. A dip = something changed that month.",
        made="Expanding-window folds; seller history recomputed from the training months each time.",
        why="Shows how the model behaves in production, where it always predicts the future.",
        careful="The August dip is drift (courier contract, new sellers) — a reason to retrain monthly."),
    "val_cost": dict(
        title="Late-delivery flag: total cost by threshold",
        what="Cost on unseen orders of false alarms (AED 8 each) plus missed late orders (AED 35 each) for every threshold.",
        read="The lowest point is the cheapest threshold; a flat bottom means several thresholds are about as good.",
        made="XGBoost late-risk scores on Aug – Oct 2026 orders; costs are stated assumptions.",
        why="Thresholds should come from business costs, not the default 0.5.",
        careful="Replace the cost assumptions with finance figures."),
    "val_fair": dict(
        title="Fairness: honest returns sent to the inspection tier",
        what="For returns that PASSED inspection, the share the model still sent to Red, by account age and emirate.",
        read="Similar bars = similar treatment. A much longer bar = that group of honest customers is checked more.",
        made="Calibrated QC model at the default cut-offs on unseen returns.",
        why="New accounts are checked far more; Red must stay 'inspect first', never 'refuse', and a photo fast-lane is proposed.",
        careful="Small groups (e.g. Umm Al Quwain) move a lot by chance."),
    "val_backtest": dict(
        title="Forecast back-test: four start dates",
        what="Error (MAPE) of 28-day forecasts started on four different dates, for SARIMA, regression and 'repeat last week'.",
        read="Lower bar = better. Compare the models within each start date.",
        made="Each model refitted on data before the start date only, with event, outage and hub flags.",
        why="One test window can flatter a model; several start dates show which is reliably good.",
        careful="'Repeat last week' looks good in quiet weeks and fails around events."),
    "val_resp_chart": dict(
        title="What the simulator learned from the data (response models)",
        what="The effect of each input on the chosen outcome, with an 80% range from 100 bootstrap refits.",
        read="Right = raises the outcome; left = lowers it. A range crossing zero = not clearly different from no effect.",
        made="Logistic (or linear, for CSAT) regression on the recorded data only; hidden generator variables are never used.",
        why="These are the numbers that turn 'faster refund' into 'more customers come back' in the simulator.",
        careful="Credit and instant-refund effects come from randomised tests; the others are associations."),
})
