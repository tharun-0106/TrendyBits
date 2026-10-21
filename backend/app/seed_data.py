"""Offline seed corpus.

IMPORTANT: every article below is SYNTHETIC DEMO CONTENT written for this repository so the
platform works immediately after cloning (no network, no API keys). The publishers/domains are
used only to exercise the source-trust registry; the stories and URLs are fictional and are
flagged with `demo: true` and shown with a "DEMO SEED" tag in the UI. Use the live RSS scraper
or the IEEE Dataport importer to load real articles.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- publisher registry
# trust_score: 0–100 prior reliability of the outlet (editorial standards, corrections record,
# transparency of ownership, fact-check partnerships). Unknown domains fall back to 35.
SOURCES = [
    dict(domain="thehindu.com", name="The Hindu", trust_score=90, domain_age_years=30, masthead_verified=1,
         corrections_policy=1, owner="THG Publishing Pvt Ltd", contact="readerseditor@thehindu.co.in",
         corrections_record="Publishes a weekly Readers' Editor column; corrections appended to articles.",
         history="National daily founded 1878; IFCN-aligned standards desk."),
    dict(domain="indianexpress.com", name="Indian Express", trust_score=86, domain_age_years=28, masthead_verified=1,
         corrections_policy=1, owner="The Indian Express Group", contact="editor@indianexpress.com",
         corrections_record="Corrections noted at article foot.", history="National daily founded 1932."),
    dict(domain="hindustantimes.com", name="Hindustan Times", trust_score=82, domain_age_years=27, masthead_verified=1,
         corrections_policy=1, owner="HT Media Ltd", contact="feedback@hindustantimes.com",
         corrections_record="Corrections published online.", history="National daily founded 1924."),
    dict(domain="timesofindia.indiatimes.com", name="Times of India", trust_score=78, domain_age_years=27,
         masthead_verified=1, corrections_policy=1, owner="Bennett, Coleman & Co. Ltd",
         contact="toi.letters@timesgroup.com", corrections_record="Corrections column.",
         history="National daily founded 1838."),
    dict(domain="reuters.com", name="Reuters", trust_score=95, domain_age_years=30, masthead_verified=1,
         corrections_policy=1, owner="Thomson Reuters", contact="fact.check@thomsonreuters.com",
         corrections_record="Trust Principles; dedicated Fact Check unit.", history="Wire agency founded 1851."),
    dict(domain="bbc.com", name="BBC", trust_score=93, domain_age_years=29, masthead_verified=1, corrections_policy=1,
         owner="British Broadcasting Corporation", contact="bbc.co.uk/contact",
         corrections_record="Public corrections & clarifications page.", history="Public broadcaster founded 1922."),
    dict(domain="cnn.com", name="CNN", trust_score=85, domain_age_years=30, masthead_verified=1, corrections_policy=1,
         owner="Warner Bros. Discovery", contact="cnn.com/feedback", corrections_record="Corrections appended.",
         history="Cable news network founded 1980."),
    dict(domain="dailymail.co.uk", name="Daily Mail", trust_score=55, domain_age_years=27, masthead_verified=1,
         corrections_policy=1, owner="DMG Media", contact="corrections@dailymail.co.uk",
         corrections_record="Corrections column; several IPSO rulings.", history="Tabloid founded 1896."),
    dict(domain="mof.gov.np", name="Ministry of Finance (Nepal)", trust_score=97, domain_age_years=20,
         masthead_verified=1, corrections_policy=1, owner="Government of Nepal", contact="info@mof.gov.np",
         corrections_record="Official gazette notices.", history="Official government ministry website."),
    dict(domain="election.gov.np", name="Election Commission (Nepal)", trust_score=96, domain_age_years=19,
         masthead_verified=1, corrections_policy=1, owner="Government of Nepal", contact="info@election.gov.np",
         corrections_record="Public notice archive.", history="Constitutional body."),
    dict(domain="ekantipur.com", name="Kantipur", trust_score=84, domain_age_years=25, masthead_verified=1,
         corrections_policy=1, owner="Kantipur Media Group", contact="news@ekantipur.com",
         corrections_record="Corrections published in print and online.", history="Nepali daily founded 1993."),
    dict(domain="kathmandupost.com", name="The Kathmandu Post", trust_score=85, domain_age_years=25,
         masthead_verified=1, corrections_policy=1, owner="Kantipur Media Group", contact="editorial@kathmandupost.com",
         corrections_record="Corrections column.", history="English daily founded 1993."),
    dict(domain="pib.gov.in", name="Press Information Bureau", trust_score=93, domain_age_years=22,
         masthead_verified=1, corrections_policy=1, owner="Government of India", contact="pib.gov.in/contact",
         corrections_record="Official releases; PIB Fact Check unit.", history="Nodal agency of the Government of India."),
    dict(domain="rbi.org.in", name="Reserve Bank of India", trust_score=97, domain_age_years=24, masthead_verified=1,
         corrections_policy=1, owner="Reserve Bank of India", contact="rbi.org.in/contact",
         corrections_record="Official press releases.", history="India's central bank."),
    dict(domain="trendybits.news", name="TrendyBits Desk", trust_score=80, domain_age_years=1, masthead_verified=1,
         corrections_policy=1, owner="TrendyBits (demo)", contact="desk@trendybits.news",
         corrections_record="Corrections appended to reports.", history="Demo newsroom for this project."),
    dict(domain="viral-khabar.blogspot.com", name="Viral Khabar (blog)", trust_score=18, domain_age_years=2,
         masthead_verified=0, corrections_policy=0, owner="Unknown", contact="none listed",
         corrections_record="No corrections record.", history="Anonymous blog; multiple debunked posts."),
    dict(domain="nepalnewsnow24.com", name="NepalNewsNow24", trust_score=35, domain_age_years=0.033,
         masthead_verified=0, corrections_policy=0, owner="Unknown (privacy-protected WHOIS)", contact="none listed",
         corrections_record="No corrections record.", history="Unknown outlet registered 12 days before the post."),
    dict(domain="dailybuzzindia.net", name="Daily Buzz India", trust_score=40, domain_age_years=3, masthead_verified=0,
         corrections_policy=0, owner="Unknown", contact="contact form only",
         corrections_record="No corrections record.", history="Click-driven aggregator."),
    dict(domain="karnatakalocalvoice.blogspot.com", name="Karnataka Local Voice (blog)", trust_score=25,
         domain_age_years=4, masthead_verified=0, corrections_policy=0, owner="Unknown", contact="none listed",
         corrections_record="No corrections record.", history="Personal blog."),
]

ARTICLES = [
    # ---------------------------------------------------------------- disputed viral item (wireframe)
    dict(id="tb-001", topic="Economy", kind="viral", source="NepalNewsNow24", domain="nepalnewsnow24.com",
         url="https://nepalnewsnow24.com/demo/rs-50000-allowance-everyone", published_at="2026-09-02",
         author="Staff (no byline)",
         title="SHOCKER: Government to secretly give Rs 50,000 to everyone — forward immediately!",
         body=("The government will give Rs 50,000 to everyone from next week, according to an anonymous ministry source. "
               "The money is being distributed secretly at ward offices before the official announcement. "
               "No citizenship certificate is needed to claim the money. "
               "Crowds have already gathered at ward offices to collect the allowance today, as shown in the photo below. "
               "Officials do not want ordinary people to know about this scheme, so forward immediately to all your groups before it is deleted."),
         image_caption="Photo: Crowds queue at a ward office to collect the Rs 50,000 allowance today.",
         extra=dict(timeline=[
             ["2026-09-02", "Viral post published on social media"],
             ["2026-09-03", "Fact-check desk receives 42 reader reports"],
             ["2026-09-03", "MoF issues statement: \"No such scheme\" (evening)"],
             ["2026-09-04", "TrendyBits article published; report tagged"],
         ], breadcrumb="Rs 50,000 allowance claim")),
    dict(id="tb-002", topic="Economy", kind="official", source="Ministry of Finance (Nepal)", domain="mof.gov.np",
         url="https://mof.gov.np/demo/notice-2082-05-02", published_at="2026-09-03", author="Ministry of Finance",
         title="MoF Gazette Notice 2082-05-02: No universal Rs 50,000 allowance scheme",
         body=("The Ministry of Finance has not announced any scheme to give Rs 50,000 to every citizen. "
               "No such scheme exists, and no money is being distributed through any government office. "
               "The figure circulating on social media is false and the cited 'ministry source' is not an official spokesperson. "
               "Any genuine relief programme will be announced only through the Nepal Gazette and on mof.gov.np.")),
    dict(id="tb-003", topic="Economy", kind="factcheck", source="Kantipur", domain="ekantipur.com",
         url="https://ekantipur.com/demo/factcheck-rs-50000-claim", published_at="2026-09-03", author="Kantipur Fact-Check Desk",
         title="Fact-check: 'Rs 50,000 to everyone' viral claim is fake",
         body=("A viral post claiming that the government will give Rs 50,000 to everyone is fake. "
               "Kantipur found no gazette notice or Ministry of Finance announcement supporting the claim. "
               "The 'ministry source' quoted in the post is anonymous and could not be identified. "
               "The post uses an urgent call to forward the message, a pattern common in misinformation.")),
    dict(id="tb-004", topic="Society", kind="factcheck", source="Reuters", domain="reuters.com",
         url="https://www.reuters.com/demo/factcheck-2021-flood-relief-photo", published_at="2026-09-03", author="Reuters Fact Check",
         title="Fact Check: Crowd photo from 2021 flood relief shared out of context as allowance queue",
         body=("A photo of crowds queuing at a ward office is being shared with the claim that people are collecting a cash allowance today. "
               "The photo is from 2021 flood relief distribution in Saptari and has been cropped and re-captioned. "
               "It does not show any allowance distribution in 2026. "
               "Reuters rates the post as missing context.")),
    dict(id="tb-005", topic="Politics", kind="official", source="Election Commission (Nepal)", domain="election.gov.np",
         url="https://election.gov.np/demo/public-notice-voter-roll-update", published_at="2026-10-16",
         author="Election Commission",
         title="EC publishes voter roll update ahead of local by-elections",
         body=("The Election Commission published the updated voter roll for the upcoming local by-elections on 16 October 2026. "
               "Voters can verify their names online or at district election offices until 30 October. "
               "The Commission said 1.2 million new voters were added since the last update.")),
    # ---------------------------------------------------------------- verified news + corroboration
    dict(id="tb-006", topic="Society", kind="news", source="The Hindu", domain="thehindu.com",
         url="https://www.thehindu.com/demo/kerala-monsoon-floods-red-alert", published_at="2026-10-17", author="K. S. Sudhi",
         title="Monsoon floods in Kerala: IMD issues red alert in five districts",
         body=("The India Meteorological Department issued a red alert for five districts in Kerala on 17 October 2026 as heavy monsoon rain continued. "
               "Idukki, Ernakulam, Kottayam, Pathanamthitta and Alappuzha recorded more than 200 mm of rainfall in 24 hours. "
               "The State Disaster Management Authority said 3,400 people had been moved to 46 relief camps. "
               "Schools in the five districts will remain closed on Saturday.")),
    dict(id="tb-007", topic="Society", kind="news", source="Times of India", domain="timesofindia.indiatimes.com",
         url="https://timesofindia.indiatimes.com/demo/kerala-relief-camps", published_at="2026-10-18", author="TOI Kochi",
         title="Kerala floods: 46 relief camps opened as rain lashes central districts",
         body=("Kerala opened 46 relief camps after flooding in central districts, officials said on 18 October 2026. "
               "The India Meteorological Department kept a red alert in place for five districts. "
               "NDRF teams were deployed in Idukki and Alappuzha.")),
    dict(id="tb-008", topic="Tech", kind="news", source="Indian Express", domain="indianexpress.com",
         url="https://indianexpress.com/demo/isro-launches-eos-09", published_at="2026-10-15", author="Express News Service",
         title="ISRO successfully launches EOS-09 earth observation satellite from Sriharikota",
         body=("ISRO launched the EOS-09 earth observation satellite from Sriharikota on 15 October 2026 aboard the PSLV-C62 rocket. "
               "The satellite was placed in a sun-synchronous orbit about 19 minutes after lift-off. "
               "ISRO chairman V. Narayanan said the mission will support agriculture and disaster monitoring.")),
    dict(id="tb-009", topic="Tech", kind="official", source="Press Information Bureau", domain="pib.gov.in",
         url="https://pib.gov.in/demo/isro-eos-09-release", published_at="2026-10-15", author="PIB Delhi",
         title="PSLV-C62 places EOS-09 in orbit: official release",
         body=("The PSLV-C62 mission successfully placed the EOS-09 satellite in its intended orbit on 15 October 2026. "
               "The launch took place from the Satish Dhawan Space Centre, Sriharikota.")),
    dict(id="tb-010", topic="Economy", kind="news", source="Reuters", domain="reuters.com",
         url="https://www.reuters.com/demo/rbi-keeps-repo-rate-unchanged", published_at="2026-10-08", author="Reuters Staff",
         title="RBI keeps repo rate unchanged at 5.5% as inflation stays within target",
         body=("The Reserve Bank of India kept the repo rate unchanged at 5.5 per cent on 8 October 2026. "
               "Governor Sanjay Malhotra said inflation remained within the target band of 2 to 6 per cent. "
               "The central bank retained its neutral policy stance.")),
    dict(id="tb-011", topic="Economy", kind="official", source="Reserve Bank of India", domain="rbi.org.in",
         url="https://rbi.org.in/demo/monetary-policy-statement-oct-2026", published_at="2026-10-08", author="RBI",
         title="Monetary Policy Statement: policy repo rate unchanged",
         body=("The Monetary Policy Committee decided to keep the policy repo rate unchanged at 5.5 per cent. "
               "The stance remains neutral.")),
    dict(id="tb-012", topic="Politics", kind="news", source="Hindustan Times", domain="hindustantimes.com",
         url="https://www.hindustantimes.com/demo/parliament-session-postponed", published_at="2026-10-18", author="HT Correspondent",
         title="Parliament session postponed by a week, says Lok Sabha secretariat",
         body=("Parliament's winter session has been postponed by a week to 1 December 2026, the Lok Sabha secretariat said. "
               "The secretariat cited pending committee reports.")),
    dict(id="tb-013", topic="Politics", kind="official", source="Press Information Bureau", domain="pib.gov.in",
         url="https://pib.gov.in/demo/winter-session-rescheduled", published_at="2026-10-18", author="PIB Delhi",
         title="Winter session of Parliament rescheduled to 1 December",
         body=("The winter session of Parliament has been rescheduled and will now begin on 1 December 2026.")),
    dict(id="tb-014", topic="World", kind="news", source="The Kathmandu Post", domain="kathmandupost.com",
         url="https://kathmandupost.com/demo/koshi-flood-alert", published_at="2026-10-19", author="Post Report",
         title="Monsoon alert: Koshi river crosses danger level at Chatara",
         body=("The Koshi river crossed the danger level at Chatara on 19 October 2026, the Department of Hydrology and Meteorology said. "
               "Residents of low-lying settlements in Sunsari and Saptari were asked to move to safer places.")),
    dict(id="tb-015", topic="Sports", kind="news", source="Times of India", domain="timesofindia.indiatimes.com",
         url="https://timesofindia.indiatimes.com/demo/india-win-t20-series", published_at="2026-10-14", author="TOI Sports",
         title="India win T20 series against Australia 3-1",
         body=("India won the T20 series against Australia 3-1 after a six-wicket victory in the final match on 14 October 2026. "
               "Captain Suryakumar Yadav scored 68 runs in the decider.")),
    dict(id="tb-016", topic="Sports", kind="news", source="BBC", domain="bbc.com",
         url="https://www.bbc.com/demo/australia-lose-t20-series-india", published_at="2026-10-14", author="BBC Sport",
         title="Australia lose T20 series in India",
         body=("Australia lost the five-match T20 series in India 3-1. India won the final match by six wickets.")),
    dict(id="tb-017", topic="Entertainment", kind="news", source="Hindustan Times", domain="hindustantimes.com",
         url="https://www.hindustantimes.com/demo/mumbai-film-festival-opens", published_at="2026-10-11", author="HT Entertainment",
         title="Mumbai Film Festival opens with 120 films from 40 countries",
         body=("The Mumbai Film Festival opened on 11 October 2026 with 120 films from 40 countries. "
               "Organisers said 35 films will have their world premieres at the festival.")),
    dict(id="tb-018", topic="Opinion", kind="news", source="Indian Express", domain="indianexpress.com",
         url="https://indianexpress.com/demo/opinion-misinformation-elections", published_at="2026-10-09", author="Guest Columnist",
         title="Opinion: Fighting misinformation needs media literacy, not just fact-checks",
         body=("In my view, fact-checks alone cannot stop viral misinformation. "
               "Schools should teach students how to verify sources before sharing. "
               "Platforms must also slow down the forwarding of unverified messages.")),
    dict(id="tb-019", topic="World", kind="news", source="BBC", domain="bbc.com",
         url="https://www.bbc.com/demo/climate-summit-finance-pledge", published_at="2026-10-12", author="BBC Climate Desk",
         title="Climate summit agrees new $300 billion finance pledge",
         body=("Countries at the climate summit agreed a new finance pledge of $300 billion a year by 2035. "
               "Developing nations said the amount fell short of their needs.")),
    dict(id="tb-020", topic="Economy", kind="official", source="Press Information Bureau", domain="pib.gov.in",
         url="https://pib.gov.in/demo/pm-kisan-instalment", published_at="2026-10-05", author="PIB Delhi",
         title="PM-KISAN: eligible small and marginal farmers to receive Rs 6,000 a year",
         body=("Eligible small and marginal farmers will receive Rs 6,000 a year under PM-KISAN in three instalments. "
               "Only registered farmers whose land records are verified qualify for the benefit.")),
    # ---------------------------------------------------------------- second disputed rumour + debunk
    dict(id="tb-021", topic="Tech", kind="viral", source="Viral Khabar (blog)", domain="viral-khabar.blogspot.com",
         url="https://viral-khabar.blogspot.com/demo/isro-alien-signal", published_at="2026-10-12", author="Admin",
         title="SHOCKING: ISRO secretly confirms alien signal from deep space!",
         body=("ISRO has secretly confirmed that it received an alien signal from deep space, according to an anonymous scientist. "
               "The agency is hiding the news from the public. Share before it is deleted!")),
    dict(id="tb-022", topic="Tech", kind="factcheck", source="Press Information Bureau", domain="pib.gov.in",
         url="https://pib.gov.in/demo/factcheck-isro-alien-signal", published_at="2026-10-13", author="PIB Fact Check",
         title="PIB Fact Check: ISRO has not received any alien signal; viral claim is fake",
         body=("A viral blog post claims that ISRO received an alien signal from deep space. "
               "This claim is fake. ISRO has not received any such signal and has issued an official clarification.")),
    # ---------------------------------------------------------------- TC-11 conflicting sources (old blog vs new wire)
    dict(id="tb-023", topic="Economy", kind="blog", source="Karnataka Local Voice (blog)", domain="karnatakalocalvoice.blogspot.com",
         url="https://karnatakalocalvoice.blogspot.com/demo/chitradurga-solar-cancelled", published_at="2025-03-10", author="Blogger",
         title="Chitradurga solar park project cancelled, say locals",
         body=("The Chitradurga solar park project has been cancelled after the state withdrew support, according to local residents. "
               "Work at the site has stopped permanently, the blog claims.")),
    dict(id="tb-024", topic="Economy", kind="news", source="Reuters", domain="reuters.com",
         url="https://www.reuters.com/demo/chitradurga-solar-park-first-phase", published_at="2026-10-10", author="Reuters Staff",
         title="Chitradurga solar park to commission first 300 MW phase in December",
         body=("The Chitradurga solar park project has not been cancelled and will commission its first 300 MW phase in December, Karnataka's energy department said. "
               "Officials said reports of the project being cancelled that circulated last year were inaccurate. "
               "Construction resumed in early 2026 after a land-lease dispute was resolved.")),
    # ---------------------------------------------------------------- TC-12 misleading framing (accurate body)
    dict(id="tb-025", topic="Economy", kind="news", source="Daily Buzz India", domain="dailybuzzindia.net",
         url="https://dailybuzzindia.net/demo/rbi-shocker-savings-danger", published_at="2026-10-09", author="Buzz Desk",
         title="RBI SHOCKER: Your savings are in danger as repo rate stays unchanged!",
         body=("The Reserve Bank of India kept the repo rate unchanged at 5.5 per cent on 8 October 2026. "
               "Governor Sanjay Malhotra said inflation remained within the target band. "
               "The central bank retained its neutral policy stance.")),
    dict(id="tb-026", topic="Economy", kind="news", source="TrendyBits Desk", domain="trendybits.news",
         url="https://trendybits.news/demo/what-we-know-rs-50000-claim", published_at="2026-09-04", author="TrendyBits Desk",
         title="Viral Rs 50,000 allowance post: what we know so far",
         body=("A social media post claiming a Rs 50,000 allowance for every citizen has been shared widely since 2 September 2026. "
               "The Ministry of Finance said on 3 September that no such scheme exists. "
               "Kantipur and Reuters fact-checkers have flagged the post and its photo.")),
]

# Gold credibility bands used by the ablation study (Model 5) — article-level ground truth for the
# demo corpus. low < 40 <= mixed < 65 <= high.
GOLD_CREDIBILITY = {
    "tb-001": "low", "tb-021": "low", "tb-023": "low", "tb-025": "mixed",
    "tb-006": "high", "tb-008": "high", "tb-010": "high", "tb-012": "high", "tb-015": "high", "tb-005": "high",
}
GOLD_MISLEADING = {"tb-001", "tb-025"}

DEFAULT_USERS = [
    dict(user_id="demo", name="Demo Reader", topics=["Tech", "Economy"], sources=["The Hindu", "Reuters"],
         entities=["ISRO", "RBI", "Kerala"], language="en"),
    dict(user_id="guest", name="Guest", topics=[], sources=[], entities=[], language="en"),
]

TICKER = [
    "Parliament session postponed by a week",
    "EC publishes voter roll update",
    "Monsoon alert in Koshi and Kerala",
    "RBI keeps repo rate at 5.5%",
    "Fact-check: 'Rs 50,000 to everyone' claim is false",
]
