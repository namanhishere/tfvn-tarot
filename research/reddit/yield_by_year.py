import json,urllib.request,urllib.parse,re,time,collections
from concurrent.futures import ThreadPoolExecutor
exec(open('/tmp/tarot/extract.py').read().split('stats=collections.Counter()')[0].split('FLAIRS=')[0])
FLAIRS={"tarot":["Second Opinion on Reading Interpretation Only","Weekly Help"],
 "tarotpractice":["Readings","Interpretations","Requests"],
 "Tarotpractices":["Interpretation Help","Advice"],
 "Tarots":["Interpretation Help Request","Interpretation- Second Opinion"],
 "Divination":["Interpretation Help"]}
def handle(a):
    sub,p=a
    body=(p.get("title","")+"\n"+(p.get("selftext") or ""))
    if (p.get("selftext") or "").strip() in ("[removed]","[deleted]"): return "removed"
    if len(find_cards(body))<2: return "few_cards"
    if not QRE.search(body): return "no_q"
    if (p.get("num_comments") or 0)<1: return "no_cm"
    cm=get("/api/comments/search",link_id="t3_"+p["id"],limit=100,fields="body,author,score,parent_id").get("data",[])
    op=p.get("author")
    good=[c for c in cm if c.get("author")!=op and c.get("author") not in("AutoModerator","[deleted]")
          and len(c.get("body","") or "")>=350 and len(find_cards(c["body"]))>=2
          and (c.get("parent_id") or "").startswith("t3_")]
    return "TRIPLE" if good else "no_reply"
for year in [2019,2021,2022,2023,2024,2025]:
    jobs=[]; scanned=0
    for sub,fl in FLAIRS.items():
        before=f"{year}-12-31"
        for _ in range(3):
            d=get("/api/posts/search",subreddit=sub,limit=100,sort="desc",before=before,after=f"{year}-01-01",
                  fields="id,title,selftext,link_flair_text,num_comments,created_utc,author").get("data",[])
            if not d: break
            before=str(min(p["created_utc"] for p in d)); scanned+=len(d)
            jobs+= [(sub,p) for p in d if p.get("link_flair_text") in fl]
    c=collections.Counter()
    with ThreadPoolExecutor(max_workers=12) as ex:
        for r in ex.map(handle,jobs): c[r]+=1
    fm=len(jobs)
    print(f"{year}: scanned={scanned:5d} flair_match={fm:5d} removed={c['removed']:4d}({100*c['removed']//max(fm,1)}%) "
          f"few_cards={c['few_cards']:4d} no_reply={c['no_reply']:4d} TRIPLES={c['TRIPLE']:4d} "
          f"yield_of_flaired={100*c['TRIPLE']/max(fm,1):.1f}%",flush=True)
