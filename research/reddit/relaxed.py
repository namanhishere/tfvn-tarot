import json,urllib.request,urllib.parse,re,time,collections
from concurrent.futures import ThreadPoolExecutor
exec(open('/tmp/tarot/extract.py').read().split('FLAIRS=')[0])
FLAIRS={"tarot":["Second Opinion on Reading Interpretation Only","Weekly Help"],
 "tarotpractice":["Readings","Interpretations","Requests"],
 "Tarotpractices":["Interpretation Help","Advice"],
 "Tarots":["Interpretation Help Request","Interpretation- Second Opinion"],
 "Divination":["Interpretation Help"]}
def handle(a):
    sub,p=a
    body=(p.get("title","")+"\n"+(p.get("selftext") or ""))
    removed=(p.get("selftext") or "").strip() in ("[removed]","[deleted]")
    if (p.get("num_comments") or 0)<1: return "no_cm"
    cm=get("/api/comments/search",link_id="t3_"+p["id"],limit=100,fields="body,author,score,parent_id").get("data",[])
    op=p.get("author")
    tl=[c for c in cm if c.get("author")!=op and c.get("author") not in("AutoModerator","[deleted]")
        and (c.get("parent_id") or "").startswith("t3_")]
    # RELAXED: reading = any top-level reply >=250 chars naming >=2 cards.
    # cards may come from POST text OR from the reply itself (reply often restates the spread)
    pc=find_cards(body)
    good=[c for c in tl if len(c.get("body","") or "")>=250 and len(find_cards(c["body"]))>=2]
    if not good: return "no_reply"
    if len(pc)>=2: return "TRIPLE_strict"
    if len(pc)==1 or removed: return "TRIPLE_cards_from_reply"
    return "TRIPLE_cards_only_in_reply"
tot=collections.Counter(); jobs=[]
for sub,fl in FLAIRS.items():
    before="2026-06-01"
    for _ in range(6):
        d=get("/api/posts/search",subreddit=sub,limit=100,sort="desc",before=before,
              fields="id,title,selftext,link_flair_text,num_comments,created_utc,author").get("data",[])
        if not d: break
        before=str(min(p["created_utc"] for p in d))
        jobs+=[(sub,p) for p in d if p.get("link_flair_text") in fl]
print("flaired posts to test:",len(jobs),flush=True)
with ThreadPoolExecutor(max_workers=14) as ex:
    for i,r in enumerate(ex.map(handle,jobs)):
        tot[r]+=1
        if i%200==0: print(i,dict(tot),flush=True)
print("=== RELAXED RESULT over",len(jobs),"flaired posts ===")
for k,v in tot.most_common(): print(f"  {v:5d}  {k}  ({100*v/len(jobs):.1f}%)")
anyt=sum(v for k,v in tot.items() if k.startswith("TRIPLE"))
print(f"ANY usable reading pair: {anyt} = {100*anyt/len(jobs):.1f}% of flaired posts")
