import json,urllib.request,urllib.parse,re,time,sys,collections
from concurrent.futures import ThreadPoolExecutor
BASE="https://arctic-shift.photon-reddit.com"; UA={"User-Agent":"tarot-research/1.0"}
def get(path,**p):
    u=BASE+path+"?"+urllib.parse.urlencode(p)
    for a in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=90) as r: return json.load(r)
        except Exception: time.sleep(1+2*a)
    return {"data":[]}
MAJORS=["the fool","the magician","the high priestess","the empress","the emperor","the hierophant","the lovers",
"the chariot","strength","the hermit","wheel of fortune","justice","the hanged man","death","temperance",
"the devil","the tower","the star","the moon","the sun","judgement","judgment","the world"]
RANKS=["ace","two","three","four","five","six","seven","eight","nine","ten","page","knight","queen","king",
       "2","3","4","5","6","7","8","9","10"]
SUITS=["wands","cups","swords","pentacles","coins","discs","rods","chalices"]
GAZ=MAJORS+[f"{r} of {s}" for r in RANKS for s in SUITS]
GAZ_RE=re.compile("|".join(re.escape(c) for c in sorted(GAZ,key=len,reverse=True)),re.I)
REV=re.compile(r"\b(reversed|rx|inverted|upside[- ]down)\b",re.I)
def find_cards(t):
    out=[];seen=set()
    for m in GAZ_RE.finditer(t or ""):
        c=m.group(0).lower(); r=bool(REV.search(t[m.end():m.end()+22]))
        if (c,r) in seen: continue
        seen.add((c,r)); out.append((c,r))
    return out
QRE=re.compile(r"\?|\b(asked|question|wondering|guidance|advice|does he|will i|should i|what does|help me|pulled|drew|spread|about my|my ex|clarify)\b",re.I)
FLAIRS={"tarot":["Second Opinion on Reading Interpretation Only","Weekly Help"],
 "tarotpractice":["Readings","Interpretations","Requests"],
 "Tarotpractices":["Interpretation Help","Advice"],
 "Tarots":["Interpretation Help Request","Interpretation- Second Opinion"],
 "Divination":["Interpretation Help"]}
stats=collections.Counter(); examples=[]
PAGES=int(sys.argv[1]) if len(sys.argv)>1 else 6
def handle(args):
    sub,p=args
    body=(p.get("title","")+"\n"+(p.get("selftext") or ""))
    if (p.get("selftext") or "").strip() in ("[removed]","[deleted]"): return (sub,"removed",None)
    cards=find_cards(body)
    if len(cards)<2: return (sub,"too_few_cards",None)
    if not QRE.search(body): return (sub,"no_question",None)
    if (p.get("num_comments") or 0)<1: return (sub,"no_comments",None)
    cm=get("/api/comments/search",link_id="t3_"+p["id"],limit=100,fields="body,author,score,parent_id").get("data",[])
    op=p.get("author")
    good=[c for c in cm if c.get("author")!=op and c.get("author") not in ("AutoModerator","[deleted]")
          and len(c.get("body","") or "")>=350 and len(find_cards(c["body"]))>=2
          and (c.get("parent_id") or "").startswith("t3_")]
    if not good: return (sub,"no_good_reply",None)
    good.sort(key=lambda c:-(c.get("score") or 0))
    return (sub,"TRIPLE",{"sub":sub,"post_id":p["id"],"flair":p.get("link_flair_text"),
      "url":f"https://reddit.com/r/{sub}/comments/{p['id']}","question":body[:1500],
      "cards":cards,"n_replies":len(good),"reading":good[0]["body"][:2500],"reading_score":good[0].get("score")})
jobs=[]
for sub,flairs in FLAIRS.items():
    before="2026-06-01"
    for page in range(PAGES):
        d=get("/api/posts/search",subreddit=sub,limit=100,sort="desc",before=before,
              fields="id,title,selftext,link_flair_text,num_comments,created_utc,score,author").get("data",[])
        if not d: break
        before=str(min(p["created_utc"] for p in d))
        stats[f"{sub}/1_scanned"]+=len(d)
        for p in d:
            if p.get("link_flair_text") in flairs:
                stats[f"{sub}/2_flair_match"]+=1; jobs.append((sub,p))
    print("collected",sub,len(jobs),flush=True)
with ThreadPoolExecutor(max_workers=12) as ex:
    for i,(sub,tag,res) in enumerate(ex.map(handle,jobs)):
        stats[f"{sub}/z_{tag}"]+=1
        if tag=="TRIPLE": stats["TOTAL_TRIPLES"]+=1; examples.append(res)
        if i%100==0: print("processed",i,"/",len(jobs),"triples",stats['TOTAL_TRIPLES'],flush=True)
json.dump(examples,open("/tmp/tarot/triples_sample.json","w"),indent=1,ensure_ascii=False)
print("=== FUNNEL ===")
for k in sorted(stats): print(f"  {stats[k]:6d}  {k}")
print(f"\nTriples: {len(examples)}")
