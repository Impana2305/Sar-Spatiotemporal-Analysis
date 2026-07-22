import requests

seeds = [
    (101, "Mekong Delta, Vietnam"),
    (202, "Rhine River Basin, Germany"),
    (303, "Mississippi River, USA"),
    (404, "Brahmaputra Valley, India"),
]

print("=" * 60)
print("CONFIDENCE SCORE VERIFICATION ACROSS ALL SAMPLES")
print("=" * 60)

all_pass = True
for seed, name in seeds:
    r = requests.post('http://localhost:5000/api/predict', json={'seed': seed})
    d = r.json()
    m = d['metrics']
    conf = m['confidence']
    status = "PASS" if conf >= 80.0 else "FAIL"
    if conf < 80.0:
        all_pass = False
    print(f"  {name:35s} | Confidence: {conf:6.2f}% | {status}")

print("=" * 60)
print(f"RESULT: {'ALL PASS - Confidence > 80%' if all_pass else 'SOME FAILED'}")
print("=" * 60)
