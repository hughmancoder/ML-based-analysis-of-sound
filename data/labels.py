# Canonical IRMAS class order (must match training!)
CLASSES = [
    "cel", "cla", "flu", "gac", "gel",
    "org", "pia", "sax", "tru", "vio", "voi"
]

LABEL_TO_IDX_IRMAS = {
    'cel':0,'cla':1,'flu':2,'gac':3,'gel':4,'org':5,'pia':6,'sax':7,'tru':8,'vio':9,'voi':10
}
IDX_TO_LABEL_IRMAS = {v:k for k,v in LABEL_TO_IDX_IRMAS.items()}


LABEL_TO_IDX_CN = {'guzheng':0, 'suona':1, 'dizi':2, 'gong':3}
IDX_TO_LABEL_CN = {v:k for k,v in LABEL_TO_IDX_CN.items()}