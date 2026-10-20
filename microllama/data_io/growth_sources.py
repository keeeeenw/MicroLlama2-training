"""Pinned source definitions, readable before installing data dependencies."""

SOURCES = {
    'edu': {'repo':'HuggingFaceFW/fineweb-edu','revision':'87f09149ef4734204d70ed1d046ddc9ca3f2b8f9',
            'prefix':'sample/100BT/','weight':.6,'filter':'upstream-fineweb-edu'},
    'dclm': {'repo':'HuggingFaceTB/dclm-edu','revision':'dbad8ad71224482740cd9c9d353591adbf62fe04',
             'prefix':'data/','weight':.4,'filter':'edu_int_score>=3'},
}
