import os
import re
import unicodedata



for k in os.listdir("./"):
	if (not k.endswith(".txt")):
		continue
	with open(k,"rb") as f:
		txt=f.read().decode("unicode_escape")
	txt=str(unicodedata.normalize("NFD",txt).encode("ascii",errors="ignore")).lower()[2:-1].replace("\\n","\n").replace("\\t","\t").replace("\\r","")
	txt=re.sub(r"\t.+(\n|$)",r"\n",txt)
	if (txt[-1]=="\n"):
		txt=txt[:-1]
	with open(k,"w") as f:
		f.write(txt)
